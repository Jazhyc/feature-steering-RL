import torch
from torch import nn
import json
from typing import Literal, List, Any, Optional
from sae_lens import SAE, SAEConfig
from sae_lens.sae import _disable_hooks, get_activation_fn
from pathlib import Path
from safetensors.torch import save_file, load_file
from transformer_lens.hook_points import HookPoint


class SAEAdapter(SAE):
    """
    Extends a frozen SAE with a trainable adapter for feature steering.

    This class adds a parallel adapter network that learns to modulate the SAE's
    feature activations to achieve a downstream objective (e.g., via RL).
    The base SAE and LLM parameters remain frozen.
    """
    def __init__(
        self,
        cfg: SAEConfig,
        use_error_term: bool = True,
        adapter_layers: Optional[List[int]] = None,
        fusion_mode: Literal["additive", "multiplicative"] = "additive"
    ):
        """
        Initializes the SAEAdapter.

        Args:
            cfg: Configuration for the base SAE.
            use_error_term: If True, adds the SAE's reconstruction error to the output.
            adapter_layers: Hidden layer sizes for an MLP adapter. If None,
                            defaults to a single linear layer with a ReLU activation.
            fusion_mode: How to combine features and steering vector
                         ('additive' or 'multiplicative').
        """
        super(SAEAdapter, self).__init__(cfg, use_error_term=use_error_term)

        for param in self.parameters():
            param.requires_grad = False

        self.fusion_mode = fusion_mode
        assert self.fusion_mode in ["additive", "multiplicative"]
        
        # The adapter uses the same activation function as the base SAE.
        self.adapter_activation = get_activation_fn(
            cfg.activation_fn_str, **cfg.activation_fn_kwargs
        )
        
        # Define the adapter's layers, defaulting to a single linear layer 
        self.adapter_layers = nn.ModuleList()
        layer_dims = [self.cfg.d_in] + (adapter_layers or []) + [self.cfg.d_sae]
        for i in range(len(layer_dims) - 1):
            self.adapter_layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
        
        self._initialize_adapter_weights()
        self.adapter_layers.to(self.device, self.dtype)
        self.adapter_activation.to(self.device, self.dtype)
        
        # Create hook points for the adapter to cache activations
        self.hook_sae_adapter_pre = HookPoint()
        self.hook_sae_adapter_post = HookPoint()
        self.hook_sae_fusion = HookPoint()
        
        # Add hooks to the hook manager
        self.setup()
        
    def _initialize_adapter_weights(self):
        """
        Applies Kaiming uniform initialization to the adapter's layers.
        The final layer is initialized to zero to ensure it acts as an identity initially
        """
        for layer in self.adapter_layers:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
                
        # Ensure the final layer is initialized close to zero (identity)
        # Maybe use a constant later?
        final_layer = self.adapter_layers[-1]
        nn.init.normal_(final_layer.weight, mean=0.0, std=0.0001)
        if final_layer.bias is not None:
            nn.init.zeros_(final_layer.bias)

    def get_steering_vector(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the steering vector from the adapter for a given input.

        Args:
            x: The input activation vector (e.g., from the LLM's residual stream).

        Returns:
            The computed steering vector.
        """
        adapter_in = x.to(self.dtype)
        
        # Pass through hidden layers (if any) with ReLU activations
        for layer in self.adapter_layers[:-1]:
            adapter_in = self.adapter_activation(layer(adapter_in))
            
        # Process through the final layer to get pre-activations
        pre_act = self.adapter_layers[-1](adapter_in)
        
        # We need the LHS in case of interventions
        pre_act = self.hook_sae_adapter_pre(pre_act)
        
        # Apply final activation and post-activation hook
        post_act = self.adapter_activation(pre_act)
        post_act = self.hook_sae_adapter_post(post_act)
        
        return post_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: modulates SAE features with the adapter's steering vector."""
        
        # SAE Path (Frozen), we reuse variables for memory efficiency
        # Feature vector is very large
        feature_acts = self.encode(x)
        
        # Compute Error for clean intervention
        # Error based on clean (regular path)
        if self.use_error_term:
            with _disable_hooks(self):
                reconstruct_clean = self.decode(feature_acts)
            sae_error = self.hook_sae_error(x - reconstruct_clean.detach())
        
        # Adapter Path (Trainable)
        steering_vector = self.get_steering_vector(x)
        
        # Fusion
        if self.fusion_mode == "multiplicative":
            feature_acts = feature_acts * (1 + steering_vector) # Ensures that an output of 0 is identity
        else:
            feature_acts = feature_acts + steering_vector
        feature_acts = self.hook_sae_fusion(feature_acts)
            
        # Decode
        sae_out = self.decode(feature_acts)
        
        if self.use_error_term:
            sae_out = sae_out + sae_error
            
        return self.hook_sae_output(sae_out)

    @classmethod
    def from_pretrained(
        cls,
        release: str,
        sae_id: str,
        device: str = "cpu",
        force_download: bool = False,
        **adapter_kwargs: Any
    ) -> tuple["SAEAdapter", dict[str, Any], torch.Tensor | None]:
        """Loads a pretrained SAE and wraps it in an SAEAdapter."""
        
        sae_temp, cfg_dict, sparsity = SAE.from_pretrained(
            release, sae_id, device=device, force_download=force_download
        )
        state_dict = sae_temp.state_dict()
        del sae_temp

        sae_adapter_instance = cls(
            cfg=SAEConfig.from_dict(cfg_dict), 
            **adapter_kwargs
        )
        
        sae_adapter_instance.load_state_dict(state_dict, strict=False)
        
        return sae_adapter_instance, cfg_dict, sparsity

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Returns the adapter's trainable parameters for an optimizer."""
        return list(self.adapter_layers.parameters())
    
    def save_model(self, path: str | Path):
        """Saves the adapter's state_dict and JSON config to a directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save weights
        weights_path = path / "sae_weights.safetensors"
        save_file(self.state_dict(), weights_path)

        # Create and save augmented config
        config = self.cfg.to_dict()
        config["fusion_mode"] = self.fusion_mode
        config["adapter_layers"] = [
            layer.out_features for layer in self.adapter_layers[:-1]
        ]
        
        config_path = path / "sae_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
            
        print(f"SAEAdapter saved to {path}")
        
    @classmethod
    def from_disk(cls, path: str | Path, device: str = "cpu") -> "SAEAdapter":
        """Loads a saved SAEAdapter from a directory."""
        path = Path(path)
        
        # Load config and extract adapter-specific arguments
        with open(path / "sae_config.json", 'r') as f:
            config = json.load(f)
            
        adapter_kwargs = {
            "fusion_mode": config.pop("fusion_mode", "additive"),
            "adapter_layers": config.pop("adapter_layers", None),
        }
        config['device'] = device

        # Create instance from config
        instance = cls(cfg=SAEConfig.from_dict(config), **adapter_kwargs)
        
        # Load weights
        state_dict = load_file(path / "sae_weights.safetensors", device=device)
        instance.load_state_dict(state_dict)
        
        print(f"SAEAdapter loaded from {path}")
        return instance