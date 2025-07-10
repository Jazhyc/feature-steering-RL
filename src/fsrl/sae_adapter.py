import torch
from torch import nn
from typing import Literal, List, Dict, Any, Optional

from sae_lens import SAE, SAEConfig
from sae_lens.sae import _disable_hooks

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
                            defaults to a single linear layer.
            fusion_mode: How to combine features and steering vector
                         ('additive' or 'multiplicative').
        """
        # Initialize the parent SAE
        super(SAEAdapter, self).__init__(cfg, use_error_term=use_error_term)

        # Freeze the base SAE parameters
        for param in self.parameters():
            param.requires_grad = False

        self.fusion_mode = fusion_mode
        assert self.fusion_mode in ["additive", "multiplicative"]

        # Define the trainable adapter network
        if adapter_layers is None or not adapter_layers:
            # Default: a single linear layer
            self.adapter = nn.Linear(self.cfg.d_in, self.cfg.d_sae)
        else:
            # Optional: a deeper MLP
            adapter_mlp_layers = []
            current_dim = self.cfg.d_in
            for hidden_dim in adapter_layers:
                adapter_mlp_layers.append(nn.Linear(current_dim, hidden_dim))
                adapter_mlp_layers.append(nn.ReLU())
                current_dim = hidden_dim
            adapter_mlp_layers.append(nn.Linear(current_dim, self.cfg.d_sae))
            self.adapter = nn.Sequential(*adapter_mlp_layers)
            
        # Ensure adapter is on the correct device and dtype
        self.adapter.to(self.device, self.dtype)
        self._initialize_adapter_weights()
        
        # Create an extra hook point for the adapter output
        self.hook_sae_adapter = HookPoint()
        
    def _initialize_adapter_weights(self):
        """
        Applies Kaiming uniform initialization to the adapter's linear layers
        and zeros to its biases, mirroring the base SAE's initialization.
        """
        
        if isinstance(self.adapter, nn.Sequential):
            # Handle MLP case
            for module in self.adapter:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        elif isinstance(self.adapter, nn.Linear):
            # Handle single linear layer case (which does not have an activation function)
            nn.init.kaiming_uniform_(self.adapter.weight)
            if self.adapter.bias is not None:
                nn.init.zeros_(self.adapter.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: modulates SAE features with the adapter's steering vector."""
        
        # SAE Path (Frozen): Get original feature activations
        feature_acts = self.encode(x)
        
        # Adapter Path (Trainable): Get steering vector
        steering_vector = self.adapter(x.to(self.dtype))
        self.hook_sae_adapter(steering_vector)
        
        # Fusion: Modulate features
        if self.fusion_mode == "multiplicative":
            modulated_acts = feature_acts * steering_vector
        else:
            modulated_acts = feature_acts + steering_vector
            
        # Decode the modulated features
        sae_out = self.decode(modulated_acts)
        
        # Readds the clean SAE reconstruction error
        # In the original implementation, this would yield an identity function which can then be manipulated using hooks
        # In our case, the adapter itself manipulates the features, so we do need to manipulate the SAE afterwards using hooks
        with torch.no_grad():
            with _disable_hooks(self):
                feature_acts_clean = self.encode(x)
                x_reconstruct_clean = self.decode(feature_acts_clean)
            sae_error = self.hook_sae_error(x - x_reconstruct_clean)
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
        
        # Temporarily load original SAE to get its config and weights
        sae_temp, cfg_dict, sparsity = SAE.from_pretrained(
            release, sae_id, device=device, force_download=force_download
        )
        state_dict = sae_temp.state_dict()
        del sae_temp # free memory

        # Instantiate our adapter class with the loaded config
        sae_adapter_instance = cls(
            cfg=SAEConfig.from_dict(cfg_dict), 
            **adapter_kwargs
        )
        
        # Load the frozen weights, ignoring missing adapter keys
        sae_adapter_instance.load_state_dict(state_dict, strict=False)
        
        return sae_adapter_instance, cfg_dict, sparsity

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Returns the adapter's trainable parameters for an optimizer."""
        return list(self.adapter.parameters())