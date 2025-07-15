import torch
from torch import nn
import json
from typing import Literal, Any, Optional
from sae_lens import SAE, SAEConfig
from sae_lens.sae import _disable_hooks, get_activation_fn
from pathlib import Path
from safetensors.torch import save_file, load_file
from transformer_lens.hook_points import HookPoint

# New imports for LoRA
from peft import get_peft_model, LoraConfig, PeftModel
from peft.utils import set_peft_model_state_dict

CUSTOM_CONFIG_NAME = 'fsrl_adapter_config.json'

class SAEAdapter(SAE):
    """
    Extends a frozen SAE with a trainable adapter for feature steering.

    This class adds a parallel adapter network that learns to modulate the SAE's
    feature activations to achieve a downstream objective (e.g., via RL).
    The base SAE and LLM parameters remain frozen.

    This version uses a single linear layer for the adapter and includes optional
    LoRA support for memory efficiency.
    """
    def __init__(
        self,
        cfg: SAEConfig,
        use_error_term: bool = True,
        fusion_mode: Literal["additive", "multiplicative"] = "additive",
        use_lora_adapter: bool = False,
        lora_rank: int = 16,
        lora_alpha: int = 32,
    ):
        """
        Initializes the SAEAdapter.

        Args:
            cfg: Configuration for the base SAE.
            use_error_term: If True, adds the SAE's reconstruction error to the output.
            fusion_mode: How to combine features and steering vector
                         ('additive' or 'multiplicative').
            use_lora_adapter: If True, the adapter's linear layer will
                              be replaced with a LoRA layer for memory efficiency.
            lora_rank: The rank 'r' for the LoRA decomposition.
            lora_alpha: The alpha scaling parameter for LoRA.
        """
        super(SAEAdapter, self).__init__(cfg, use_error_term=use_error_term)

        # Freeze the base SAE parameters
        for param in self.parameters():
            param.requires_grad = False

        self.fusion_mode = fusion_mode
        assert self.fusion_mode in ["additive", "multiplicative"]
        
        # Define the adapter as a single linear layer.
        self.adapter = nn.Sequential(nn.Linear(self.cfg.d_in, self.cfg.d_sae))
        self._initialize_adapter_weights(use_lora_adapter)
        
        self.use_lora_adapter = use_lora_adapter
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        if self.use_lora_adapter:
            
            # If we are creating a LoRA adapter from scratch, we want the base
            # linear layer to be zero-initialized before PEFT wraps it.
            self._initialize_adapter_weights(is_lora_base=True)
            
            # Apply LoRA to the adapter's linear layer for memory-efficient training
            lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                target_modules=["0"], # Target the first (and only) layer in the nn.Sequential
                bias="none",
                init_lora_weights=True, # This forces B to be 0
            )
            self.adapter = get_peft_model(self.adapter, lora_config)
            self.adapter.print_trainable_parameters()
            
        else:
            self._initialize_adapter_weights(is_lora_base=False)
        
        self.adapter.to(self.device, self.dtype)
        
        # Create hook points for the adapter to cache activations or for interventions
        self.hook_sae_adapter = HookPoint()
        self.hook_sae_fusion = HookPoint()
        
        # Add hooks to the hook manager
        self.setup()
        
    def _initialize_adapter_weights(self, is_lora_base: bool):
        """
        Initializes the base linear adapter layer.

        Args:
            is_lora_base: If True, the layer is a base for a LoRA adapter and will be
                        zero-initialized. Otherwise, it's a full-rank adapter and will
                        be initialized with small noise.
        """
        layer = self.adapter[0]
        
        if is_lora_base:
            # Zero-initialize the base layer when LoRA will be applied on top.
            nn.init.zeros_(layer.weight)
        else:
            # Initialize with small noise for a full-rank adapter.
            nn.init.normal_(layer.weight, mean=0.0, std=1e-5)
            
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def get_steering_vector(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the steering vector from the adapter for a given input.

        Args:
            x: The input activation vector (e.g., from the LLM's residual stream).

        Returns:
            The computed steering vector.
        """
        # Get pre-activations from the adapter (works for both full-rank and LoRA)
        act = self.adapter(x.to(self.dtype))
        act = self.hook_sae_adapter(act)
        return act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: modulates SAE features with the adapter's steering vector."""
        
        # SAE Path (Frozen)
        # The feature vector can be very large
        feature_acts = self.encode(x)
        
        # Compute Error for clean intervention
        # Error is based on the clean, unsteered path
        if self.use_error_term:
            with torch.no_grad():
                with _disable_hooks(self):
                    reconstruct_clean = self.decode(feature_acts)
                sae_error = self.hook_sae_error(x - reconstruct_clean.detach())
        
        # Adapter Path (Trainable)
        steering_vector = self.get_steering_vector(x)
        
        # Fusion of SAE features and steering vector
        if self.fusion_mode == "multiplicative":
            feature_acts = feature_acts * (1 + steering_vector) # Ensures an output of 0 is identity
        else: # "additive"
            feature_acts = feature_acts + steering_vector
        feature_acts = self.hook_sae_fusion(feature_acts)
            
        # Decode the modulated features back into the residual stream
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
    ) -> tuple["SAEAdapter", dict[str, Any], Optional[torch.Tensor]]:
        """Loads a pretrained SAE and wraps it in an SAEAdapter."""
        
        sae_temp, cfg_dict, sparsity = SAE.from_pretrained(
            release, sae_id, device=device, force_download=force_download
        )

        instance = cls(
            cfg=SAEConfig.from_dict(cfg_dict), 
            **adapter_kwargs
        )
        
        # Manually add the release info to the config for later saving.
        instance.cfg.release = release
        instance.cfg.sae_id = sae_id
        
        # Load the state dict of the base SAE; `strict=False` ignores missing adapter keys
        instance.load_state_dict(sae_temp.state_dict(), strict=False)
        
        return instance, cfg_dict, sparsity

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Returns the adapter's trainable parameters for an optimizer."""
        return list(self.adapter.parameters())
    
    def save_adapter(self, path: str | Path):
        """Saves only the adapter's weights and its configuration."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # PEFT models have their own saving method that only saves the LoRA weights
        if self.use_lora_adapter:
            self.adapter.save_pretrained(path)
            
            # Delete readme.md file that PEFT adds (maybe check for a better way)
            readme_path = path / "README.md"
            if readme_path.exists():
                readme_path.unlink()
            
        else:
            save_file(self.adapter.state_dict(), path / "adapter_weights.safetensors")

        adapter_config = {
            "base_sae_release": self.cfg.release,
            "base_sae_id": self.cfg.sae_id,
            "fusion_mode": self.fusion_mode,
            "use_lora_adapter": self.use_lora_adapter,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "sae_config": self.cfg.to_dict()
        }
        with open(path / CUSTOM_CONFIG_NAME, 'w') as f:
            json.dump(adapter_config, f, indent=4)
        
        print(f"Adapter saved to {path}")

    @classmethod
    def load_from_pretrained_adapter(
        cls,
        path: str | Path,
        device: str = "cpu",
        force_download: bool = False,
    ) -> "SAEAdapter":
        """Loads a base SAE from the hub and applies local adapter weights."""
        path = Path(path)
        
        CUSTOM_CONFIG_NAME = "fsrl_adapter_config.json"
        # Load the adapter's configuration
        with open(path / CUSTOM_CONFIG_NAME, 'r') as f:
            config = json.load(f)

        # Filter the loaded config to only include kwargs valid for SAEAdapter.__init__
        valid_init_keys = ["fusion_mode", "use_lora_adapter", "lora_rank", "lora_alpha", "use_error_term"]
        adapter_init_args = {k: v for k, v in config.items() if k in valid_init_keys}

        # When loading a LoRA adapter, we create a plain one first and let PEFT wrap it.
        is_lora_model = adapter_init_args.get("use_lora_adapter", False)

        # Create the full model instance by downloading the base SAE
        instance, _, _ = cls.from_pretrained(
            release=config["base_sae_release"],
            sae_id=config["base_sae_id"],
            device=device,
            force_download=force_download,
            **adapter_init_args
        )

        # Load the locally saved adapter weights
        if is_lora_model:
            # adapter_model is hardcoded by PEFT?
            adapter_weights = load_file(path / "adapter_model.safetensors", device=device)
            set_peft_model_state_dict(instance.adapter, adapter_weights)
        else:
            state_dict = load_file(path / "adapter_weights.safetensors", device=device)
            instance.adapter.load_state_dict(state_dict)
        
        print(f"Adapter loaded from {path}")
        return instance