import torch
from torch import nn
import json
from typing import Any, Optional
from sae_lens import SAE, SAEConfig
from pathlib import Path
from safetensors.torch import save_file, load_file
from transformer_lens.hook_points import HookPoint
from contextlib import contextmanager
from torch.nn import functional as F
from .jump_relu import JumpReLU, Step

CUSTOM_CONFIG_NAME = 'fsrl_adapter_config.json'

class SAEAdapter(SAE):
    """
    A steering adapter that uses a frozen, pretrained SAE.

    This module contains a frozen SAE and a separate, trainable adapter network.
    It intervenes by adding a learned steering vector to the SAE's feature
    activations, preserving the interpretability of the original SAE features.
    """
    def __init__(
        self,
        cfg: SAEConfig,
        use_error_term: bool = True,
        **kwargs 
    ):
        """
        Initializes the SAEAdapter.

        Args:
            cfg: Configuration for the base SAE.
            use_error_term: If True, adds the SAE's reconstruction error to the output.
        """
        super(SAEAdapter, self).__init__(cfg, use_error_term=use_error_term)

        # Freeze the base SAE parameters
        for param in self.parameters():
            param.requires_grad = False
        
        self.adapter_linear = nn.Linear(self.cfg.d_in, self.cfg.d_sae, bias=True)
        
        # Activation function configuration
        self.use_jump_relu = kwargs.get("use_jump_relu", False)
        
        # JumpReLU specific parameters (only used if use_jump_relu is True)
        if self.use_jump_relu:
            initial_threshold = kwargs.get("initial_threshold", 0.001)
            self.bandwidth = kwargs.get("bandwidth", 0.001)
            self.log_threshold = nn.Parameter(
                torch.full((self.cfg.d_sae,), torch.log(torch.tensor(initial_threshold)), dtype=torch.float32)
            )
        else:
            self.bandwidth = None
            self.log_threshold = None

        self._initialize_adapter()
        self.to(self.device, self.dtype)
        
        # Ensure log_threshold stays in fp32 for numerical stability (only if using JumpReLU)
        if self.use_jump_relu:
            self.log_threshold.data = self.log_threshold.data.to(torch.float32)

        # Registers for logging and auxiliary losses
        self.register_buffer('_norm_l0', torch.tensor(0.0))
        self.register_buffer('_norm_l1', torch.tensor(0.0))
        self.register_buffer('_norm_l2', torch.tensor(0.0))

        # Create hook points for the adapter to cache activations or for interventions
        self.hook_sae_adapter = HookPoint()
        self.hook_sae_fusion = HookPoint()

        # Add hooks to the hook manager
        self.setup()

        # Steering sparsity control (fraction of features used in steering)
        # 1.0 means use all features; 0.1 means keep top 10% by |value| per item
        self.steering_fraction = 1.0
        
    @property
    def adapter_threshold(self) -> torch.Tensor:
        """Computes the threshold from the learnable log_threshold (only for JumpReLU)."""
        if self.use_jump_relu:
            return torch.exp(self.log_threshold)
        else:
            return None
    
    def to(self, *args, **kwargs):
        """Override to method to keep log_threshold in fp32 (only if using JumpReLU)."""
        # Store the original log_threshold to avoid dtype conversion
        if hasattr(self, 'log_threshold') and self.log_threshold is not None:
            original_log_threshold = self.log_threshold.data.clone()
        
        # Call parent's to method
        result = super().to(*args, **kwargs)
        
        # Restore log_threshold to fp32 without any intermediate conversions
        if hasattr(self, 'log_threshold') and self.log_threshold is not None:
            device = self.adapter_linear.weight.device
            self.log_threshold.data = original_log_threshold.to(device)

        return result
        
    def _initialize_adapter(self):
        """
        A simple initialization for the adapter weights and bias.
        For JumpReLU: biases are initialized to the initial threshold
        For ReLU: biases are initialized to zero
        """
        nn.init.uniform_(self.adapter_linear.weight, a=-1e-9, b=1e-9)

        if self.use_jump_relu:
            # Use the threshold value but ensure it matches the bias dtype
            threshold_mean = self.adapter_threshold.mean().item()
            nn.init.constant_(self.adapter_linear.bias, threshold_mean)
        else:
            # Initialize bias to zero for regular ReLU
            nn.init.constant_(self.adapter_linear.bias, 0.0)

    def get_steering_vector(self, adapter_input: torch.Tensor) -> torch.Tensor:
        """Computes the steering vector from the trainable adapter."""
        
        # Ensures adapter receives same input as regular SAE
        adapter_input = adapter_input - (self.b_dec * self.cfg.apply_b_dec_to_input)
        
        # Get pre-activations from linear layer
        pre_activations = self.adapter_linear(adapter_input.to(self.dtype))
        
        # Apply relu first (common to both activation functions)
        pre_activations = F.relu(pre_activations)

        if self.use_jump_relu:
            # JumpReLU path - use threshold and bandwidth
            threshold_tensor = self.adapter_threshold.to(pre_activations.dtype)
            bandwidth_tensor = torch.tensor(self.bandwidth, dtype=pre_activations.dtype, device=pre_activations.device)

            steered_activations = JumpReLU.apply(pre_activations, threshold_tensor, bandwidth_tensor)
            sparsity_mask = Step.apply(pre_activations, threshold_tensor, bandwidth_tensor)
        else:
            # Regular ReLU path - activations are already post-ReLU
            steered_activations = pre_activations
            
            # For regular ReLU, L0 norm is calculated from non-zero steered activations
            sparsity_mask = (steered_activations > 0).float()
            
        # Optionally restrict to top-k features by absolute value (per item)
        if 0.0 < getattr(self, "steering_fraction", 1.0) < 1.0:
            d = steered_activations.shape[-1]
            k = max(1, int(d * float(self.steering_fraction)))
            # Compute top-k indices by absolute value
            with torch.no_grad():
                topk = torch.topk(steered_activations.abs(), k=k, dim=-1, largest=True, sorted=False)
                mask = torch.zeros_like(steered_activations, dtype=steered_activations.dtype)
                mask.scatter_(-1, topk.indices, 1.0)
            steered_activations = steered_activations * mask
            sparsity_mask = mask

        # L0 is count of non-zero active features after any masking
        self._norm_l0 = torch.sum(sparsity_mask, dim=-1).mean()

        steered_activations = self.hook_sae_adapter(steered_activations)

        # Statistics for loss and logging
        # L1 and L2 norms: mean of norms across batch (sum over features, mean over batch)
        self._norm_l1 = torch.mean(torch.sum(torch.abs(steered_activations), dim=-1))
        self._norm_l2 = torch.mean(torch.sum(steered_activations**2, dim=-1))
        
        return steered_activations

    def _forward_no_checkpoint(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: modulates SAE features with the adapter's steering vector."""
        
        # SAE Path (Frozen)
        # The feature vector can be very large
        feature_acts = self.encode(x)
        
        # Adapter Path (Trainable)
        steering_vector = self.get_steering_vector(x)
        
        # Compute Error for clean intervention
        # Error is based on the clean, unsteered path
        if self.use_error_term:
            with torch.no_grad():
                with _disable_hooks(self):
                    reconstruct_clean = self.decode(feature_acts)
                sae_error = self.hook_sae_error(x - reconstruct_clean)

        fused_feature_acts = feature_acts + steering_vector
        fused_feature_acts = self.hook_sae_fusion(fused_feature_acts)

        # Decode the modulated features back into the residual stream
        sae_out = self.decode(fused_feature_acts)
        
        if self.use_error_term:
            sae_out.add_(sae_error)
            
        return self.hook_sae_output(sae_out)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional gradient checkpointing.
        Modulates SAE features with the adapter's steering vector.
        """
        return self._forward_no_checkpoint(x)

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

        instance.to(dtype=torch.bfloat16)

        # Manually add the release info to the config for later saving.
        instance.cfg.release = release
        instance.cfg.sae_id = sae_id
        
        # Load the state dict of the base SAE; `strict=False` ignores missing adapter keys
        instance.load_state_dict(sae_temp.state_dict(), strict=False)
        
        return instance, cfg_dict, sparsity

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Returns the adapter's trainable parameters for an optimizer."""
        params = [self.adapter_linear.weight, self.adapter_linear.bias]
        if self.use_jump_relu:
            params.append(self.log_threshold)
        return params

    def get_norms(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the current L0, L1, and L2 norms."""
        return self._norm_l0, self._norm_l1, self._norm_l2

    # --- Steering control API ---
    def set_steering_fraction(self, fraction: float) -> None:
        """Set fraction of features to keep in the steering vector (0-1]."""
        if not (0.0 < fraction <= 1.0):
            raise ValueError("steering_fraction must be in (0, 1].")
        self.steering_fraction = float(fraction)
    
    def save_adapter(self, path: str | Path):
        """Saves only the trainable adapter weights and its configuration."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Get adapter state dict - the to() method already handles dtype preservation
        adapter_state_dict = {
            'adapter_linear.weight': self.adapter_linear.weight,
            'adapter_linear.bias': self.adapter_linear.bias,
        }
        
        # Only save log_threshold if using JumpReLU
        if self.use_jump_relu:
            adapter_state_dict['log_threshold'] = self.log_threshold
        
        save_file(adapter_state_dict, path / "adapter_weights.safetensors")

        adapter_config = {
            "base_sae_release": self.cfg.release,
            "base_sae_id": self.cfg.sae_id,
            "use_jump_relu": self.use_jump_relu,
        }
        
        # Save JumpReLU specific config if used
        if self.use_jump_relu:
            adapter_config["bandwidth"] = self.bandwidth
            
        with open(path / CUSTOM_CONFIG_NAME, 'w') as f:
            json.dump(adapter_config, f, indent=4)
        
        print(f"Trainable adapter saved to {path}")

    @classmethod
    def load_from_pretrained_adapter(
        cls,
        path: str | Path,
        device: str = "cpu",
        force_download: bool = False,
    ) -> "SAEAdapter":
        """Loads a base SAE from the hub and applies local adapter weights."""
        path = Path(path)
        
        with open(path / CUSTOM_CONFIG_NAME, 'r') as f:
            config = json.load(f)

        # Extract adapter kwargs from config
        adapter_kwargs = {
            "use_jump_relu": config.get("use_jump_relu", False)
        }
        
        if adapter_kwargs["use_jump_relu"]:
            adapter_kwargs["bandwidth"] = config.get("bandwidth", 0.001)
            adapter_kwargs["initial_threshold"] = config.get("initial_threshold", 0.001)

        # Create the full model instance by calling our own from_pretrained
        instance, _, _ = cls.from_pretrained(
            release=config["base_sae_release"],
            sae_id=config["base_sae_id"],
            device=device,
            force_download=force_download,
            **adapter_kwargs
        )

        # Load the locally saved adapter weights
        state_dict = load_file(path / "adapter_weights.safetensors", device=device)
        
        # Load weights normally - the to() method will handle dtype preservation
        instance.adapter_linear.weight.data = state_dict['adapter_linear.weight']
        instance.adapter_linear.bias.data = state_dict['adapter_linear.bias']
        
        # Only load log_threshold if it exists in the state dict (for JumpReLU)
        if 'log_threshold' in state_dict:
            instance.log_threshold.data = state_dict['log_threshold']
        
        print(f"Adapter loaded from {path}")
        return instance

_blank_hook = nn.Identity()
@contextmanager
def _disable_hooks(sae):
    """
    Temporarily disable hooks for the SAE. Swaps out all the hooks with a fake modules that does nothing.
    """
    try:
        for hook_name in sae.hook_dict:
            setattr(sae, hook_name, _blank_hook)
        yield
    finally:
        for hook_name, hook in sae.hook_dict.items():
            setattr(sae, hook_name, hook)