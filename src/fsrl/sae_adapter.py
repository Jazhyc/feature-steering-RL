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
        
        initial_threshold = kwargs.get("initial_threshold", 0.01)
        self.bandwidth = kwargs.get("bandwidth", 0.05)
        self.log_threshold = nn.Parameter(
            torch.full((self.cfg.d_sae,), torch.log(torch.tensor(initial_threshold)), dtype=torch.float32)
        )

        self._initialize_adapter()
        self.to(self.device, self.dtype)
        
        # Ensure log_threshold stays in fp32 for numerical stability
        self.log_threshold.data = self.log_threshold.data.to(torch.float32)

        # Instance variables for logging
        self._current_steering_l0_norm = torch.tensor(0.0)
        self._current_steering_l1_norm = torch.tensor(0.0)
        self._current_steering_l2_norm = torch.tensor(0.0)
        
        # Create hook points for the adapter to cache activations or for interventions
        self.hook_sae_adapter = HookPoint()
        self.hook_sae_fusion = HookPoint()
        
        # Add hooks to the hook manager
        self.setup()
        
    @property
    def threshold(self) -> torch.Tensor:
        """Computes the threshold from the learnable log_threshold."""
        return torch.exp(self.log_threshold)
    
    def to(self, *args, **kwargs):
        """Override to method to keep log_threshold in fp32."""
        # Store the original log_threshold to avoid dtype conversion
        if hasattr(self, 'log_threshold'):
            original_log_threshold = self.log_threshold.data.clone()
        
        # Call parent's to method
        result = super().to(*args, **kwargs)
        
        # Restore log_threshold to fp32 without any intermediate conversions
        if hasattr(self, 'log_threshold'):
            self.log_threshold.data = original_log_threshold.to(torch.float32)
        
        return result
        
    def _initialize_adapter(self):
        """
        A simple initialization for the adapter weights and bias.
        Set such that all biases are initialized to the initial threshold
        """
        nn.init.normal_(self.adapter_linear.weight, mean=0.0, std=1e-6)
        # Use the threshold value but ensure it matches the bias dtype
        threshold_mean = self.threshold.mean().item()
        nn.init.constant_(self.adapter_linear.bias, threshold_mean)

    def get_steering_vector(self, adapter_input: torch.Tensor) -> torch.Tensor:
        """Computes the steering vector from the trainable adapter."""
        
        # Ensures adapter receives same input as regular SAE
        adapter_input = adapter_input - (self.b_dec * self.cfg.apply_b_dec_to_input)
        
        # JumpReLU now returns both the final activations and the sparsity mask
        pre_activations = self.adapter_linear(adapter_input.to(self.dtype))

        # Convert threshold to match pre_activations dtype for computation
        threshold_tensor = self.threshold.to(pre_activations.dtype)
        bandwidth_tensor = torch.tensor(self.bandwidth, dtype=pre_activations.dtype, device=pre_activations.device)

        steered_activations = JumpReLU.apply(pre_activations, threshold_tensor, bandwidth_tensor)
        steered_activations = self.hook_sae_adapter(steered_activations)
        
        sparsity_mask = Step.apply(pre_activations, threshold_tensor, bandwidth_tensor)

        # L0 norm is the mean ratio of active features per example in the batch
        self._current_steering_l0_norm = torch.sum(sparsity_mask, dim=-1).mean() / self.cfg.d_sae

        # Statistics for loss and logging
        # L1 and L2 norms: mean of norms across batch (sum over features, mean over batch)
        self._current_steering_l1_norm = torch.mean(torch.sum(torch.abs(steered_activations), dim=-1))
        self._current_steering_l2_norm = torch.mean(torch.sum(steered_activations**2, dim=-1))
        
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
        
        # Manually add the release info to the config for later saving.
        instance.cfg.release = release
        instance.cfg.sae_id = sae_id
        
        # Load the state dict of the base SAE; `strict=False` ignores missing adapter keys
        instance.load_state_dict(sae_temp.state_dict(), strict=False)
        
        return instance, cfg_dict, sparsity

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Returns the adapter's trainable parameters for an optimizer."""
        return [self.adapter_linear.weight, self.adapter_linear.bias, self.log_threshold]
    
    def get_steering_l1_norm(self) -> torch.Tensor:
        """
        Returns the L1 norm (sum of absolute values) of the steering vector from the most recent forward pass.
        This value can be used both for logging and as the L1 penalty in the loss function.
        """
        return self._current_steering_l1_norm
    
    def get_steering_l2_norm(self) -> torch.Tensor:
        """
        Returns the L2 norm (sum of squared values) of the steering vector from the most recent forward pass.
        This value can be used for L2 regularization or logging.
        """
        return self._current_steering_l2_norm
    
    def get_steering_l0_norm(self) -> torch.Tensor:
        """
        Returns the L0 norm (sparsity) of the steering vector from the most recent forward pass.
        This represents the fraction of non-zero activations in the steering vector.
        """
        return self._current_steering_l0_norm
    
    def save_adapter(self, path: str | Path):
        """Saves only the trainable adapter weights and its configuration."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Get adapter state dict - the to() method already handles dtype preservation
        adapter_state_dict = {
            'adapter_linear.weight': self.adapter_linear.weight,
            'adapter_linear.bias': self.adapter_linear.bias,
            'log_threshold': self.log_threshold
        }
        
        save_file(adapter_state_dict, path / "adapter_weights.safetensors")

        adapter_config = {
            "base_sae_release": self.cfg.release,
            "base_sae_id": self.cfg.sae_id,
        }
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

        # Create the full model instance by calling our own from_pretrained
        instance, _, _ = cls.from_pretrained(
            release=config["base_sae_release"],
            sae_id=config["base_sae_id"],
            device=device,
            force_download=force_download,
        )

        # Load the locally saved adapter weights
        state_dict = load_file(path / "adapter_weights.safetensors", device=device)
        
        # Load weights normally - the to() method will handle dtype preservation
        instance.adapter_linear.weight.data = state_dict['adapter_linear.weight']
        instance.adapter_linear.bias.data = state_dict['adapter_linear.bias']
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