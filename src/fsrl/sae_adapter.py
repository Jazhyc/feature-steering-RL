import torch
from torch import nn
import json
from typing import Any, Optional
from sae_lens import SAE, SAEConfig
from pathlib import Path
from safetensors.torch import save_file, load_file
from transformer_lens.hook_points import HookPoint
from contextlib import contextmanager

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
        
        self.adapter = nn.Sequential(
            nn.Linear(self.cfg.d_in, self.cfg.d_sae, bias=True),
            nn.ReLU()
        )
        
        self._initialize_adapter()
        
        self.adapter.to(self.device, self.dtype)

        # Instance variables for logging
        self._current_steering_l0_norm = torch.tensor(0.0)
        self._current_steering_l1_norm = torch.tensor(0.0)
        self._current_steering_l2_norm = torch.tensor(0.0)
        
        # Create hook points for the adapter to cache activations or for interventions
        self.hook_sae_adapter = HookPoint()
        self.hook_sae_fusion = HookPoint()
        
        # Add hooks to the hook manager
        self.setup()
        
    def _initialize_adapter(self):
        """
        A simple initialization to kickstart learning.
        A small positive bias ensures the ReLU is active for some inputs initially.
        """
        linear_layer = self.adapter[0]
        # Uniform initialization for weights
        nn.init.uniform_(linear_layer.weight, -1e-9, 1e-9)
        # A small positive bias to get out of the "dead ReLU" region.
        nn.init.constant_(linear_layer.bias, 0.0)

    def get_steering_vector(self, adapter_input: torch.Tensor) -> torch.Tensor:
        """Computes the steering vector from the trainable adapter."""
        
        # Ensures adapter receives same input as regular SAE
        adapter_input = adapter_input - (self.b_dec * self.cfg.apply_b_dec_to_input)

        steered_activations = self.adapter(adapter_input.to(self.dtype))
        
        # Statistics for loss and logging 
        # L1 norm: mean of L1 norms across batch (sum over features, mean over batch)
        self._current_steering_l1_norm = torch.mean(torch.sum(torch.abs(steered_activations), dim=-1))
        
        # L2 norm: mean of L2 norms across batch (sum over features, mean over batch)
        self._current_steering_l2_norm = torch.mean(torch.sum(steered_activations**2, dim=-1))

        # L0 norm: mean fraction of non-zero elements across batch
        non_zero_mask = torch.abs(steered_activations) > 1e-6
        self._current_steering_l0_norm = torch.mean(non_zero_mask.float().sum(dim=-1))
        
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
        return list(self.adapter.parameters())
    
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
        
        save_file(self.adapter.state_dict(), path / "adapter_weights.safetensors")

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
        instance.adapter.load_state_dict(state_dict)
        
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