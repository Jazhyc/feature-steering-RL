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
        
        # Steering magnitude (optional, default: not used)
        self.use_steering_magnitude = kwargs.get("use_steering_magnitude", False)
        if self.use_steering_magnitude:
            # Initialized to 0 so steering doesn't affect model initially
            self.steering_magnitude = nn.Parameter(torch.zeros(self.cfg.d_sae, dtype=torch.float32))
        else:
            self.steering_magnitude = None
        
        # Activation function configuration
        self.activation_type = kwargs.get("activation_type", "soft_threshold")  # Default to soft threshold
        
        # Set up activation-specific parameters
        if self.activation_type == "jump_relu":
            initial_threshold = kwargs.get("initial_threshold", 0.001)
            self.bandwidth = kwargs.get("bandwidth", 0.001)
            self.log_threshold = nn.Parameter(
                torch.full((self.cfg.d_sae,), torch.log(torch.tensor(initial_threshold)), dtype=torch.float32)
            )
        elif self.activation_type == "soft_threshold":
            initial_threshold = kwargs.get("initial_threshold", 1e-6)
            self.soft_threshold = nn.Parameter(
                torch.full((self.cfg.d_sae,), initial_threshold)
            )
        # For "relu", no additional parameters needed

        self._initialize_adapter()
        self.to(self.device, self.dtype)
        
        # Ensure log_threshold stays in fp32 for numerical stability (only JumpReLU needs this)
        if self.activation_type == "jump_relu":
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
        
        # Feature masking for ablation studies (set of feature indices to turn off)
        self.masked_features: set[int] = set()
        self._masked_indices_tensor: torch.Tensor = None  # Precomputed tensor for efficiency
        
    @property
    def adapter_threshold(self) -> torch.Tensor:
        """Computes the threshold from the learnable log_threshold (only for JumpReLU)."""
        if self.activation_type == "jump_relu":
            return torch.exp(self.log_threshold)
        else:
            return None
    
    def to(self, *args, **kwargs):
        """Override to method to keep log_threshold in fp32 for numerical stability (JumpReLU only)."""
        # Store the original log_threshold to avoid dtype conversion (only JumpReLU needs fp32)
        original_log_threshold = None
        
        if hasattr(self, 'log_threshold') and self.log_threshold is not None:
            original_log_threshold = self.log_threshold.data.clone()
        
        # Call parent's to method
        result = super().to(*args, **kwargs)
        
        # Restore log_threshold to fp32 without any intermediate conversions (JumpReLU only)
        if original_log_threshold is not None:
            device = self.adapter_linear.weight.device
            self.log_threshold.data = original_log_threshold.to(device)

        # Update masked indices tensor to match new device
        if hasattr(self, '_masked_indices_tensor') and self._masked_indices_tensor is not None:
            self._update_masked_indices_tensor()

        return result
        
    def _initialize_adapter(self):
        """
        Initialization for adapter weights.
        
        If using steering magnitude:
            - Kaiming (He) initialization for weights (suitable for ReLU-like activations)
            - For JumpReLU: biases initialized to threshold
            - For soft threshold/ReLU: biases initialized to zero
        
        If not using steering magnitude:
            - Uniform initialization between -initial_threshold and +initial_threshold
            - Biases initialized to zero
        """
        if self.use_steering_magnitude:
            # Standard Kaiming initialization
            nn.init.kaiming_uniform_(self.adapter_linear.weight, nonlinearity='relu')

            if self.activation_type == "jump_relu":
                # Use the threshold value but ensure it matches the bias dtype
                threshold_mean = self.adapter_threshold.mean().item()
                nn.init.constant_(self.adapter_linear.bias, threshold_mean)
            else:
                # Initialize bias to zero for soft threshold and regular ReLU
                nn.init.constant_(self.adapter_linear.bias, 0.0)
        else:
            # Uniform initialization with +/- initial_threshold
            if self.activation_type == "jump_relu":
                initial_threshold = self.adapter_threshold.mean().item()
            elif self.activation_type == "soft_threshold":
                initial_threshold = self.soft_threshold.mean().item()
            else:
                # For ReLU without steering magnitude, use a small default threshold
                initial_threshold = 0.01
            
            nn.init.uniform_(self.adapter_linear.weight, -initial_threshold, initial_threshold)
            nn.init.constant_(self.adapter_linear.bias, 0.0)

    def _apply_topk_feature_selection(
        self, 
        steered_activations: torch.Tensor, 
        steering_fraction: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply top-k feature selection to steering vector based on absolute values.
        
        Args:
            steered_activations: The steering activations tensor [batch_size, d_sae]
            steering_fraction: Fraction of features to keep (0.0, 1.0]
            
        Returns:
            Tuple of (masked_activations, sparsity_mask)
        """
        d = steered_activations.shape[-1]
        k = max(1, int(d * float(steering_fraction)))
        
        # Compute top-k indices by absolute value
        with torch.no_grad():
            topk = torch.topk(steered_activations.abs(), k=k, dim=-1, largest=True, sorted=False)
            mask = torch.zeros_like(steered_activations, dtype=steered_activations.dtype)
            mask.scatter_(-1, topk.indices, 1.0)
        
        masked_activations = steered_activations * mask
        return masked_activations, mask
    
    def _apply_feature_masking(self, steered_activations: torch.Tensor, sparsity_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply feature masking by setting specified features to zero.
        
        Returns:
            Tuple of (masked_activations, updated_sparsity_mask)
        """
        if self._masked_indices_tensor is None or len(self._masked_indices_tensor) == 0:
            return steered_activations, sparsity_mask
            
        # Create a mask tensor with ones, then set masked features to zero
        mask = torch.ones_like(steered_activations)
        
        # Efficiently set all masked features to zero using precomputed indices
        mask[..., self._masked_indices_tensor] = 0.0
        
        # Update sparsity mask to reflect masked features
        updated_sparsity_mask = sparsity_mask * mask
        
        return steered_activations * mask, updated_sparsity_mask

    def _apply_jump_relu_activation(self, pre_activations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply JumpReLU activation with learnable threshold and bandwidth."""
        # Apply ReLU first, then threshold and bandwidth
        relu_activations = F.relu(pre_activations)
        threshold_tensor = self.adapter_threshold.to(relu_activations.dtype)
        bandwidth_tensor = torch.tensor(self.bandwidth, dtype=relu_activations.dtype, device=relu_activations.device)

        steered_activations = JumpReLU.apply(relu_activations, threshold_tensor, bandwidth_tensor)
        sparsity_mask = Step.apply(relu_activations, threshold_tensor, bandwidth_tensor)
        
        return steered_activations, sparsity_mask

    def _apply_soft_threshold_activation(self, pre_activations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply soft threshold activation: sign(x) * ReLU(|x| - threshold)."""
        threshold_tensor = self.soft_threshold.to(pre_activations.dtype)
        steered_activations = torch.sign(pre_activations) * F.relu(torch.abs(pre_activations) - threshold_tensor)
        
        # Sparsity mask: active when |pre_activation| > threshold
        sparsity_mask = (torch.abs(pre_activations) > threshold_tensor).float()
        
        return steered_activations, sparsity_mask

    def _apply_relu_activation(self, pre_activations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply standard ReLU activation."""
        steered_activations = F.relu(pre_activations)
        
        # For regular ReLU, L0 norm is calculated from non-zero steered activations
        sparsity_mask = (steered_activations > 0).float()
        
        return steered_activations, sparsity_mask

    def _compute_statistics(self, steered_activations: torch.Tensor, sparsity_mask: torch.Tensor) -> None:
        """Compute and store L0, L1, and L2 norms for logging."""
        # L0 is count of non-zero active features after any masking
        self._norm_l0 = torch.sum(sparsity_mask, dim=-1).mean()

        # Statistics for loss and logging
        # L1 and L2 norms: mean of norms across batch (sum over features, mean over batch)
        self._norm_l1 = torch.mean(torch.sum(torch.abs(steered_activations), dim=-1))
        self._norm_l2 = torch.mean(torch.sum(steered_activations**2, dim=-1))

    def get_steering_vector(self, adapter_input: torch.Tensor) -> torch.Tensor:
        """Computes the steering vector from the trainable adapter."""
        
        # Apply the same preprocessing as the main SAE to ensure consistency
        processed_input = self.process_sae_in(adapter_input)
        
        # Get pre-activations from linear layer
        pre_activations = self.adapter_linear(processed_input.to(self.dtype))
        
        # Apply the appropriate activation function
        if self.activation_type == "jump_relu":
            steered_activations, sparsity_mask = self._apply_jump_relu_activation(pre_activations)
        elif self.activation_type == "soft_threshold":
            steered_activations, sparsity_mask = self._apply_soft_threshold_activation(pre_activations)
        else:  # activation_type == "relu"
            steered_activations, sparsity_mask = self._apply_relu_activation(pre_activations)
            
        # Optionally restrict to top-k features by absolute value (per item)
        if 0.0 < getattr(self, "steering_fraction", 1.0) < 1.0:
            steered_activations, sparsity_mask = self._apply_topk_feature_selection(
                steered_activations, self.steering_fraction
            )

        # Apply feature masking for ablation studies (turn off specific features)
        # This updates both activations and sparsity mask to reflect masked features
        steered_activations, sparsity_mask = self._apply_feature_masking(steered_activations, sparsity_mask)
        
        # Optionally apply steering magnitude
        if self.use_steering_magnitude:
            # Clip steering magnitude to [0, 1] and apply
            clipped_magnitude = torch.clamp(self.steering_magnitude, min=0.0, max=1.0)
            steered_activations = steered_activations * clipped_magnitude.to(steered_activations.dtype)

        # Apply hook and compute statistics
        steered_activations = self.hook_sae_adapter(steered_activations)
        self._compute_statistics(steered_activations, sparsity_mask)
        
        return steered_activations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        fused_feature_acts = F.relu(fused_feature_acts) # Prevents negative feature activations which should not be possible
        fused_feature_acts = self.hook_sae_fusion(fused_feature_acts)

        # Decode the modulated features back into the residual stream
        sae_out = self.decode(fused_feature_acts)
        
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

        instance.to(dtype=torch.bfloat16)

        # Manually add the release info to the config for later saving.
        instance.cfg.release = release
        instance.cfg.sae_id = sae_id
        # Construct neuronpedia_id from release and sae_id for API access
        instance.cfg.neuronpedia_id = f"{release}/{sae_id}"
        
        # Load the state dict of the base SAE; `strict=False` ignores missing adapter keys
        instance.load_state_dict(sae_temp.state_dict(), strict=False)
        
        return instance, cfg_dict, sparsity

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Returns the adapter's trainable parameters for an optimizer."""
        params = [self.adapter_linear.weight, self.adapter_linear.bias]
        
        if self.use_steering_magnitude:
            params.append(self.steering_magnitude)
            
        if self.activation_type == "jump_relu":
            params.append(self.log_threshold)
        elif self.activation_type == "soft_threshold":
            params.append(self.soft_threshold)
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
        
    def set_masked_features(self, feature_indices: list[int]) -> None:
        """Set which features should be masked (turned off) during inference."""
        self.masked_features = set(feature_indices)
        # Precompute the tensor for efficient masking
        self._update_masked_indices_tensor()
        
    def clear_masked_features(self) -> None:
        """Clear all masked features."""
        self.masked_features = set()
        self._masked_indices_tensor = None
        
    def _update_masked_indices_tensor(self) -> None:
        """Update the precomputed tensor of masked indices for efficient masking."""
        if not self.masked_features:
            self._masked_indices_tensor = None
            return
            
        # Filter indices to only include valid ones (within SAE dimension)
        valid_indices = [idx for idx in self.masked_features if 0 <= idx < self.cfg.d_sae]
        
        if valid_indices:
            self._masked_indices_tensor = torch.tensor(
                valid_indices,
                device=self.device,
                dtype=torch.long
            )
        else:
            self._masked_indices_tensor = None
        
    def get_masked_features(self) -> set[int]:
        """Get the current set of masked feature indices."""
        return self.masked_features.copy()
    
    def save_adapter(self, path: str | Path):
        """Saves only the trainable adapter weights and its configuration."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Get adapter state dict - the to() method already handles dtype preservation
        adapter_state_dict = {
            'adapter_linear.weight': self.adapter_linear.weight,
            'adapter_linear.bias': self.adapter_linear.bias,
        }
        
        if self.use_steering_magnitude:
            adapter_state_dict['steering_magnitude'] = self.steering_magnitude
        
        # Save threshold parameters based on activation type
        if self.activation_type == "jump_relu":
            adapter_state_dict['log_threshold'] = self.log_threshold
        elif self.activation_type == "soft_threshold":
            adapter_state_dict['soft_threshold'] = self.soft_threshold
        
        save_file(adapter_state_dict, path / "adapter_weights.safetensors")

        adapter_config = {
            "base_sae_release": self.cfg.release,
            "base_sae_id": self.cfg.sae_id,
            "activation_type": self.activation_type,
            "use_steering_magnitude": self.use_steering_magnitude,
            "model_name": self.cfg.model_name,
            "neuronpedia_id": f"{self.cfg.release}/{self.cfg.sae_id}",
        }
        
        # Save activation-specific config
        if self.activation_type == "jump_relu":
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
        activation_type = config.get("activation_type", "relu")  # Default to relu for legacy models
        
        # Legacy support for old boolean flags
        if config.get("use_jump_relu", False):
            activation_type = "jump_relu"
        elif config.get("use_soft_threshold", False):
            activation_type = "soft_threshold"
        
        adapter_kwargs = {
            "activation_type": activation_type,
            "use_steering_magnitude": config.get("use_steering_magnitude", False)
        }
        
        if activation_type == "jump_relu":
            adapter_kwargs["bandwidth"] = config.get("bandwidth", 0.001)
            adapter_kwargs["initial_threshold"] = config.get("initial_threshold", 0.001)
        elif activation_type == "soft_threshold":
            adapter_kwargs["initial_threshold"] = config.get("initial_threshold", 0.01)

        # Create the full model instance by calling our own from_pretrained
        instance, _, _ = cls.from_pretrained(
            release=config["base_sae_release"],
            sae_id=config["base_sae_id"],
            device=device,
            force_download=force_download,
            **adapter_kwargs
        )
        
        # Override with saved model_name and neuronpedia_id if available in config
        # (for backward compatibility and future configs)
        if "model_name" in config:
            instance.cfg.model_name = config["model_name"]
        if "neuronpedia_id" in config:
            instance.cfg.neuronpedia_id = config["neuronpedia_id"]

        # Load the locally saved adapter weights
        state_dict = load_file(path / "adapter_weights.safetensors", device=device)
        
        # Load weights normally - the to() method will handle dtype preservation
        instance.adapter_linear.weight.data = state_dict['adapter_linear.weight']
        instance.adapter_linear.bias.data = state_dict['adapter_linear.bias']
        
        # Load steering magnitude if it's being used
        if instance.use_steering_magnitude:
            if 'steering_magnitude' in state_dict:
                instance.steering_magnitude.data = state_dict['steering_magnitude']
            else:
                # Backwards compatibility: default to 1.0 if not present
                instance.steering_magnitude.data = torch.ones(instance.cfg.d_sae, dtype=torch.float32, device=device)
        
        # Load threshold parameters based on what's in the state dict
        if 'log_threshold' in state_dict:
            instance.log_threshold.data = state_dict['log_threshold']
        elif 'soft_threshold' in state_dict:
            instance.soft_threshold.data = state_dict['soft_threshold']
        
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