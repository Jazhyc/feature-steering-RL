import torch
from torch import nn
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformers import AutoConfig
from typing import List, Tuple, Any
from transformer_lens.ActivationCache import ActivationCache
from transformers.modeling_outputs import CausalLMOutput
from transformers import AutoConfig

from .sae_adapter import SAEAdapter

HF_SPACE_MAPPING = {
    "gpt": "openai-community",
    "gemma": "google"
}

def get_hf_space(model_name: str) -> str:
    """
    Returns the Hugging Face space based on the model name.
    """
    for key in HF_SPACE_MAPPING:
        if key in model_name.lower():
            return HF_SPACE_MAPPING[key]
    raise ValueError(f"Model name '{model_name}' does not match any known Hugging Face space.")

class HookedModel(nn.Module):
    """
    Wraps a base LLM and an SAEAdapter for training and analysis.

    This version is optimized for performance and `torch.compile` compatibility.
    Instead of using `run_with_hooks`, it structurally modifies the base model
    by replacing a HookPoint with the SAEAdapter module.
    """
    def __init__(self, model: HookedTransformer, sae_adapter: SAEAdapter):
        super().__init__()
        self.model = model
        self.sae_adapter = sae_adapter
        
        space = get_hf_space(model.cfg.model_name)
        self.config = AutoConfig.from_pretrained(f"{space}/{model.cfg.model_name}")
        self.hook_name = self.sae_adapter.cfg.hook_name

        # Get device from the model's parameters
        try:
            self.device = next(self.model.parameters()).device
        except StopIteration:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # This allows us to disable steering and restore the original model state.
        self._original_hook_point = self._get_deep_attr(self.model, self.hook_name)
        
        # Check if it's a valid HookPoint to ensure we're not overwriting something else.
        if not isinstance(self._original_hook_point, HookPoint):
            raise ValueError(
                f"The attribute at '{self.hook_name}' is not a HookPoint. "
                "Cannot attach SAE."
            )
            
        # Freezes all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # By default, steering is enabled on initialization.
        self.enable_steering()

    def _get_deep_attr(self, obj: Any, path: str) -> Any:
        """Helper to access nested attributes (e.g., 'blocks.0.mlp.hook_post')."""
        parts = path.split(".")
        for part in parts:
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
        return obj

    def _set_deep_attr(self, obj: Any, path: str, value: Any):
        """Helper to set nested attributes."""
        parts = path.split(".")
        for part in parts[:-1]:
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
        setattr(obj, parts[-1], value)

    def enable_steering(self):
        """
        Activates the SAEAdapter intervention by structurally replacing the
        target HookPoint with the SAEAdapter module.
        """
        self._set_deep_attr(self.model, self.hook_name, self.sae_adapter)
        # We must call setup() for transformer-lens to recognize the change.
        self.model.setup()

    def disable_steering(self):
        """
        Deactivates the SAEAdapter by restoring the original HookPoint,
        making the model behave like the base LLM.
        """
        self._set_deep_attr(self.model, self.hook_name, self._original_hook_point)
        # We must call setup() for transformer-lens to recognize the change.
        self.model.setup()

    def forward(self, tokens: torch.Tensor, **kwargs) -> CausalLMOutput:
        """
        Performs a forward pass. The behavior (steered or not) depends on
        whether enable_steering() or disable_steering() was last called.

        This method is now static and compatible with `torch.compile`.
        """
        # The logic is now handled by the model's structure, not dynamic hooks.
        # We simply call the base model's forward pass.
        logits = self.model(tokens, **kwargs)
        
        # We maintain the convenient CausalLMOutput wrapper.
        return CausalLMOutput(logits=logits)
    
    def generate(self, *args, **kwargs):
        """
        Generates text using the model's generate method.
        
        Args:
            *args: Positional arguments for generation
            **kwargs: Keyword arguments for generation
        """
        return self.model.generate(*args, **kwargs)

    def run_with_cache(
        self, 
        tokens: torch.Tensor, 
        **kwargs
    ) -> Tuple[torch.Tensor, ActivationCache]:
        """
        Runs a forward pass and returns final logits and a combined activation cache.
        The returned cache automatically includes SAE activations if steering is enabled.
        This method is designed for analysis.
        """
        # With the new structural approach, transformer-lens's run_with_cache
        # will automatically find the HookPoints *inside* our SAEAdapter
        # (e.g., hook_sae_adapter, hook_sae_fusion) and cache them.
        # No manual cache merging is needed.
        logits, cache = self.model.run_with_cache(tokens, 
                                                  **kwargs)
        return logits, cache

    def tie_weights(self):
        """
        Tie the weights of the input embedding and output layers.
        Delegates to the underlying model's tie_weights method if available.
        This is needed for compatibility with evaluation libraries like lm_eval.
        """
        # Check if the HookedTransformer has tie_weights method
        if hasattr(self.model, 'tie_weights') and callable(getattr(self.model, 'tie_weights')):
            return self.model.tie_weights()
        
        # Check if the underlying PyTorch model (model.model) has tie_weights
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'tie_weights') and callable(getattr(self.model.model, 'tie_weights')):
            return self.model.model.tie_weights()
        
        # If no tie_weights method is found, this is likely fine for many models
        # Some models don't need weight tying, so we'll just pass silently
        pass

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Helper to get only the adapter's trainable parameters for the optimizer."""
        # This remains unchanged as it correctly points to the adapter's params.
        return self.sae_adapter.get_trainable_parameters()
    
    def get_steering_l1_norm(self) -> torch.Tensor:
        """
        Returns the L1 norm of the steering vector from the most recent forward pass.
        This value can be used both for logging and as the L1 penalty in the loss function.
        """
        return self.sae_adapter.get_steering_l1_norm()
    
    def get_steering_l2_norm(self) -> torch.Tensor:
        """
        Returns the L2 norm (mean of squared values) of the steering vector from the most recent forward pass.
        This value can be used for regularization in the loss function.
        """
        return self.sae_adapter.get_steering_l2_norm()
    
    def get_steering_l0_norm(self) -> torch.Tensor:
        """Returns the L0 norm (sparsity) of the steering vector from the most recent forward pass."""
        return self.sae_adapter.get_steering_l0_norm()