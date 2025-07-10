import torch
from torch import nn
from transformer_lens import HookedTransformer
from typing import List

from .sae_adapter import SAEAdapter

class HookedModel(nn.Module):
    """
    Wraps a base LLM and an SAEAdapter for training via hooks.
    
    This module steers the LLM's activations through the SAEAdapter at a
    specified layer, making it compatible with training libraries like TRL.
    For examining features, it might be better to use the SAEAdapter directly.
    """
    def __init__(self, model: HookedTransformer, sae_adapter: SAEAdapter):
        super().__init__()
        self.model = model
        self.sae_adapter = sae_adapter
        self.hook_name = self.sae_adapter.cfg.hook_name

        # Freeze the base LLM; adapter's trainable part remains active.
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, tokens: torch.Tensor, **kwargs):
        """Performs a forward pass through the LLM, steered by the SAEAdapter."""

        def steering_hook(activation_value, hook):
            # This hook function replaces the original activation with the
            # steered output from the SAEAdapter.
            return self.sae_adapter(activation_value)

        # run_with_hooks executes the model, applying the hook at the target layer.
        return self.model.run_with_hooks(
            tokens,
            fwd_hooks=[(self.hook_name, steering_hook)],
            **kwargs
        )

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Helper to get only the adapter's trainable parameters for the optimizer."""
        return self.sae_adapter.get_trainable_parameters()