import torch
from torch import nn
from transformer_lens import HookedTransformer
from transformers import AutoConfig
from typing import List, Tuple
from transformer_lens.ActivationCache import ActivationCache
from transformers.modeling_outputs import CausalLMOutput

from .sae_adapter import SAEAdapter

class HookedModel(nn.Module):
    """
    Wraps a base LLM and an SAEAdapter for training and analysis.
    
    This module can be run with or without steering enabled and provides
    a method to cache both LLM and SAE internal activations.
    
    #* Maybe we should add one more flag to only use the Base SAE without steering?
    """
    def __init__(self, model: HookedTransformer, sae_adapter: SAEAdapter):
        super().__init__()
        self.model = model
        self.config = AutoConfig.from_pretrained(model.cfg.model_name)
        self.sae_adapter = sae_adapter
        self.hook_name = self.sae_adapter.cfg.hook_name
        self.steering_active = True  # Steering is on by default
        self.warnings_issued = dict()

        for param in self.model.parameters():
            param.requires_grad = False
            
    def enable_steering(self):
        """Activates the SAEAdapter intervention for subsequent forward passes."""
        self.steering_active = True

    def disable_steering(self):
        """Deactivates the SAEAdapter, making the model behave like the base LLM."""
        self.steering_active = False

    def forward(self, tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Performs a forward pass. Runs with or without steering based on current state.
        This method is optimized for training loops.
        """
        if not self.steering_active:
            return self.model(tokens, **kwargs)

        def steering_hook(activation_value, hook):
            return self.sae_adapter(activation_value)

        final_logits = self.model.run_with_hooks(
            tokens,
            fwd_hooks=[(self.hook_name, steering_hook)],
            **kwargs
        )
        
        return CausalLMOutput(logits=final_logits)

    def run_with_cache(
        self, 
        tokens: torch.Tensor, 
        **kwargs
    ) -> Tuple[torch.Tensor, ActivationCache]:
        """
        Runs a forward pass and returns final logits and a combined activation cache.
        This method is designed for analysis.

        Args:
            tokens: Input tokens.
            **kwargs: Additional arguments for the model's forward pass.

        Returns:
            A tuple of (logits, combined_cache), where the cache contains
            activations from both the LLM and the SAEAdapter.
        """
        if not self.steering_active:
            return self.model.run_with_cache(tokens, **kwargs)

        sae_cache_storage = {}
        def hook_fn_with_cache(activation_value, hook):
            sae_output, sae_cache = self.sae_adapter.run_with_cache(activation_value)
            sae_cache_storage.update(sae_cache)
            return sae_output

        logits, llm_cache = self.model.run_with_cache(
            tokens,
            fwd_hooks=[(self.hook_name, hook_fn_with_cache)],
            **kwargs
        )
        
        # Merge the SAE cache into the LLM's cache
        llm_cache.update(sae_cache_storage)
        
        return logits, llm_cache

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Helper to get only the adapter's trainable parameters for the optimizer."""
        return self.sae_adapter.get_trainable_parameters()