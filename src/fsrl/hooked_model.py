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


class BaseHookedModel(nn.Module):
    """
    Base wrapper for HookedTransformer that adds the config attribute
    and ensures output compatibility with SimPOTrainer.
    """
    def __init__(self, model: HookedTransformer):
        super().__init__()
        self.model = model
        self.tokenizer = model.tokenizer
        self.tokenizer.chat_template = model.tokenizer.chat_template
        
        # Add config attribute that SimPOTrainer expects
        space = get_hf_space(model.cfg.model_name)
        self.config = AutoConfig.from_pretrained(f"{space}/{model.cfg.model_name}")
        
        # Get device from the model's parameters
        try:
            self.device = next(self.model.parameters()).device
        except StopIteration:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @classmethod
    def from_pretrained(
        cls, 
        model_path: str, 
        device: str = "cuda",
        dtype: str = "bfloat16",
        **kwargs
    ) -> "BaseHookedModel":
        """
        Load a trained BaseHookedModel from a saved path.
        
        Args:
            model_path: Path to the saved model directory
            device: Device to load the model on
            dtype: Data type for the model
            **kwargs: Additional arguments for HookedTransformer.from_pretrained
        
        Returns:
            BaseHookedModel instance with loaded weights
        """
        from pathlib import Path
        import json
        
        # Auto-detect device if not specified or if cuda is requested but not available
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)
        
        model_path = Path(model_path)
        
        # Read the config to determine the base model name
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # First try to get the original model name we saved
            base_model_name = config.get('_original_model_name', None)
            
            # If not available, try _name_or_path
            if base_model_name is None:
                base_model_name = config.get('_name_or_path', None)
            
            # If still not available or points to a local path, use model_type
            if base_model_name is None or ('/' in base_model_name and not base_model_name.startswith(('gpt2', 'meta-llama', 'google', 'microsoft'))):
                model_type = config.get('model_type', 'gpt2')
                if model_type == 'gpt2':
                    # Fallback to gpt2 - this is the dangerous case you pointed out
                    # We should log a warning here
                    print(f"Warning: Using fallback model 'gpt2' for model_type '{model_type}'. Original model variant unknown.")
                    base_model_name = 'gpt2'
                elif model_type == 'gemma2':
                    base_model_name = 'google/gemma-2-2b-it'
                elif model_type == 'gemma':
                    base_model_name = 'google/gemma-2-2b-it'
                else:
                    # For other model types, try to use the model_type directly
                    base_model_name = model_type
        else:
            # Fallback to gpt2 if no config found
            print("Warning: No config.json found. Using fallback model 'gpt2'.")
            base_model_name = 'gpt2'
        
        # First load the base model architecture
        base_model = HookedTransformer.from_pretrained_no_processing(
            base_model_name,
            device=device,
            torch_dtype=torch_dtype,
            attn_implementation="sdpa",
            **kwargs
        )
        
        # Then load the fine-tuned weights
        model_file = model_path / "pytorch_model.bin"
        if model_file.exists():
            state_dict = torch.load(model_file, map_location=device)
            # Remove any prefix that might be added during saving
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                # Remove 'model.' prefix if present
                clean_key = key.replace('model.', '') if key.startswith('model.') else key
                cleaned_state_dict[clean_key] = value
            
            base_model.load_state_dict(cleaned_state_dict, strict=True)
        
        return cls(base_model)
    
    def save_pretrained(self, save_path: str):
        """
        Save the model to the specified path.
        
        Args:
            save_path: Directory path where to save the model
        """
        from pathlib import Path
        import json
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save the underlying HookedTransformer model
        # This will save both the model weights and config
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(str(save_path))
            
            # After saving, update the config to include the original model name
            config_path = save_path / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Add the original model name for proper loading later
                config['_original_model_name'] = self.model.cfg.model_name
                config['_name_or_path'] = self.model.cfg.model_name
                
                # Write back the updated config
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
        else:
            # Fallback: save the model state dict and create config manually
            torch.save(self.model.state_dict(), save_path / "pytorch_model.bin")
            
            # Save the config with original model name
            if hasattr(self.config, 'to_dict'):
                config_dict = self.config.to_dict()
            else:
                config_dict = vars(self.config)
            
            config_dict['_original_model_name'] = self.model.cfg.model_name  
            config_dict['_name_or_path'] = self.model.cfg.model_name
            
            with open(save_path / "config.json", 'w') as f:
                json.dump(config_dict, f, indent=2)
    
    def forward(self, tokens: torch.Tensor, **kwargs) -> CausalLMOutput:
        """Forward pass that returns CausalLMOutput for SimPOTrainer compatibility."""
        logits = self.model(tokens, **kwargs)
        return CausalLMOutput(logits=logits)
    
    def generate(self, *args, **kwargs):
        """Generates text using the model's generate method."""
        # Handle parameter compatibility with lm-eval
        if 'input_ids' in kwargs:
            input_ids = kwargs.pop('input_ids')
            # If no positional args were provided, use input_ids as the input
            if len(args) == 0:
                args = (input_ids,)
            # If input was already provided as positional arg, keep it
        
        # Map Hugging Face parameter names to HookedTransformer parameter names
        if 'max_length' in kwargs:
            max_length = kwargs.pop('max_length')
            # HookedTransformer uses max_new_tokens instead of max_length
            # We need to calculate max_new_tokens from max_length and input length
            if len(args) > 0:
                input_length = args[0].shape[-1] if hasattr(args[0], 'shape') else len(args[0])
                kwargs['max_new_tokens'] = max(1, max_length - input_length)
            else:
                kwargs['max_new_tokens'] = max_length
        
        # Map other common parameter differences
        if 'num_return_sequences' in kwargs:
            # HookedTransformer doesn't support this, remove it
            kwargs.pop('num_return_sequences')
        
        if 'pad_token_id' in kwargs:
            # HookedTransformer doesn't use this parameter
            kwargs.pop('pad_token_id')
        
        # Capture and remove attention_mask (HookedTransformer doesn't accept it)
        attention_mask = kwargs.pop('attention_mask', None)
        
        if 'stopping_criteria' in kwargs:
            # HookedTransformer doesn't use stopping_criteria
            kwargs.pop('stopping_criteria')
        
        if 'use_cache' in kwargs:
            # HookedTransformer doesn't use use_cache (it has use_past_kv_cache instead)
            use_cache = kwargs.pop('use_cache')
            # Map to HookedTransformer equivalent if needed
            if use_cache and 'use_past_kv_cache' not in kwargs:
                kwargs['use_past_kv_cache'] = use_cache
        
        # If we received an attention mask for a batched tokens input, trim padding and
        # re-route through string inputs to avoid EOS-as-padding issues.
        if (
            attention_mask is not None
            and len(args) > 0
            and isinstance(args[0], torch.Tensor)
            and args[0].dim() == 2
        ):
            tokens_batch: torch.Tensor = args[0]
            # Ensure mask is on CPU for sum, but keep indexing on original device
            if attention_mask.dim() != 2 or attention_mask.shape != tokens_batch.shape:
                # Fallback: ignore malformed masks
                pass
            else:
                # Compute true lengths from mask and decode each example
                lengths = attention_mask.sum(dim=1).tolist()
                input_texts = []
                for row_idx, length in enumerate(lengths):
                    length_int = int(length)
                    # Guard against zero-length sequences
                    if length_int <= 0:
                        # Empty input: decode as empty string
                        input_texts.append("")
                        continue
                    seq_tokens = tokens_batch[row_idx, :length_int]
                    text = self.tokenizer.decode(seq_tokens, skip_special_tokens=False)
                    input_texts.append(text)
                # Replace positional tokens arg with list[str]
                args = (input_texts,)
                # Ensure we get tokens back for downstream code that expects tensors
                if 'return_type' not in kwargs:
                    kwargs['return_type'] = 'tokens'
                # Left-pad during internal tokenization to ensure last token is from content
                if 'padding_side' not in kwargs:
                    kwargs['padding_side'] = 'left'
        
        return self.model.generate(*args, **kwargs)
    
    def tie_weights(self):
        """Tie the weights of the input embedding and output layers."""
        # Check if the HookedTransformer has tie_weights method
        if hasattr(self.model, 'tie_weights') and callable(getattr(self.model, 'tie_weights')):
            return self.model.tie_weights()
        
        # Check if the underlying PyTorch model (model.model) has tie_weights
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'tie_weights') and callable(getattr(self.model.model, 'tie_weights')):
            return self.model.model.tie_weights()
        
        # If no tie_weights method is found, this is likely fine for many models
        pass

    def get_norms(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Helper to get norm values for compatibility with SAE training.
        Returns zeros since full model training doesn't use SAE adapters.
        """
        device = self.device
        zero_tensor = torch.tensor(0.0, device=device)
        return zero_tensor, zero_tensor, zero_tensor


class HookedModel(BaseHookedModel):
    """
    Wraps a base LLM and an SAEAdapter for training and analysis.

    This version is optimized for performance and `torch.compile` compatibility.
    Instead of using `run_with_hooks`, it structurally modifies the base model
    by replacing a HookPoint with the SAEAdapter module.
    """
    def __init__(self, model: HookedTransformer, sae_adapter: SAEAdapter):
        super().__init__(model)
        self.sae_adapter = sae_adapter
        self.hook_name = self.sae_adapter.cfg.hook_name

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

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Helper to get only the adapter's trainable parameters for the optimizer."""
        # This remains unchanged as it correctly points to the adapter's params.
        return self.sae_adapter.get_trainable_parameters()

    def get_norms(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Helper to get the adapter's norm values."""
        return self.sae_adapter.get_norms()

    # --- Steering control passthrough ---
    def set_steering_fraction(self, fraction: float) -> None:
        """Set the fraction of features used by the adapter's steering vector."""
        self.sae_adapter.set_steering_fraction(fraction)
        
    def set_masked_features(self, feature_indices: List[int]) -> None:
        """Set which features should be masked (turned off) during inference."""
        self.sae_adapter.set_masked_features(feature_indices)
        
    def clear_masked_features(self) -> None:
        """Clear all masked features."""
        self.sae_adapter.clear_masked_features()
        
    def get_masked_features(self) -> set[int]:
        """Get the current set of masked feature indices."""
        return self.sae_adapter.get_masked_features()