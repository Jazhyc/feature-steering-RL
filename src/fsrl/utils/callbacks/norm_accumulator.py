from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

class NormAccumulatorResetCallback(TrainerCallback):
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        This event is triggered at the end of a training step, after the optimizer has performed its update.
        Reset the norms that should be synchronized with gradient updates for compatibility with gradient accumulation.
        """
        model = kwargs.get("model")
        if model is not None and hasattr(model, 'sae_adapter'):
            model.sae_adapter.reset_norm_accumulators()