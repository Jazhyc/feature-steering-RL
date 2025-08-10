import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

NUM_STEPS_TO_CHECK = 10

class DebugInspector(TrainerCallback):
    """
    This callback does two things:
    1. On train begin, it inspects the optimizer groups to see which params are where.
    2. On each step, it stores a snapshot of the parameters BEFORE the step,
       and compares it to the parameters AFTER the step to prove if they changed.
    """
    def __init__(self):
        super().__init__()
        self.param_snapshots = {}

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        optimizer = kwargs.get("optimizer")
        model = kwargs.get("model")
        if optimizer is None or model is None:
            return

        print("\n--- OPTIMIZER GROUP INSPECTION ---")
        
        # Create a map from parameter ID to name for easy lookup
        param_map = {id(p): name for name, p in model.named_parameters() if p.requires_grad}

        for i, param_group in enumerate(optimizer.param_groups):
            print(f"\nParameter Group {i}:")
            print(f"  LR: {param_group['lr']}")
            print(f"  Weight Decay: {param_group['weight_decay']}")
            print(f"  Num Params in group: {len(param_group['params'])}")
            print("  Parameters in this group:")
            for param in param_group['params']:
                param_id = id(param)
                param_name = param_map.get(param_id, "Unknown Parameter")
                print(f"    - {param_name} (Shape: {param.shape})")
        print("--------------------------------\n")

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Only check on the first few steps to avoid spamming logs
        if state.global_step <= NUM_STEPS_TO_CHECK:
            model = kwargs.get("model")
            if model is None:
                return
            
            # Store a copy of the parameters before the optimizer step
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.param_snapshots[name] = param.data.clone()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Only check on the first few steps
        if state.global_step <= NUM_STEPS_TO_CHECK:
            model = kwargs.get("model")
            if not self.param_snapshots or model is None:
                return

            print(f"\n--- PARAMETER UPDATE CHECK (End of Step {state.global_step}) ---")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    prev_param = self.param_snapshots.get(name)
                    if prev_param is not None:
                        # Check if the parameter tensor is exactly the same
                        are_equal = torch.equal(prev_param, param.data)
                        if are_equal:
                            print(f"  ❌ {name}: DID NOT CHANGE.")
                        else:
                            # Calculate the change magnitude to show it moved
                            change = torch.abs(param.data - prev_param).mean().item()
                            print(f"  ✅ {name}: CHANGED (avg magnitude of change: {change:.4e})")
            print("---------------------------------------\n")
            # Clear snapshots for the next check
            self.param_snapshots = {}
