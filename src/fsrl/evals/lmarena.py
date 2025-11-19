import os
import lm_eval
import torch
import wandb
import json
from dotenv import load_dotenv
from pathlib import Path
from .. import SAEAdapter, HookedModel, BaseHookedModel
from ..utils.wandb_utils import (
    WandBModelDownloader,
    download_model_family,
    list_model_family,
)
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval.models.huggingface import HFLM

def load_classification_labels(classification_file: str) -> dict[str, list[int]]:
    """
    Load feature classification labels from DeepSeek JSON file.
    
    Returns:
        Dict with 'related' and 'not-related' keys containing lists of feature indices
    """
    print(f"Loading classification labels from: {classification_file}")
    
    with open(classification_file, 'r') as f:
        classifications = json.load(f)
    
    related_features = []
    not_related_features = []
    
    for item in classifications:
        # Extract feature index from feature_id like "gemma-2-2b-12-gemmascope-res-65k-53844"
        feature_id = item['feature_id']
        feature_index = int(feature_id.split('-')[-1])
        
        if item['label'] == 'related':
            related_features.append(feature_index)
        else:
            not_related_features.append(feature_index)
    
    print(f"Found {len(related_features)} related features, {len(not_related_features)} not-related features")
    
    return {
        'related': sorted(related_features),
        'not-related': sorted(not_related_features)
    }

def verify_model_state(hooked_model, exp_name: str, with_adapter: bool, full_ft: bool):
    """Verify and log model state for debugging consistency issues."""
    if not with_adapter or full_ft:
        return
    
    if hasattr(hooked_model, 'sae_adapter'):
        adapter = hooked_model.sae_adapter
        print(f"Model State Verification for {exp_name}:")
        print(f"  - Has adapter: True")
        print(f"  - Masked features: {len(adapter.get_masked_features()) if hasattr(adapter, 'get_masked_features') else 'N/A'}")
        print(f"  - Steering fraction: {getattr(adapter, 'steering_fraction', 1.0)}")
        print(f"  - Device: {adapter.device if hasattr(adapter, 'device') else 'N/A'}")
        print(f"  - Dtype: {adapter.dtype if hasattr(adapter, 'dtype') else 'N/A'}")
        
        # Check if steering is actually enabled by comparing current hook with adapter
        hook_name = adapter.cfg.hook_name if hasattr(adapter, 'cfg') else 'unknown'
        current_hook = hooked_model._get_deep_attr(hooked_model.model, hook_name) if hasattr(hooked_model, '_get_deep_attr') else None
        is_steering_active = current_hook is adapter if current_hook else False
        original_hook = getattr(hooked_model, '_original_hook_point', None)
        is_baseline_mode = current_hook is original_hook if original_hook else False
        
        print(f"  - Steering active: {is_steering_active}")
        print(f"  - Baseline mode (no steering): {is_baseline_mode}")
        print(f"  - Hook name: {hook_name}")
        
        # Additional verification for baseline
        if exp_name == "baseline":
            if not is_steering_active:
                print(f"  - WARNING: Baseline should have steering enabled, but it's disabled!")
            elif is_steering_active:
                masked_count = len(adapter.get_masked_features()) if hasattr(adapter, 'get_masked_features') else 0
                if masked_count == 0:
                    print(f"  - CORRECT: Baseline has steering enabled with no features masked")
                else:
                    print(f"  - WARNING: Baseline should have no features masked, but {masked_count} features are masked!")
        elif not is_steering_active:
            print(f"  - WARNING: Non-baseline experiment should have steering enabled, but it's disabled!")
    else:
        print(f"Model State Verification for {exp_name}: No adapter found")

def run_eval(runs, tasks, wandb_project="Gemma2-2B-muon", limit=0.01, with_adapter=True, full_ft=False, 
             alignment_classification_file=None, style_classification_file=None,
             ablation_experiments=None):
    
    root = Path(__file__).resolve().parent.parent.parent  # climb up to project root
    models_path = f"{root}/models/{wandb_project}" if not full_ft else f"{root}/models/full-gemma2_2B"
    
    load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(entity="feature-steering-RL", project="lm-eval")

    downloader = WandBModelDownloader(
        entity="feature-steering-RL",
        project=wandb_project,
        verbose=True
    )

    # Load classification labels if provided for ablation
    alignment_features = []
    style_features = []
    if alignment_classification_file:
        alignment_labels = load_classification_labels(alignment_classification_file)
        alignment_features = alignment_labels['related']
    if style_classification_file:
        style_labels = load_classification_labels(style_classification_file)
        style_features = style_labels['related']
    
    # Define ablation experiments
    if ablation_experiments is None and (alignment_features or style_features):
        # Default ablation experiments if features are provided
        both_features = sorted(list(set(alignment_features) | set(style_features)))
        ablation_experiments = [
            ("baseline", []),  # Always run baseline first
            ("no_alignment", alignment_features),
            ("no_style", style_features),
            ("no_both", both_features),
        ]
        print(f"Feature overlap analysis:")
        print(f"  Alignment features: {len(alignment_features)}")
        print(f"  Style features: {len(style_features)}")
        print(f"  Union (both): {len(both_features)}")
        print(f"  Overlap: {len(alignment_features) + len(style_features) - len(both_features)}")
    elif ablation_experiments is None:
        # No ablation, just run baseline
        ablation_experiments = [("baseline", [])]
    else:
        # Ensure baseline is first if custom ablation experiments are provided
        baseline_exists = any(exp[0] == "baseline" for exp in ablation_experiments)
        if not baseline_exists:
            ablation_experiments = [("baseline", [])] + ablation_experiments
        else:
            # Move baseline to front if it exists
            baseline_exp = next((exp for exp in ablation_experiments if exp[0] == "baseline"), None)
            if baseline_exp:
                ablation_experiments = [baseline_exp] + [exp for exp in ablation_experiments if exp[0] != "baseline"]

    summary_results = {}

    for run in runs:
        print(f"##### Running evaluation for run: {run} #####")
        print("=" * 30)
        project_name = f"feature-steering-rl/{wandb_project}" if full_ft else f"feature-steering-rl/{wandb_project}"
        run_objs = downloader.api.runs(project_name, filters={"display_name": run})
        downloader.download_model(run_objs[0], wandb_project)
        
        # fresh base model for each adapter to avoid hook point conflicts
        print(f"Loading fresh base model for run: {run}")
        if full_ft:
            base_model_path = downloader.models_base_dir / wandb_project / run / "full_model"
            base_model = BaseHookedModel.from_pretrained(base_model_path, device="cuda", dtype=torch.bfloat16)
            hooked_model = base_model
        else:
            base_model = HookedTransformer.from_pretrained_no_processing("google/gemma-2-9b-it", device="cuda", dtype=torch.bfloat16)

            adapter_path = downloader.models_base_dir / wandb_project / run / "adapter"

            print(f"Loading adapter from: {adapter_path}")
            sae_adapter = SAEAdapter.load_from_pretrained_adapter(adapter_path, device="cuda")
            hooked_model = HookedModel(base_model, sae_adapter)
        
            if not with_adapter:
                hooked_model.disable_steering()
                print(f"Evaluating model without steering")
        
        tokenizer = base_model.tokenizer
        
        # Run ablation experiments for this run
        run_results = {}
        for exp_name, masked_features in ablation_experiments:
            print(f"\n=== Running experiment: {exp_name} for run: {run} ===")
            print(f"Masking {len(masked_features)} features")
            
            # Handle baseline vs ablation experiments differently
            if hasattr(hooked_model, 'sae_adapter') and with_adapter and not full_ft:
                if exp_name == "baseline":
                    # For baseline: enable steering with no features masked (normal steering)
                    hooked_model.enable_steering()
                    hooked_model.clear_masked_features()  # Ensure no features are masked
                    print(f"Baseline experiment: steering enabled with no features masked (normal steering)")
                else:
                    # For ablation experiments: ensure steering is enabled, then apply masking
                    hooked_model.enable_steering()
                    hooked_model.clear_masked_features()  # Clear any previous masking
                    if masked_features:  # Only set masking if we have features to mask
                        hooked_model.set_masked_features(masked_features)
                    print(f"Ablation experiment: steering enabled with {len(masked_features)} features masked")
                
                print(f"Active masked features: {len(hooked_model.get_masked_features()) if hasattr(hooked_model, 'get_masked_features') else 'N/A'}")
                
                # Additional debugging for baseline case
                if exp_name == "baseline":
                    print(f"BASELINE DEBUG:")
                    adapter = hooked_model.sae_adapter
                    hook_name = adapter.cfg.hook_name if hasattr(adapter, 'cfg') else getattr(hooked_model, 'hook_name', 'unknown')
                    current_hook = hooked_model._get_deep_attr(hooked_model.model, hook_name) if hasattr(hooked_model, '_get_deep_attr') else None
                    steering_active = current_hook is adapter if current_hook is not None else False
                    print(f"  - Steering active: {steering_active}")
                    print(f"  - Adapter device: {adapter.device if hasattr(adapter, 'device') else 'N/A'}")
                    print(f"  - Base model device: {hooked_model.model.cfg.device}")
                    print(f"  - Masked features count: {len(hooked_model.get_masked_features())}")
                    print(f"  - Steering fraction: {getattr(adapter, 'steering_fraction', 'N/A')}")
            elif masked_features and not full_ft:
                print(f"Warning: Cannot mask features - model doesn't support feature masking")
            
            # Verify model state for consistency debugging
            verify_model_state(hooked_model, exp_name, with_adapter, full_ft)

            # For HookedTransformer without HookedModel wrapper, we need to access the actual PyTorch model
            if hasattr(hooked_model, 'cfg') and hasattr(hooked_model, 'model'):
                # This is a HookedTransformer, get the actual PyTorch model
                hf_model = hooked_model.model
                print(f"Debug: Using HookedTransformer's underlying model")
            else:
                hf_model = hooked_model
                print(f"Debug: Using model directly")
            
            # start with a large batch size and reduce it if we get an error
            initial_batch_size = 16
            batch_size = initial_batch_size
            while True:
                try:
                    eval_model = HFLM(pretrained=hf_model, tokenizer=tokenizer, batch_size=batch_size)
                    
                    # apparently not used anymore
                    #task_manager = lm_eval.tasks.TaskManager()

                    results = lm_eval.simple_evaluate(
                        model=eval_model,
                        tasks=tasks,
                        #task_manager=task_manager,
                        limit=limit,
                        apply_chat_template=True,
                    )
                except Exception as e:
                    if isinstance(e, torch.cuda.OutOfMemoryError):
                        batch_size = batch_size // 2
                        print(f"Error: {e}. Retrying with batch size {batch_size}")
                    else:
                        raise e
                    if batch_size < 1:
                        raise Exception("Batch size of 1 still causes out of memory error. Stopping evaluation.")
                    continue
                break
            
            # Store results with experiment name
            exp_key = f"{run}_{exp_name}" if len(ablation_experiments) > 1 else run
            run_results[exp_name] = results["results"]
            
            # Log to wandb with experiment-specific metrics
            if len(ablation_experiments) > 1:
                wandb.log({
                    f"eval/{run}/{exp_name}/num_masked": len(masked_features),
                    f"experiment": exp_name,
                    f"run": run
                })
        
        summary_results[run] = run_results

    results_file = "eval_results.json"
    with open(results_file, 'w') as f:
        json.dump(summary_results, f, indent=2)
    
    artifact = wandb.Artifact("eval_results", type="results")
    artifact.add_file(results_file)
    wandb.log_artifact(artifact)

    wandb.log(summary_results)
    os.remove(results_file)

    return summary_results

def pretty_results(results):
    for run, run_data in results.items():
        print(f"Run: {run}")
        print("=" * 50)
        
        # Check if run_data contains ablation experiments or direct task results
        if isinstance(run_data, dict) and any(isinstance(v, dict) and 'baseline' in str(v) for v in run_data.values()):
            # This is the new format with ablation experiments
            for exp_name, task_results in run_data.items():
                print(f"  Experiment: {exp_name}")
                print("  " + "-" * 40)
                for task, result in task_results.items():
                    print(f"    Task: {task}")
                    print(f"    Result: {result}")
                    print("    " + "." * 30)
                print()
        else:
            # This is the old format or single experiment
            if isinstance(run_data, dict) and all(isinstance(v, dict) for v in run_data.values()):
                # Multiple experiments format
                for exp_name, task_results in run_data.items():
                    print(f"  Experiment: {exp_name}")
                    print("  " + "-" * 40)
                    for task, result in task_results.items():
                        print(f"    Task: {task}")
                        print(f"    Result: {result}")
                        print("    " + "." * 30)
                    print()
            else:
                # Old single experiment format - direct task results
                for task, result in run_data.items():
                    print(f"  Task: {task}")
                    print(f"  Result: {result}")
                    print("  " + "." * 30)
        
        print("-" * 100)
        print() 