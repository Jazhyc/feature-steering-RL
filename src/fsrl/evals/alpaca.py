import sys
import os
import json
import tempfile
from pathlib import Path
import argparse
import torch
import wandb
from dotenv import load_dotenv
import pandas as pd

from .. import SAEAdapter, HookedModel, BaseHookedModel
from ..utils.wandb_utils import WandBModelDownloader
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
import alpaca_eval


def make_json_serializable(obj):
    """Convert objects to JSON-serializable format, handling DataFrames and other non-serializable types."""
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        # Try to convert to string for other types
        return str(obj)

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
        is_unablated_mode = current_hook is original_hook if original_hook else False
        
        print(f"  - Steering active: {is_steering_active}")
        print(f"  - Unablated mode (full steering enabled): {is_unablated_mode}")
        print(f"  - Hook name: {hook_name}")
        
        # Additional verification for the steered but unablated case
        if exp_name == "unablated":
            if not is_steering_active:
                print(f"  - WARNING: Unablated should have steering enabled, but it's disabled!")
            elif is_steering_active:
                masked_count = len(adapter.get_masked_features()) if hasattr(adapter, 'get_masked_features') else 0
                if masked_count == 0:
                    print(f"  - CORRECT: Unablated has steering enabled with no features masked")
                else:
                    print(f"  - WARNING: Unablated should have no features masked, but {masked_count} features are masked!")
        elif not is_steering_active:
            print(f"  - WARNING: Ablation experiment should have steering enabled, but it's disabled!")
    else:
        print(f"Model State Verification for {exp_name}: No adapter found")


def load_model_and_adapter(run, base_model="google/gemma-2-2b-it", with_adapter=True, full_ft=False, wandb_project_of_adapter="Gemma2-2B-muon"):
    """Load model and adapter using the same logic as evals.py"""
    root = Path(__file__).resolve().parent.parent.parent
    #models_path = f"{root}/models/{wandb_project_of_adapter}" if not full_ft else f"{root}/models/full-gemma2_2B"
    
    load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    
    downloader = WandBModelDownloader(
        entity="feature-steering-RL",
        project=wandb_project_of_adapter if not full_ft else "full-gemma2_2B",
        verbose=True
    )
    
    print(f"##### Loading model for run: {run} #####")
    
    # Search in the correct project based on full_ft flag
    project_name = "feature-steering-rl/full-gemma2_2B" if full_ft else f"feature-steering-rl/{wandb_project_of_adapter}"
    run_objs = downloader.api.runs(project_name, filters={"display_name": run})
    
    if not run_objs:
        raise ValueError(f"No run found with name '{run}' in project '{project_name}'")
    
    downloader.download_model(run_objs[0], wandb_project_of_adapter)
    
    if full_ft:
        base_model_path = downloader.models_base_dir / "full-gemma2_2B" / run / "full_model"
        base_model = BaseHookedModel.from_pretrained(base_model_path, device="cuda", dtype=torch.bfloat16)
        hooked_model = base_model
    else:
        base_model = HookedTransformer.from_pretrained_no_processing(
            base_model, 
            device="cuda", 
            dtype=torch.bfloat16
        )
        
        adapter_path = downloader.models_base_dir / wandb_project_of_adapter / run / "adapter"
        print(f"Loading adapter from: {adapter_path}")
        sae_adapter = SAEAdapter.load_from_pretrained_adapter(adapter_path, device="cuda")
        hooked_model = HookedModel(base_model, sae_adapter)
        
        # Disable steering if not using adapter
        if not with_adapter:
            hooked_model.disable_steering()
            print(f"Evaluating model without steering")
    
    tokenizer = base_model.tokenizer
    return hooked_model, tokenizer

def generate_model_outputs(hooked_model, tokenizer, run_name, exp_name="baseline", limit=10, batch_size=64):
    """Generate outputs for AlpacaEval dataset using our model with batching"""
    import datasets

    eval_data = datasets.load_dataset("tatsu-lab/alpaca_farm", "alpaca_farm_evaluation", trust_remote_code=True)["eval"]
    
    original_len = len(eval_data)
    if limit is not None:
        eval_data = eval_data.select(range(limit))
        print(f"Limited dataset from {original_len} to {len(eval_data)} examples (limit={limit})")
    else:
        print(f"Using full dataset with {len(eval_data)} examples (no limit specified)")
    
    model_outputs = []
    output_filename = f"model_outputs_{run_name}_{exp_name}.json"
    
    print(f"Generating outputs for {len(eval_data)} examples with batch size {batch_size}...")
    print(f"Saving outputs to: {output_filename}")
    
    current_batch_size = batch_size
    min_batch_size = 1
    
    batch_start = 0
    while batch_start < len(eval_data):
        batch_end = min(batch_start + current_batch_size, len(eval_data))
        batch = eval_data.select(range(batch_start, batch_end))
        
        if batch_start % (current_batch_size * 10) == 0:  # Progress every 10 batches
            print(f"Progress: {batch_start}/{len(eval_data)} (batch size: {current_batch_size})")
        
        try:
            prompts = []
            instructions = []
            
            for example in batch:
                instruction = example['instruction']
                instructions.append(instruction)
                
                if hasattr(tokenizer, 'apply_chat_template'):
                    messages = [{"role": "user", "content": instruction}]
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                else:
                    prompt = f"<bos><start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
                
                prompts.append(prompt)
                print("PROMPT: ", prompt)
            
            tokenized = tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=2048
            ).to(hooked_model.device)
            
            with torch.no_grad():
                output_ids = hooked_model.generate(
                    tokenized.input_ids,
                    attention_mask=tokenized.attention_mask,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            batch_outputs = []
            for i, instruction in enumerate(instructions):
                response_ids = output_ids[i][tokenized.input_ids.shape[1]:]
                response = tokenizer.decode(response_ids, skip_special_tokens=True)

                if batch_start == 0 and i == 0:  # Only print first response for debugging
                    print(f"DEBUG - Input length: {tokenized.input_ids.shape[1]}, Output length: {len(output_ids[i])}")
                    print(f"DEBUG - Response IDs shape: {response_ids.shape}, Response length: {len(response)}")
                    print(f"DEBUG - First response: {response[:200]}...")
                
                # Warn about empty responses
                if len(response.strip()) == 0:
                    print(f"WARNING: Empty response generated for instruction: {instruction[:100]}...")
                
                output_entry = {
                    'instruction': instruction,
                    'output': response,
                    'generator': f"hooked_model_{run_name}_{exp_name}"
                }
                model_outputs.append(output_entry)
                batch_outputs.append(output_entry)
            
            batch_start = batch_end
            
            with open(output_filename, 'w') as f: # continuous saving w each batch
                json.dump(model_outputs, f, indent=2)
            
            if wandb.run is not None:
                artifact = wandb.Artifact(f"model_outputs_{run_name}_{exp_name}", type="model_outputs")
                artifact.add_file(output_filename)
                wandb.log_artifact(artifact)
                
                wandb.log({
                    "batch_progress": batch_end,
                    "total_examples": len(eval_data),
                    "completion_percentage": (batch_end / len(eval_data)) * 100,
                    "current_batch_size": len(batch_outputs),
                    "effective_batch_size": current_batch_size
                })
                
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or "cuda out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                
                if current_batch_size <= min_batch_size:
                    print(f"ERROR: OOM even with minimum batch size of {min_batch_size}. Skipping batch {batch_start}-{batch_end}")
                    batch_start = batch_end  # skip this batch and continue
                    continue
                
                new_batch_size = max(current_batch_size // 2, min_batch_size)
                print(f"OOM Error detected. Reducing batch size from {current_batch_size} to {new_batch_size}")
                current_batch_size = new_batch_size
                
                if wandb.run is not None:
                    wandb.log({
                        "oom_error": True,
                        "batch_size_reduced_to": current_batch_size,
                        "failed_at_batch_start": batch_start
                    })
                
                continue # don't increment batch_start, retry with smaller batch size
            else:
                raise e
    
    print(f"Model outputs saved to: {output_filename}")
    return model_outputs

def run_alpaca_eval(runs, base_model="google/gemma-2-2b-it", with_adapter=True, full_ft=False, annotator="config/alpaca_eval/gemini_2_5_flash/configs.yaml", limit=None,
                    alignment_classification_file=None, style_classification_file=None,
                    ablation_experiments=None, wandb_project_of_adapter="Gemma2-2B-muon"):
    """Run AlpacaEval evaluation on the specified runs with ablation experiments"""

    load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(entity="feature-steering-RL", project="alpaca-eval")
    
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
            ("unablated", []),  # Always run baseline first
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
        ablation_experiments = [("unablated", [])]
    else:
        # Ensure baseline is first if custom ablation experiments are provided
        baseline_exists = any(exp[0] == "unablated" for exp in ablation_experiments)
        if not baseline_exists:
            ablation_experiments = [("unablated", [])] + ablation_experiments
        else:
            # Move baseline to front if it exists
            baseline_exp = next((exp for exp in ablation_experiments if exp[0] == "unablated"), None)
            if baseline_exp:
                ablation_experiments = [baseline_exp] + [exp for exp in ablation_experiments if exp[0] != "unablated"]

    summary_results = {}
    
    for run in runs:
        print(f"##### Running AlpacaEval for run: {run} #####")
        print("=" * 30)
        
        hooked_model, tokenizer = load_model_and_adapter(run, base_model, with_adapter, full_ft, wandb_project_of_adapter=wandb_project_of_adapter)

        # Initialize run_results for this run
        run_results = {}
        
        if not with_adapter:
            print(f"Evaluating model without steering")
            exp_name = "baseline"
            model_outputs = generate_model_outputs(hooked_model, tokenizer, run, exp_name, limit)
            eval_results = alpaca_eval.evaluate(
                model_outputs=model_outputs,
                annotators_config=annotator,
                name=f"{run}_{exp_name}_{'with_adapter' if with_adapter else 'no_adapter'}",
                output_path=f"alpaca_eval_results_{run}_{exp_name}",
                max_instances=limit,
                is_return_instead_of_print=True,
            )
            run_results[exp_name] = eval_results
            wandb.log({
                f"eval/{run}/{exp_name}/num_masked": 0,
                f"experiment": exp_name,
                f"run": run
            })
        elif with_adapter:
            # Run ablation experiments for this run
            for exp_name, masked_features in ablation_experiments:
                print(f"\n=== Running experiment: {exp_name} for run: {run} ===")
                print(f"Masking {len(masked_features)} features")

                if hasattr(hooked_model, 'sae_adapter') and not full_ft:
                    hooked_model.enable_steering()
                    hooked_model.clear_masked_features()  # Ensure no features are masked
                    if exp_name == "unablated":
                        print(f"Unablated experiment: steering enabled with no features masked (normal steering)")
                    else:
                        if masked_features:  # Only set masking if we have features to mask
                            hooked_model.set_masked_features(masked_features)
                            print(f"Ablation experiment: steering enabled with {len(masked_features)} features masked")
                    
                    print(f"Active masked features: {len(hooked_model.get_masked_features()) if hasattr(hooked_model, 'get_masked_features') else 'N/A'}")
                elif masked_features and not full_ft:
                    print(f"Warning: Cannot mask features - model doesn't support feature masking")
            
                verify_model_state(hooked_model, exp_name, with_adapter, full_ft)
                model_outputs = generate_model_outputs(hooked_model, tokenizer, run, exp_name, limit)
                
                eval_results = alpaca_eval.evaluate(
                    model_outputs=model_outputs,
                    annotators_config=annotator,
                    name=f"{run}_{exp_name}_{'with_adapter' if with_adapter else 'no_adapter'}",
                    output_path=f"alpaca_eval_results_{run}_{exp_name}",
                    max_instances=limit,
                    is_return_instead_of_print=True,
                )
                
                run_results[exp_name] = eval_results
                wandb.log({
                    f"eval/{run}/{exp_name}/num_masked": len(masked_features),
                    f"experiment": exp_name,
                    f"run": run
                })
        
        summary_results[run] = run_results
        
        del hooked_model, tokenizer
        torch.cuda.empty_cache()
    
    results_file = "alpaca_eval_results.json"
    # Convert to JSON-serializable format
    serializable_results = make_json_serializable(summary_results)
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    artifact = wandb.Artifact("alpaca_eval_results", type="results")
    artifact.add_file(results_file)
    wandb.log_artifact(artifact)

    wandb.log(serializable_results)
    os.remove(results_file)

    return summary_results 