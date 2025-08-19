#!/usr/bin/env python3
"""
Experiment: Analyze whether our SAE adapter steers features that are more 
related to a specific classification (alignment, formatting, etc.) than random chance.

This script:
1. Loads the evaluation dataset used during training
2. Applies the chat template and processes it as in training
3. Analyzes which features are being steered by the adapter
4. Compares steered features against feature classifications (using new standardized format)
5. Computes metrics to determine if steering is better than random

Usage:
    python src/fsrl/scripts/alignment_steering_analysis.py
    python src/fsrl/scripts/alignment_steering_analysis.py --classification_mode alignment --classification_file path/to/alignment_classifications.json
    python src/fsrl/scripts/alignment_steering_analysis.py --classification_mode formatting --classification_file path/to/formatting_classifications.json
    python src/fsrl/scripts/alignment_steering_analysis.py --adapter_path models/custom/path --classification_file path/to/classifications.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from datasets import load_dataset
from transformer_lens import HookedTransformer
from tqdm.auto import tqdm

from fsrl import SAEAdapter, HookedModel, apply_chat_template
# --- FIX 1: Import the function to disable dropout ---
from trl.trainer.utils import disable_dropout_in_model

# Set up environment
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuration based on gemma2_2B.yaml
CONFIG = {
    "model": {
        "name": "gemma-2-2b-it",
        "device": "cuda",
        "dtype": "bfloat16"
    },
    "dataset": {
        "name": "princeton-nlp/llama3-ultrafeedback-armorm",
        "train_split": "train",
        "eval_split": "test",
        "sample_size": None,
        "dataset_num_proc": 20,
        "chat_template": "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] | trim + '\n\n' %}{% set messages = messages[1:] %}{% else %}{% set system_message = '' %}{% endif %}{% for message in messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + content | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"
    },
    "analysis": {
        "append_response": None,  # Options: None, "chosen", "rejected". Default None (prompt only)
        "ignore_attention_mask": False,  # If True, treat all positions as valid (ignore padding)
    }
}

# Default adapter paths - different for alignment vs formatting
DEFAULT_ALIGNMENT_ADAPTER_PATH = "models/Gemma2-2B-clean/mild-glade-10/adapter"
DEFAULT_FORMATTING_ADAPTER_PATH = "models/Gemma2-2B-clean/mild-glade-10/adapter"  # Update this when you have a formatting-trained model (OK?)

# Default classification file paths
DEFAULT_ALIGNMENT_CLASSIFICATION_FILE = "outputs/feature_classification/gemma-2-2b/12-gemmascope-res-65k__l0-21_alignment_classified_deepseek-v3-0324.json"
DEFAULT_FORMATTING_CLASSIFICATION_FILE = "outputs/feature_classification/gemma-2-2b/12-gemmascope-res-65k__l0-21_formatting_classified_deepseek-v3-0324.json"

dtype_map = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_model_and_tokenizer():
    """Load the base model and tokenizer."""
    device = CONFIG["model"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
    dtype = dtype_map.get(CONFIG["model"]["dtype"], torch.bfloat16)
    
    model = HookedTransformer.from_pretrained_no_processing(
        model_name=CONFIG["model"]["name"],
        device=device,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    
    return model, model.tokenizer, device


def load_sae_adapter(adapter_path: str, device: str) -> SAEAdapter:
    """Load the SAE adapter from local path."""
    adapter_path = Path(adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path '{adapter_path}' not found")
    
    sae = SAEAdapter.load_from_pretrained_adapter(adapter_path, device=device)
    return sae


def load_eval_dataset(tokenizer, sample_size: Optional[int] = None):
    """Load and process the evaluation dataset."""
    eval_dataset = load_dataset(
        CONFIG["dataset"]["name"], 
        split=CONFIG["dataset"]["eval_split"]
    )
    
    if sample_size is not None:
        eval_dataset = eval_dataset.select(range(min(sample_size, len(eval_dataset))))
    
    print(f"Loaded {len(eval_dataset)} evaluation samples")
    
    column_names = list(eval_dataset.features)
    append_response = CONFIG["analysis"]["append_response"]
    
    # Always use "simpo" task to ensure we get text_chosen and text_rejected fields
    # even when we only need text_prompt, so the script can handle all modes
    task = "simpo"
    
    if append_response is None:
        print("Using prompt-only text (simpo task, will use text_prompt field)")
    else:
        print(f"Appending {append_response} response to prompt (simpo task, will use text_{append_response} field)")

    eval_dataset = eval_dataset.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": task,
            "chat_template": CONFIG["dataset"]["chat_template"],
        },
        num_proc=CONFIG["dataset"]["dataset_num_proc"],
        remove_columns=column_names,
        desc="Formatting prompts with chat template",
    )
    
    return eval_dataset


def load_feature_classifications(classification_file: str) -> Dict[int, str]:
    """Load feature classifications from JSON file using new standardized format."""
    with open(classification_file, 'r') as f:
        data = json.load(f)
    
    classifications = {}
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        for key in ['features', 'items', 'data']:
            if key in data and isinstance(data[key], list):
                records = data[key]
                break
        else:
            records = [{'feature_id': k, **v} if isinstance(v, dict) else {'feature_id': k, 'label': v} 
                      for k, v in data.items()]
    
    for record in records:
        feature_idx = None
        if 'index' in record and record['index'] is not None:
            try: feature_idx = int(record['index'])
            except: pass
        elif 'feature_id' in record:
            import re
            fid = record['feature_id']
            if isinstance(fid, str):
                m = re.search(r'-(\d+)$', fid)
                if m: feature_idx = int(m.group(1))

        # Use new standardized 'label' column instead of 'classification'
        label = record.get('label')
        if feature_idx is not None and label is not None:
            classifications[feature_idx] = label
    
    print(f"Loaded classifications for {len(classifications)} features")
    return classifications


def analyze_steering_features(
    model: HookedModel, 
    eval_dataset, 
    feature_classifications: Dict[int, str],
    classification_mode: str = "formatting",  # New parameter to specify the analysis mode
    num_samples: Optional[int] = None,
    batch_size: int = 2
) -> Dict[str, any]:
    """
    Analyze which features are being steered and their classification properties.
    
    Args:
        model: The hooked model with SAE adapter
        eval_dataset: Dataset to analyze
        feature_classifications: Dict mapping feature indices to labels ('related'/'not-related')
        classification_mode: The type of classification being analyzed ('alignment', 'formatting', etc.)
        num_samples: Number of samples to analyze
        batch_size: Batch size for processing
    """
    if num_samples is None:
        sample_indices = list(range(len(eval_dataset)))
    else:
        sample_indices = np.random.choice(len(eval_dataset), size=min(num_samples, len(eval_dataset)), replace=False)
    
    print(f"Analyzing steering on {len(sample_indices)} samples...")
    
    all_steered_features = []
    related_steered = []
    not_related_steered = []
    l0_norms = []
    
    # Track position-level statistics for proper averaging
    position_related_ratios = []
    position_total_steered = []
    
    # Initialize feature usage counter for all SAE features
    # Get total number of features from first batch (will be consistent)
    feature_usage_counter = None
    
    model.eval()
    with torch.no_grad():
        num_batches = (len(sample_indices) + batch_size - 1) // batch_size
        batch_iterator = tqdm(range(0, len(sample_indices), batch_size), desc="Processing batches", total=num_batches)
        
        for i in batch_iterator:
            batch_indices = sample_indices[i:i + batch_size]
            final_input_ids_list = []
            append_response = CONFIG["analysis"]["append_response"]
            
            max_length = 2048
            max_prompt_length = 1800

            for idx in batch_indices:
                sample = eval_dataset[int(idx)]
                prompt_text = sample["text_prompt"]
                
                if append_response == "chosen":
                    response_text = sample["text_chosen"]
                elif append_response == "rejected":
                    response_text = sample["text_rejected"]
                else:
                    response_text = ""
                
                prompt_tokens = model.model.tokenizer(prompt_text, add_special_tokens=False)['input_ids']
                response_tokens = model.model.tokenizer(response_text, add_special_tokens=False)['input_ids']

                if len(prompt_tokens) + len(response_tokens) > max_length:
                    prompt_tokens = prompt_tokens[-max_prompt_length:]

                if len(prompt_tokens) + len(response_tokens) > max_length:
                    new_response_len = max_length - len(prompt_tokens)
                    response_tokens = response_tokens[:new_response_len]
                
                final_input_ids_list.append(prompt_tokens + response_tokens)

            batch_tokens = model.model.tokenizer.pad(
                {"input_ids": final_input_ids_list},
                padding=True,
                return_tensors="pt"
            ).to(model.device)
            
            # --- FIX 2: Pass the attention mask to the forward pass ---
            _, llm_cache = model.run_with_cache(
                batch_tokens["input_ids"],
                attention_mask=batch_tokens["attention_mask"], # Pass the mask
                prepend_bos=False
            )
            
            steering_vector = llm_cache["blocks.12.hook_resid_post.hook_sae_adapter"]
            
            if isinstance(steering_vector, torch.Tensor):
                steering_vector = steering_vector.float().cpu().numpy()
            
            # Initialize feature usage counter on first batch
            if feature_usage_counter is None:
                num_features = steering_vector.shape[2]
                feature_usage_counter = np.zeros(num_features, dtype=int)
                print(f"Tracking usage for {num_features} SAE features...")
            
            # --- FIX 3: Replicate the trainer's naive L0 norm calculation ---
            # The trainer's metric is a simple mean over the whole tensor, including padding.
            l0_per_position_batch = np.count_nonzero(np.abs(steering_vector) > 1e-6, axis=-1)
            batch_l0_mean = np.mean(l0_per_position_batch)
            l0_norms.append(batch_l0_mean)

            # For the actual CLASSIFICATION ANALYSIS, we correctly use the attention mask.
            attention_mask = batch_tokens["attention_mask"].cpu().numpy()
            if CONFIG["analysis"]["ignore_attention_mask"]:
                effective_attention_mask = np.ones_like(attention_mask)
            else:
                effective_attention_mask = attention_mask
            
            feature_indices = np.arange(steering_vector.shape[2])
            # Use new standardized labels: 'related' and 'not-related'
            related_mask = np.array([feature_classifications.get(idx) == "related" for idx in feature_indices])
            not_related_mask = np.array([feature_classifications.get(idx) == "not-related" for idx in feature_indices])

            for batch_idx in range(steering_vector.shape[0]):
                sample_len = int(effective_attention_mask[batch_idx].sum())
                if sample_len == 0: continue

                sample_steering = steering_vector[batch_idx, :sample_len, :]
                sample_steered_mask = (np.abs(sample_steering) > 1e-6)
                
                if not np.any(sample_steered_mask): continue

                related_counts = np.sum(sample_steered_mask & related_mask[None, :], axis=1)
                not_related_counts = np.sum(sample_steered_mask & not_related_mask[None, :], axis=1)
                total_classified_counts = related_counts + not_related_counts
                
                valid_positions = total_classified_counts > 0
                if np.any(valid_positions):
                    position_ratios = related_counts[valid_positions] / total_classified_counts[valid_positions]
                    position_related_ratios.extend(position_ratios.tolist())
                    position_total_steered.extend(total_classified_counts[valid_positions].tolist())
                
                steered_features_in_sample = np.where(np.any(sample_steered_mask, axis=0))[0]
                all_steered_features.extend(steered_features_in_sample.tolist())
                
                # Update feature usage counter for this sample
                # Count how many times each feature is used (across all positions in this sample)
                feature_usage_in_sample = np.sum(sample_steered_mask, axis=0)  # Shape: [num_features]
                feature_usage_counter += feature_usage_in_sample.astype(int)
                
                steered_related_features = steered_features_in_sample[related_mask[steered_features_in_sample]]
                steered_not_related_features = steered_features_in_sample[not_related_mask[steered_features_in_sample]]
                
                related_steered.extend(steered_related_features.tolist())
                not_related_steered.extend(steered_not_related_features.tolist())
    
    unique_steered = list(set(all_steered_features))
    unique_related_steered = list(set(related_steered))
    unique_not_related_steered = list(set(not_related_steered))
    
    all_classifications = list(feature_classifications.values())
    total_related = sum(1 for c in all_classifications if c == "related")
    baseline_related_rate = total_related / len(all_classifications) if all_classifications else 0
    
    if len(position_related_ratios) > 0:
        steering_related_rate = float(np.mean(position_related_ratios))
        total_positions_analyzed = len(position_related_ratios)
        mean_steered_per_position = float(np.mean(position_total_steered))
    else:
        steering_related_rate = 0.0
        total_positions_analyzed = 0
        mean_steered_per_position = 0.0
    
    l0_norms_array = np.array(l0_norms)
    l0_mean = float(np.mean(l0_norms_array)) if len(l0_norms_array) > 0 else 0.0
    l0_std = float(np.std(l0_norms_array)) if len(l0_norms_array) > 0 else 0.0
    l0_stderr = float(l0_std / np.sqrt(len(l0_norms_array))) if len(l0_norms_array) > 0 else 0.0
    
    results = {
        "configuration": {
            "append_response": CONFIG["analysis"]["append_response"],
            "ignore_attention_mask": CONFIG["analysis"]["ignore_attention_mask"],
            "classification_mode": classification_mode,
        },
        "total_steered_features": len(unique_steered),
        "related_steered": len(unique_related_steered),
        "not_related_steered": len(unique_not_related_steered),
        "total_positions_analyzed": total_positions_analyzed,
        "mean_steered_per_position": mean_steered_per_position,
        "baseline_related_rate": float(baseline_related_rate),
        "steering_related_rate": float(steering_related_rate),
        "improvement_over_baseline": float(steering_related_rate - baseline_related_rate),
        "l0_norm_mean": l0_mean,
        "l0_norm_std": l0_std,
        "l0_norm_stderr": l0_stderr,
        "feature_usage_counter": feature_usage_counter.tolist() if feature_usage_counter is not None else [],
        # Raw data for statistical testing
        "position_level_data": {
            "position_related_ratios": position_related_ratios,
            "position_total_steered": position_total_steered,
            "num_positions": len(position_related_ratios)
        }
    }
    
    return results


def save_feature_usage_analysis(feature_usage_counter, feature_classifications, output_file):
    """
    Save detailed feature usage statistics to a separate file.
    """
    if feature_usage_counter is None or len(feature_usage_counter) == 0:
        print("No feature usage data to save.")
        return
    
    feature_usage_counter = np.array(feature_usage_counter)
    
    # Create detailed feature usage analysis
    feature_usage_data = []
    total_usage = np.sum(feature_usage_counter)
    
    for feature_idx in range(len(feature_usage_counter)):
        usage_count = int(feature_usage_counter[feature_idx])
        usage_percentage = float(usage_count / total_usage * 100) if total_usage > 0 else 0.0
        
        classification = feature_classifications.get(feature_idx, "unknown")
        
        feature_usage_data.append({
            "feature_index": feature_idx,
            "usage_count": usage_count,
            "usage_percentage": usage_percentage,
            "classification": classification
        })
    
    # Sort by usage count (most used first)
    feature_usage_data.sort(key=lambda x: x["usage_count"], reverse=True)
    
    # Calculate summary statistics
    active_features = np.sum(feature_usage_counter > 0)
    top_10_usage = np.sum([f["usage_count"] for f in feature_usage_data[:10]])
    top_10_percentage = float(top_10_usage / total_usage * 100) if total_usage > 0 else 0.0
    
    # Count by classification
    classification_usage = {}
    for feature in feature_usage_data:
        cls = feature["classification"]
        if cls not in classification_usage:
            classification_usage[cls] = {"count": 0, "total_usage": 0}
        if feature["usage_count"] > 0:
            classification_usage[cls]["count"] += 1
        classification_usage[cls]["total_usage"] += feature["usage_count"]
    
    # Create summary
    usage_summary = {
        "total_features": len(feature_usage_counter),
        "active_features": int(active_features),
        "inactive_features": int(len(feature_usage_counter) - active_features),
        "total_usage_events": int(total_usage),
        "mean_usage_per_active_feature": float(np.mean(feature_usage_counter[feature_usage_counter > 0])) if active_features > 0 else 0.0,
        "std_usage_per_active_feature": float(np.std(feature_usage_counter[feature_usage_counter > 0])) if active_features > 0 else 0.0,
        "top_10_features_usage_percentage": top_10_percentage,
        "classification_breakdown": classification_usage
    }
    
    # Combine everything
    full_usage_analysis = {
        "summary": usage_summary,
        "feature_usage_details": feature_usage_data
    }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(full_usage_analysis, f, indent=2)
    
    print(f"Feature usage analysis saved to: {output_file}")
    print(f"  - {active_features}/{len(feature_usage_counter)} features were used")
    print(f"  - Top 10 features account for {top_10_percentage:.1f}% of all usage")
    print(f"  - Mean usage per active feature: {usage_summary['mean_usage_per_active_feature']:.1f}")


def run_single_experiment(
    model, 
    eval_dataset, 
    classification_mode: str, 
    append_response: str, 
    ignore_attention_mask: bool,
    classification_file: str,
    output_base_dir: str,
    num_analysis_samples: Optional[int] = None,
    batch_size: int = 2
) -> dict:
    """Run a single experiment configuration."""
    
    # Set global config for this experiment
    CONFIG["analysis"]["append_response"] = append_response
    CONFIG["analysis"]["ignore_attention_mask"] = ignore_attention_mask
    
    print(f"\n{'='*80}")
    print(f"RUNNING EXPERIMENT: {classification_mode.upper()}")
    print(f"  - Append response: {append_response}")
    print(f"  - Ignore attention mask: {ignore_attention_mask}")
    print(f"  - Classification file: {Path(classification_file).name}")
    print(f"{'='*80}")
    
    # Load feature classifications for this mode
    feature_classifications = load_feature_classifications(classification_file)
    
    # Run analysis
    results = analyze_steering_features(
        model, 
        eval_dataset, 
        feature_classifications, 
        classification_mode=classification_mode,
        num_samples=num_analysis_samples,
        batch_size=batch_size
    )
    
    # Generate output filename
    response_suffix = f"_prompt_{append_response}" if append_response else "_prompt_only"
    mask_suffix = "_ignore_mask" if ignore_attention_mask else ""
    output_file = Path(output_base_dir) / f"{classification_mode}_steering_analysis{response_suffix}{mask_suffix}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save main results (without detailed feature usage)
    main_results = {k: v for k, v in results.items() if k != "feature_usage_counter"}
    with open(output_file, 'w') as f:
        json.dump(main_results, f, indent=2, sort_keys=True)
    
    # Save detailed feature usage
    feature_usage_file = str(output_file).replace(".json", "_feature_usage.json")
    if "feature_usage_counter" in results and results["feature_usage_counter"]:
        save_feature_usage_analysis(
            results["feature_usage_counter"], 
            feature_classifications, 
            feature_usage_file
        )
    
    # Print summary
    print(f"\n{classification_mode.upper()} STEERING ANALYSIS RESULTS")
    print(f"Append response: {append_response}, Ignore mask: {ignore_attention_mask}")
    print(f"Total unique features steered: {results['total_steered_features']}")
    print(f"Unique {classification_mode}-related features steered: {results['related_steered']}")
    print(f"Unique not-{classification_mode}-related features steered: {results['not_related_steered']}")
    print(f"Baseline {classification_mode}-related rate: {results['baseline_related_rate']:.3f}")
    print(f"Steering {classification_mode}-related rate: {results['steering_related_rate']:.3f}")
    print(f"Improvement over baseline: {results['improvement_over_baseline']:.3f}")
    
    if results['improvement_over_baseline'] > 0:
        print(f"✅ POSITIVE: Steers more {classification_mode}-related features than random")
    else:
        print(f"❌ NEGATIVE: Does not preferentially steer {classification_mode}-related features")
    
    print(f"Results saved to: {output_file}")
    print(f"Feature usage saved to: {feature_usage_file}")
    
    return results


def run_all_experiments(
    alignment_adapter_path: str,
    formatting_adapter_path: str,
    alignment_classification_file: str,
    formatting_classification_file: str,
    output_base_dir: str,
    sample_size: Optional[int] = None,
    num_analysis_samples: Optional[int] = None,
    batch_size: int = 2
) -> dict:
    """Run all experiment combinations: alignment/formatting × prompt/chosen/rejected × ignore_mask=True."""
    
    print("="*100)
    print("RUNNING COMPREHENSIVE STEERING ANALYSIS")
    print("Testing: alignment/formatting × prompt/chosen/rejected × ignore_attention_mask=True")
    print("="*100)
    
    all_results = {}
    
    # Define experiment configurations
    classification_configs = [
        ("alignment", alignment_adapter_path, alignment_classification_file),
        ("formatting", formatting_adapter_path, formatting_classification_file)
    ]
    
    append_response_options = [None, "chosen", "rejected"]  # None = prompt only
    ignore_attention_mask = True  # Fixed to True as requested
    
    for classification_mode, adapter_path, classification_file in classification_configs:
        print(f"\n{'='*60}")
        print(f"LOADING MODEL FOR {classification_mode.upper()} ANALYSIS")
        print(f"Adapter path: {adapter_path}")
        print(f"{'='*60}")
        
        # Load model for this classification mode
        base_model, tokenizer, device = load_model_and_tokenizer()
        sae = load_sae_adapter(adapter_path, device)
        model = HookedModel(base_model, sae)
        
        # Disable dropout to match trainer state
        from trl.trainer.utils import disable_dropout_in_model
        disable_dropout_in_model(model)
        model.eval()
        
        # Load evaluation dataset (only once per classification mode)
        eval_dataset = load_eval_dataset(tokenizer, sample_size=sample_size)
        
        # Run experiments for each append_response option
        for append_response in append_response_options:
            experiment_key = f"{classification_mode}_{append_response or 'prompt_only'}_ignore_mask"
            
            try:
                results = run_single_experiment(
                    model=model,
                    eval_dataset=eval_dataset,
                    classification_mode=classification_mode,
                    append_response=append_response,
                    ignore_attention_mask=ignore_attention_mask,
                    classification_file=classification_file,
                    output_base_dir=output_base_dir,
                    num_analysis_samples=num_analysis_samples,
                    batch_size=batch_size
                )
                
                all_results[experiment_key] = results
                
            except Exception as e:
                print(f"❌ ERROR in experiment {experiment_key}: {e}")
                all_results[experiment_key] = {"error": str(e)}
        
        # Clean up GPU memory between classification modes
        del model, sae, base_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Save comprehensive summary
    summary_file = Path(output_base_dir) / "comprehensive_steering_analysis_summary.json"
    with open(summary_file, 'w') as f:
        # Save summary without feature_usage_counter to keep file manageable
        summary_results = {}
        for key, result in all_results.items():
            if isinstance(result, dict) and "feature_usage_counter" in result:
                summary_results[key] = {k: v for k, v in result.items() if k != "feature_usage_counter"}
            else:
                summary_results[key] = result
        json.dump(summary_results, f, indent=2, sort_keys=True)
    
    print(f"\n{'='*100}")
    print("COMPREHENSIVE ANALYSIS COMPLETE")
    print(f"Summary saved to: {summary_file}")
    print("Individual experiment results saved in output directory")
    print(f"{'='*100}")
    
def main():
    parser = argparse.ArgumentParser(description="Analyze steering in SAE adapter for various classification modes")
    parser.add_argument("--run_all_experiments", action="store_true", help="Run comprehensive experiments: alignment/formatting × prompt/chosen/rejected × ignore_mask=True")
    parser.add_argument("--alignment_adapter_path", type=str, default=DEFAULT_ALIGNMENT_ADAPTER_PATH, help=f"Path to alignment-trained SAE adapter (default: {DEFAULT_ALIGNMENT_ADAPTER_PATH})")
    parser.add_argument("--formatting_adapter_path", type=str, default=DEFAULT_FORMATTING_ADAPTER_PATH, help=f"Path to formatting-trained SAE adapter (default: {DEFAULT_FORMATTING_ADAPTER_PATH})")
    parser.add_argument("--adapter_path", type=str, help="Path to SAE adapter (for single experiments)")
    parser.add_argument("--alignment_classification_file", type=str, default=DEFAULT_ALIGNMENT_CLASSIFICATION_FILE, help=f"Path to alignment classification JSON file (default: {DEFAULT_ALIGNMENT_CLASSIFICATION_FILE})")
    parser.add_argument("--formatting_classification_file", type=str, default=DEFAULT_FORMATTING_CLASSIFICATION_FILE, help=f"Path to formatting classification JSON file (default: {DEFAULT_FORMATTING_CLASSIFICATION_FILE})")
    parser.add_argument("--classification_file", type=str, help="Path to feature classification JSON file (for single experiments)")
    parser.add_argument("--classification_mode", type=str, default="formatting", help="Classification mode being analyzed (e.g., 'alignment', 'formatting') - used for output naming and reporting (default: formatting)")
    parser.add_argument("--sample_size", type=int, default=None, help="Limit dataset size (for testing)")
    parser.add_argument("--num_analysis_samples", type=int, default=None, help="Number of samples to use for steering analysis (default: use all)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for processing samples (default: 2)")
    parser.add_argument("--output_file", type=str, help="Path to save analysis results (for single experiments)")
    parser.add_argument("--output_base_dir", type=str, default="outputs/feature_classification/comprehensive_analysis", help="Base directory for comprehensive experiment outputs")
    parser.add_argument("--append_response", type=str, choices=[None, "chosen", "rejected"], default=None, help="Append response to prompt: None (prompt only, default), 'chosen' (prompt+chosen), 'rejected' (prompt+rejected)")
    parser.add_argument("--ignore_attention_mask", action="store_true", help="Ignore attention mask (treat all positions as valid, including padding)")
    
    args = parser.parse_args()
    
    if args.run_all_experiments:
        # Run comprehensive experiments
        print("Running comprehensive steering analysis...")
        results = run_all_experiments(
            alignment_adapter_path=args.alignment_adapter_path,
            formatting_adapter_path=args.formatting_adapter_path,
            alignment_classification_file=args.alignment_classification_file,
            formatting_classification_file=args.formatting_classification_file,
            output_base_dir=args.output_base_dir,
            sample_size=args.sample_size,
            num_analysis_samples=args.num_analysis_samples,
            batch_size=args.batch_size
        )
        return results
    
    else:
        # Run single experiment (existing behavior)
        # Determine defaults based on classification mode
        if not args.adapter_path:
            args.adapter_path = DEFAULT_ALIGNMENT_ADAPTER_PATH if args.classification_mode == "alignment" else DEFAULT_FORMATTING_ADAPTER_PATH
        
        if not args.classification_file:
            args.classification_file = DEFAULT_ALIGNMENT_CLASSIFICATION_FILE if args.classification_mode == "alignment" else DEFAULT_FORMATTING_CLASSIFICATION_FILE
        
        CONFIG["analysis"]["append_response"] = args.append_response
        CONFIG["analysis"]["ignore_attention_mask"] = args.ignore_attention_mask
        
        if not args.output_file:
            response_suffix = "_prompt_chosen" if CONFIG["analysis"]["append_response"] == "chosen" else ("_prompt_rejected" if CONFIG["analysis"]["append_response"] == "rejected" else "_prompt_only")
            mask_suffix = "_ignore_mask" if CONFIG["analysis"]["ignore_attention_mask"] else ""
            args.output_file = f"outputs/feature_classification/{args.classification_mode}_steering_analysis{response_suffix}{mask_suffix}.json"
        
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        
        print("Loading model and tokenizer...")
        base_model, tokenizer, device = load_model_and_tokenizer()
        
        print("Loading SAE adapter...")
        sae = load_sae_adapter(args.adapter_path, device)
        
        print("Creating hooked model...")
        model = HookedModel(base_model, sae)
        
        # --- FIX 1 (continued): Apply the dropout fix to the model ---
        # The SimPOTrainer permanently disables dropout layers. To replicate its
        # evaluation environment, we must do the same.
        print("Disabling dropout in the model to match the trainer's state...")
        disable_dropout_in_model(model)
        model.eval()
        
        print("Loading evaluation dataset...")
        eval_dataset = load_eval_dataset(tokenizer, sample_size=args.sample_size)
        
        print("Loading feature classifications...")
        feature_classifications = load_feature_classifications(args.classification_file)
        
        print("Analyzing steering features...")
        results = analyze_steering_features(
            model, 
            eval_dataset, 
            feature_classifications, 
            classification_mode=args.classification_mode,
            num_samples=args.num_analysis_samples,
            batch_size=args.batch_size
        )
        
        print("\n" + "="*60)
        print(f"{args.classification_mode.upper()} STEERING ANALYSIS RESULTS")
        print("="*60)
        print(f"Configuration: append_response = {CONFIG['analysis']['append_response']}")
        print(f"Classification mode: {args.classification_mode}")
        print(f"Total unique features steered: {results['total_steered_features']}")
        print(f"Unique {args.classification_mode}-related features steered: {results['related_steered']}")
        print(f"Unique not-{args.classification_mode}-related features steered: {results['not_related_steered']}")
        print(f"Total positions analyzed: {results['total_positions_analyzed']}")
        print(f"Mean features steered per position (masked): {results['mean_steered_per_position']:.2f}")
        print()
        print(f"Baseline {args.classification_mode}-related rate (all features): {results['baseline_related_rate']:.3f}")
        print(f"Position-averaged steering {args.classification_mode}-related rate: {results['steering_related_rate']:.3f}")
        print(f"Improvement over baseline: {results['improvement_over_baseline']:.3f}")
        print()
        print("L0 Norm Statistics (replicating trainer's metric):")
        print(f"L0 norm mean: {results['l0_norm_mean']:.2f}")
        print(f"L0 norm std: {results['l0_norm_std']:.2f}")
        print(f"L0 norm stderr: {results['l0_norm_stderr']:.2f}")
        print()
        
        if results['improvement_over_baseline'] > 0:
            print(f"✅ POSITIVE RESULT: Adapter steers more {args.classification_mode}-related features than random!")
        else:
            print(f"❌ NEGATIVE RESULT: Adapter does not preferentially steer {args.classification_mode}-related features")
        
        print("="*60)
        
        # Save main analysis results (without detailed feature usage)
        main_results = {k: v for k, v in results.items() if k != "feature_usage_counter"}
        with open(args.output_file, 'w') as f:
            json.dump(main_results, f, indent=2, sort_keys=True)
        
        print(f"\nMain results saved to: {args.output_file}")
        
        # Generate feature usage filename and save detailed feature usage
        base_name = args.output_file.replace(".json", "")
        feature_usage_file = f"{base_name}_feature_usage.json"
        
        if "feature_usage_counter" in results and results["feature_usage_counter"]:
            save_feature_usage_analysis(
                results["feature_usage_counter"], 
                feature_classifications, 
                feature_usage_file
            )
        else:
            print("No feature usage data available to save.")
        
        return results


if __name__ == "__main__":
    main()