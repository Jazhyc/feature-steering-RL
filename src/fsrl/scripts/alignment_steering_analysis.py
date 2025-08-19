#!/usr/bin/env python3
"""
Experiment: Analyze whether our SAE adapter steers features that are more 
alignment-related than random chance.

This script:
1. Loads the evaluation dataset used during training
2. Applies the chat template and processes it as in training
3. Analyzes which features are being steered by the adapter
4. Compares steered features against alignment classifications
5. Computes metrics to determine if steering is better than random

Usage:
    python src/fsrl/scripts/alignment_steering_analysis.py
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

# Default adapter path
DEFAULT_ADAPTER_PATH = "models/Gemma2-2B-clean/mild-glade-10/adapter"

# Default classification file path
DEFAULT_CLASSIFICATION_FILE = "models/NeuronpediaCache/gemma-2-2b/12-gemmascope-res-65k__l0-21_classified_deepseek-v3-0324.json"

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
    
    if append_response is None:
        task = "simpo_generation"
        print("Using prompt-only text (simpo_generation task)")
    else:
        task = "simpo"
        print(f"Appending {append_response} response to prompt (matching training behavior)")

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
    """Load feature classifications from JSON file."""
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
            records = [{'feature_id': k, **v} if isinstance(v, dict) else {'feature_id': k, 'classification': v} 
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

        classification = record.get('classification')
        if feature_idx is not None and classification is not None:
            classifications[feature_idx] = classification
    
    print(f"Loaded classifications for {len(classifications)} features")
    return classifications


def analyze_steering_features(
    model: HookedModel, 
    eval_dataset, 
    feature_classifications: Dict[int, str],
    num_samples: Optional[int] = None,
    batch_size: int = 2
) -> Dict[str, any]:
    """
    Analyze which features are being steered and their alignment properties.
    """
    if num_samples is None:
        sample_indices = list(range(len(eval_dataset)))
    else:
        sample_indices = np.random.choice(len(eval_dataset), size=min(num_samples, len(eval_dataset)), replace=False)
    
    print(f"Analyzing steering on {len(sample_indices)} samples...")
    
    all_steered_features = []
    alignment_related_steered = []
    not_alignment_related_steered = []
    l0_norms = []
    
    position_alignment_ratios = []
    position_total_steered = []
    
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
            
            # --- FIX 3: Replicate the trainer's naive L0 norm calculation ---
            # The trainer's metric is a simple mean over the whole tensor, including padding.
            l0_per_position_batch = np.count_nonzero(np.abs(steering_vector) > 1e-6, axis=-1)
            batch_l0_mean = np.mean(l0_per_position_batch)
            l0_norms.append(batch_l0_mean)

            # For the actual ALIGNMENT ANALYSIS, we correctly use the attention mask.
            attention_mask = batch_tokens["attention_mask"].cpu().numpy()
            if CONFIG["analysis"]["ignore_attention_mask"]:
                effective_attention_mask = np.ones_like(attention_mask)
            else:
                effective_attention_mask = attention_mask
            
            feature_indices = np.arange(steering_vector.shape[2])
            alignment_mask = np.array([feature_classifications.get(idx) == "alignment-related" for idx in feature_indices])
            not_alignment_mask = np.array([feature_classifications.get(idx) == "not-alignment-related" for idx in feature_indices])

            for batch_idx in range(steering_vector.shape[0]):
                sample_len = int(effective_attention_mask[batch_idx].sum())
                if sample_len == 0: continue

                sample_steering = steering_vector[batch_idx, :sample_len, :]
                sample_steered_mask = (np.abs(sample_steering) > 1e-6)
                
                if not np.any(sample_steered_mask): continue

                alignment_counts = np.sum(sample_steered_mask & alignment_mask[None, :], axis=1)
                not_alignment_counts = np.sum(sample_steered_mask & not_alignment_mask[None, :], axis=1)
                total_classified_counts = alignment_counts + not_alignment_counts
                
                valid_positions = total_classified_counts > 0
                if np.any(valid_positions):
                    position_ratios = alignment_counts[valid_positions] / total_classified_counts[valid_positions]
                    position_alignment_ratios.extend(position_ratios.tolist())
                    position_total_steered.extend(total_classified_counts[valid_positions].tolist())
                
                steered_features_in_sample = np.where(np.any(sample_steered_mask, axis=0))[0]
                all_steered_features.extend(steered_features_in_sample.tolist())
                
                steered_alignment_features = steered_features_in_sample[alignment_mask[steered_features_in_sample]]
                steered_not_alignment_features = steered_features_in_sample[not_alignment_mask[steered_features_in_sample]]
                
                alignment_related_steered.extend(steered_alignment_features.tolist())
                not_alignment_related_steered.extend(steered_not_alignment_features.tolist())
    
    unique_steered = list(set(all_steered_features))
    unique_alignment_steered = list(set(alignment_related_steered))
    unique_not_alignment_steered = list(set(not_alignment_related_steered))
    
    all_classifications = list(feature_classifications.values())
    total_alignment = sum(1 for c in all_classifications if c == "alignment-related")
    baseline_alignment_rate = total_alignment / len(all_classifications) if all_classifications else 0
    
    if len(position_alignment_ratios) > 0:
        steering_alignment_rate = float(np.mean(position_alignment_ratios))
        total_positions_analyzed = len(position_alignment_ratios)
        mean_steered_per_position = float(np.mean(position_total_steered))
    else:
        steering_alignment_rate = 0.0
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
        },
        "total_steered_features": len(unique_steered),
        "alignment_related_steered": len(unique_alignment_steered),
        "not_alignment_related_steered": len(unique_not_alignment_steered),
        "total_positions_analyzed": total_positions_analyzed,
        "mean_steered_per_position": mean_steered_per_position,
        "baseline_alignment_rate": float(baseline_alignment_rate),
        "steering_alignment_rate": float(steering_alignment_rate),
        "improvement_over_baseline": float(steering_alignment_rate - baseline_alignment_rate),
        "l0_norm_mean": l0_mean,
        "l0_norm_std": l0_std,
        "l0_norm_stderr": l0_stderr,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze alignment-related steering in SAE adapter")
    parser.add_argument("--adapter_path", type=str, default=DEFAULT_ADAPTER_PATH, help=f"Path to the trained SAE adapter (default: {DEFAULT_ADAPTER_PATH})")
    parser.add_argument("--classification_file", type=str, default=DEFAULT_CLASSIFICATION_FILE, help=f"Path to feature classification JSON file (default: {DEFAULT_CLASSIFICATION_FILE})")
    parser.add_argument("--sample_size", type=int, default=None, help="Limit dataset size (for testing)")
    parser.add_argument("--num_analysis_samples", type=int, default=None, help="Number of samples to use for steering analysis (default: use all)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for processing samples (default: 4)")
    parser.add_argument("--output_file", type=str, default="outputs/alignment_steering_analysis.json", help="Path to save analysis results")
    parser.add_argument("--append_response", type=str, choices=[None, "chosen", "rejected"], default=None, help="Append response to prompt: None (prompt only, default), 'chosen' (prompt+chosen), 'rejected' (prompt+rejected)")
    parser.add_argument("--ignore_attention_mask", action="store_true", help="Ignore attention mask (treat all positions as valid, including padding)")
    
    args = parser.parse_args()
    
    CONFIG["analysis"]["append_response"] = args.append_response
    CONFIG["analysis"]["ignore_attention_mask"] = args.ignore_attention_mask
    
    if args.output_file == "outputs/alignment_steering_analysis.json":
        response_suffix = "_prompt_chosen" if CONFIG["analysis"]["append_response"] == "chosen" else ("_prompt_rejected" if CONFIG["analysis"]["append_response"] == "rejected" else "_prompt_only")
        mask_suffix = "_ignore_mask" if CONFIG["analysis"]["ignore_attention_mask"] else ""
        args.output_file = f"outputs/alignment_steering_analysis{response_suffix}{mask_suffix}.json"
    
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
        num_samples=args.num_analysis_samples,
        batch_size=args.batch_size
    )
    
    print("\n" + "="*60)
    print("ALIGNMENT STEERING ANALYSIS RESULTS")
    print("="*60)
    print(f"Configuration: append_response = {CONFIG['analysis']['append_response']}")
    print(f"Total unique features steered: {results['total_steered_features']}")
    print(f"Unique alignment-related features steered: {results['alignment_related_steered']}")
    print(f"Unique not alignment-related features steered: {results['not_alignment_related_steered']}")
    print(f"Total positions analyzed: {results['total_positions_analyzed']}")
    print(f"Mean features steered per position (masked): {results['mean_steered_per_position']:.2f}")
    print()
    print(f"Baseline alignment rate (all features): {results['baseline_alignment_rate']:.3f}")
    print(f"Position-averaged steering alignment rate: {results['steering_alignment_rate']:.3f}")
    print(f"Improvement over baseline: {results['improvement_over_baseline']:.3f}")
    print()
    print("L0 Norm Statistics (replicating trainer's metric):")
    print(f"L0 norm mean: {results['l0_norm_mean']:.2f}")
    print(f"L0 norm std: {results['l0_norm_std']:.2f}")
    print()
    
    if results['improvement_over_baseline'] > 0:
        print("✅ POSITIVE RESULT: Adapter steers more alignment-related features than random!")
    else:
        print("❌ NEGATIVE RESULT: Adapter does not preferentially steer alignment-related features")
    
    print("="*60)
    
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2, sort_keys=True)
    
    print(f"\nDetailed results saved to: {args.output_file}")


if __name__ == "__main__":
    main()