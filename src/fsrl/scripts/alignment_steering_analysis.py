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
    }
}

# Default adapter path
DEFAULT_ADAPTER_PATH = "models/Gemma2_2B-clean/mild-glade-10/adapter"

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
    # Load the evaluation split
    eval_dataset = load_dataset(
        CONFIG["dataset"]["name"], 
        split=CONFIG["dataset"]["eval_split"]
    )
    
    # Limit dataset size for testing if specified
    if sample_size is not None:
        eval_dataset = eval_dataset.select(range(min(sample_size, len(eval_dataset))))
    
    print(f"Loaded {len(eval_dataset)} evaluation samples")
    
    # Get original column names to remove after processing
    column_names = list(eval_dataset.features)
    
    # Determine task based on append_response configuration
    append_response = CONFIG["analysis"]["append_response"]
    
    if append_response is None:
        # Default: prompt only (for generation)
        task = "simpo_generation"
        print("Using prompt-only text (simpo_generation task)")
    elif append_response == "chosen":
        # Append chosen response to prompt (as in training)
        task = "simpo"  # This provides prompt + chosen and prompt + rejected
        print("Appending chosen response to prompt (matching training behavior)")
    elif append_response == "rejected":
        # Append rejected response to prompt  
        task = "simpo"  # This provides prompt + chosen and prompt + rejected
        print("Appending rejected response to prompt")
    else:
        raise ValueError(f"Invalid append_response option: {append_response}. Must be None, 'chosen', or 'rejected'")
    
    # Apply chat template
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
    
    # Handle different JSON structures
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        # Look for common keys that contain the list
        for key in ['features', 'items', 'data']:
            if key in data and isinstance(data[key], list):
                records = data[key]
                break
        else:
            # Convert dict to list of records
            records = [{'feature_id': k, **v} if isinstance(v, dict) else {'feature_id': k, 'classification': v} 
                      for k, v in data.items()]
    
    # Extract feature index and classification
    for record in records:
        # Try to extract feature index
        feature_idx = None
        if 'index' in record and record['index'] is not None:
            try:
                feature_idx = int(record['index'])
            except:
                pass
        elif 'feature_id' in record:
            # Try to extract index from feature_id
            import re
            fid = record['feature_id']
            if isinstance(fid, str):
                m = re.search(r'-(\d+)$', fid)
                if m:
                    feature_idx = int(m.group(1))

        # Get classification
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
    # Use all samples if num_samples is None, otherwise sample
    if num_samples is None:
        sample_indices = list(range(len(eval_dataset)))
        print(f"Analyzing steering on all {len(sample_indices)} samples...")
    else:
        sample_indices = np.random.choice(len(eval_dataset), size=min(num_samples, len(eval_dataset)), replace=False)
        print(f"Analyzing steering on {len(sample_indices)} samples...")
    
    all_steered_features = []
    alignment_related_steered = []
    not_alignment_related_steered = []
    l0_norms = []  # Track L0 norms for each sample
    
    # Track position-level statistics for proper averaging
    position_alignment_ratios = []  # Ratio of alignment features per position
    position_total_steered = []     # Total steered features per position
    
    model.eval()
    with torch.no_grad():
        # Process samples in batches with progress bar
        num_batches = (len(sample_indices) + batch_size - 1) // batch_size
        batch_iterator = tqdm(
            range(0, len(sample_indices), batch_size), 
            desc="Processing batches",
            total=num_batches,
            unit="batch"
        )
        
        for i in batch_iterator:
            batch_indices = sample_indices[i:i + batch_size]
            
            # Update progress bar with current batch info
            batch_iterator.set_postfix({
                'samples': f"{i + len(batch_indices)}/{len(sample_indices)}",
                'batch_size': len(batch_indices)
            })
            
            # Prepare batch of prompts based on configuration
            batch_prompts = []
            append_response = CONFIG["analysis"]["append_response"]
            
            for idx in batch_indices:
                sample = eval_dataset[int(idx)]
                prompt = sample["text_prompt"]
                
                if append_response == "chosen":
                    chosen_response = sample["text_chosen"]
                    text = prompt + chosen_response
                elif append_response == "rejected":
                    rejected_response = sample["text_rejected"]
                    text = prompt + rejected_response
                else:
                    text = sample["text_prompt"]
                    
                batch_prompts.append(text)
            
            # Tokenize the batch
            batch_tokens = model.model.tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1800,
                padding=True
            )
            
            # Use run_with_cache to capture steering vectors
            _, llm_cache = model.run_with_cache(batch_tokens["input_ids"], prepend_bos=False)
            
            # Get the steering vector from the hook_sae_adapter key
            steering_vector = llm_cache["blocks.12.hook_resid_post.hook_sae_adapter"]
            
            # Convert to numpy if it's a tensor
            if isinstance(steering_vector, torch.Tensor):
                steering_vector = steering_vector.float().cpu().numpy()
            
            # Handle batch and sequence dimensions
            # steering_vector shape: [batch_size, seq_len, num_features]
            
            # 1. L0 NORM CALCULATION: Replicate the trainer's "true mean" over the entire padded batch
            # This matches: torch.sum(sparsity_mask, dim=-1).mean() from get_steering_vector()
            l0_per_position_batch = np.count_nonzero(np.abs(steering_vector) > 1e-6, axis=-1)  # Shape: [batch, seq]
            batch_l0_mean = np.mean(l0_per_position_batch)  # True mean across ALL positions (including padding)
            l0_norms.append(batch_l0_mean)

            # 2. ALIGNMENT ANALYSIS: Use attention mask to analyze only real tokens
            attention_mask = batch_tokens["attention_mask"].cpu().numpy()  # [batch_size, seq_len]
            
            # The BOS token is prepended by HookedTransformer, so we need to add it to our mask
            bos_mask = np.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype)
            full_mask = np.concatenate([bos_mask, attention_mask], axis=1)  # [batch_size, seq_len + 1]
            
            # Create classification masks for vectorized operations (outside the loop for efficiency)
            feature_indices = np.arange(steering_vector.shape[2])  # [features]
            alignment_mask = np.array([
                feature_classifications.get(idx) == "alignment-related" 
                for idx in feature_indices
            ])  # [features]
            not_alignment_mask = np.array([
                feature_classifications.get(idx) == "not-alignment-related" 
                for idx in feature_indices
            ])  # [features]

            # Process each sample in the batch for alignment analysis
            for batch_idx in range(steering_vector.shape[0]):
                # Get the true length of the sequence including BOS
                sample_len = int(full_mask[batch_idx].sum())
                if sample_len == 0:
                    continue

                # We only want to analyze features from REAL tokens (non-padded)
                sample_steering = steering_vector[batch_idx, :sample_len, :]  # [real_seq_len, features]
                sample_steered_mask = (np.abs(sample_steering) > 1e-6)  # [real_seq_len, features]
                
                # Check if any steering happens in the real part of this sample
                if not np.any(sample_steered_mask):
                    continue

                # Vectorized calculation of alignment counts per position (for real tokens only)
                alignment_counts = np.sum(sample_steered_mask & alignment_mask[None, :], axis=1)  # [real_seq_len]
                not_alignment_counts = np.sum(sample_steered_mask & not_alignment_mask[None, :], axis=1)  # [real_seq_len]
                total_classified_counts = alignment_counts + not_alignment_counts  # [real_seq_len]
                
                # Only keep positions with classified features
                valid_positions = total_classified_counts > 0
                if np.any(valid_positions):
                    position_ratios = alignment_counts[valid_positions] / total_classified_counts[valid_positions]
                    position_alignment_ratios.extend(position_ratios.tolist())
                    position_total_steered.extend(total_classified_counts[valid_positions].tolist())
                
                # Track unique steered features (from real tokens only)
                steered_features_in_sample = np.where(np.any(sample_steered_mask, axis=0))[0]
                all_steered_features.extend(steered_features_in_sample.tolist())
                
                # Track classified steered features
                steered_alignment_features = steered_features_in_sample[alignment_mask[steered_features_in_sample]]
                steered_not_alignment_features = steered_features_in_sample[not_alignment_mask[steered_features_in_sample]]
                
                alignment_related_steered.extend(steered_alignment_features.tolist())
                not_alignment_related_steered.extend(steered_not_alignment_features.tolist())
    
    # Calculate statistics using position-level analysis
    unique_steered = list(set(all_steered_features))
    unique_alignment_steered = list(set(alignment_related_steered))
    unique_not_alignment_steered = list(set(not_alignment_related_steered))
    
    # Get baseline statistics from all classified features
    all_classifications = list(feature_classifications.values())
    total_alignment = sum(1 for c in all_classifications if c == "alignment-related")
    total_not_alignment = sum(1 for c in all_classifications if c == "not-alignment-related")
    baseline_alignment_rate = total_alignment / len(all_classifications) if all_classifications else 0
    
    # Calculate steering statistics from position-level analysis
    if len(position_alignment_ratios) > 0:
        # Average alignment rate across all positions where steering occurs
        steering_alignment_rate = float(np.mean(position_alignment_ratios))
        total_positions_analyzed = len(position_alignment_ratios)
        mean_steered_per_position = float(np.mean(position_total_steered))
    else:
        steering_alignment_rate = 0.0
        total_positions_analyzed = 0
        mean_steered_per_position = 0.0
    
    # Calculate L0 norm statistics
    l0_norms_array = np.array(l0_norms)
    l0_mean = float(np.mean(l0_norms_array)) if len(l0_norms_array) > 0 else 0.0
    l0_variance = float(np.var(l0_norms_array)) if len(l0_norms_array) > 0 else 0.0
    l0_std = float(np.std(l0_norms_array)) if len(l0_norms_array) > 0 else 0.0
    l0_stderr = float(l0_std / np.sqrt(len(l0_norms_array))) if len(l0_norms_array) > 0 else 0.0
    
    results = {
        "configuration": {
            "append_response": CONFIG["analysis"]["append_response"],
            "description": "prompt-only" if CONFIG["analysis"]["append_response"] is None else f"prompt + {CONFIG['analysis']['append_response']} response"
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
        "l0_norm_variance": l0_variance,
        "l0_norm_std": l0_std,
        "l0_norm_stderr": l0_stderr,
        "l0_norms_per_sample": [float(x) for x in l0_norms],
        "position_alignment_ratios": [float(x) for x in position_alignment_ratios],
        "steered_feature_indices": [int(x) for x in unique_steered],
        "alignment_steered_indices": [int(x) for x in unique_alignment_steered],
        "not_alignment_steered_indices": [int(x) for x in unique_not_alignment_steered],
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze alignment-related steering in SAE adapter")
    parser.add_argument("--adapter_path", type=str, default=DEFAULT_ADAPTER_PATH,
                        help=f"Path to the trained SAE adapter (default: {DEFAULT_ADAPTER_PATH})")
    parser.add_argument("--classification_file", type=str, default=DEFAULT_CLASSIFICATION_FILE,
                        help=f"Path to feature classification JSON file (default: {DEFAULT_CLASSIFICATION_FILE})")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Limit dataset size (for testing)")
    parser.add_argument("--num_analysis_samples", type=int, default=None,
                        help="Number of samples to use for steering analysis (default: use all)")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for processing samples (default: 2)")
    parser.add_argument("--output_file", type=str, default="outputs/alignment_steering_analysis.json",
                        help="Path to save analysis results")
    parser.add_argument("--append_response", type=str, choices=[None, "chosen", "rejected"], default=None,
                        help="Append response to prompt: None (prompt only, default), 'chosen' (prompt+chosen), 'rejected' (prompt+rejected)")
    
    args = parser.parse_args()
    
    # Update configuration with command-line argument
    CONFIG["analysis"]["append_response"] = args.append_response
    
    # Create output directory
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    
    print("Loading model and tokenizer...")
    base_model, tokenizer, device = load_model_and_tokenizer()
    
    print("Loading SAE adapter...")
    sae = load_sae_adapter(args.adapter_path, device)
    
    print("Creating hooked model...")
    model = HookedModel(base_model, sae)
    
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
    
    # Print results
    print("\n" + "="*60)
    print("ALIGNMENT STEERING ANALYSIS RESULTS")
    print("="*60)
    print(f"Configuration: append_response = {CONFIG['analysis']['append_response']}")
    if CONFIG['analysis']['append_response'] is None:
        print("  (Using prompt-only text)")
    elif CONFIG['analysis']['append_response'] == "chosen":
        print("  (Using prompt + chosen response, matching training)")
    elif CONFIG['analysis']['append_response'] == "rejected":
        print("  (Using prompt + rejected response)")
    print()
    print(f"Total unique features steered: {results['total_steered_features']}")
    print(f"Unique alignment-related features steered: {results['alignment_related_steered']}")
    print(f"Unique not alignment-related features steered: {results['not_alignment_related_steered']}")
    print(f"Total positions analyzed: {results['total_positions_analyzed']}")
    print(f"Mean features steered per position: {results['mean_steered_per_position']:.2f}")
    print()
    print(f"Baseline alignment rate (all features): {results['baseline_alignment_rate']:.3f}")
    print(f"Position-averaged steering alignment rate: {results['steering_alignment_rate']:.3f}")
    print(f"Improvement over baseline: {results['improvement_over_baseline']:.3f}")
    print()
    print("L0 Norm Statistics (sparsity):")
    print(f"L0 norm mean: {results['l0_norm_mean']:.2f}")
    print(f"L0 norm std: {results['l0_norm_std']:.2f}")
    print(f"L0 norm stderr: {results['l0_norm_stderr']:.2f}")
    print(f"L0 norm variance: {results['l0_norm_variance']:.2f}")
    print()
    
    if results['improvement_over_baseline'] > 0:
        print("✅ POSITIVE RESULT: Adapter steers more alignment-related features than random!")
    else:
        print("❌ NEGATIVE RESULT: Adapter does not preferentially steer alignment-related features")
    
    print("="*60)
    
    # Save detailed results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
