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
    }
}

# Default adapter path
DEFAULT_ADAPTER_PATH = "models/Gemma2-2B-clean/mild-glade-10"

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
    
    # Apply chat template - use simpo_generation to get prompts for generation
    eval_dataset = eval_dataset.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "simpo_generation",  # This extracts prompts and adds generation prompt
            "chat_template": CONFIG["dataset"]["chat_template"],
        },
        num_proc=CONFIG["dataset"]["dataset_num_proc"],
        remove_columns=column_names,
        desc="Formatting prompts with chat template",
        disable_tqdm=False,
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
                    try:
                        feature_idx = int(m.group(1))
                    except:
                        pass
        
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
            
            # Prepare batch of prompts
            batch_prompts = []
            for idx in batch_indices:
                sample = eval_dataset[int(idx)]
                batch_prompts.append(sample["text_prompt"])
            
            # Tokenize the batch
            batch_tokens = model.base_model.tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1800,
                padding=True
            )
            
            # Use run_with_cache to capture steering vectors
            _, llm_cache = model.run_with_cache(batch_tokens["input_ids"], prepend_bos=True)
            
            # Get the steering vector from the hook_sae_adapter key
            if "hook_sae_adapter" in llm_cache:
                steering_vector = llm_cache["hook_sae_adapter"]
                
                # Convert to numpy if it's a tensor
                if isinstance(steering_vector, torch.Tensor):
                    steering_vector = steering_vector.cpu().numpy()
                
                # Handle batch and sequence dimensions
                # steering_vector shape: [batch_size, seq_len, num_features]
                # Take mean across sequence length for each sample in batch
                if steering_vector.ndim == 3:  # [batch, seq, features]
                    steering_vector = np.mean(steering_vector, axis=1)  # [batch, features]
                elif steering_vector.ndim > 3:
                    # Handle any additional dimensions
                    axes_to_mean = tuple(range(1, steering_vector.ndim - 1))
                    steering_vector = np.mean(steering_vector, axis=axes_to_mean)
                
                # Process each sample in the batch
                for batch_idx in range(steering_vector.shape[0]):
                    sample_steering = steering_vector[batch_idx]  # [features]
                    
                    # Find features that are being steered (non-zero values)
                    steered_indices = np.where(np.abs(sample_steering) > 1e-6)[0]
                    
                    all_steered_features.extend(steered_indices.tolist())
                    
                    # Classify steered features
                    for feature_idx in steered_indices:
                        if feature_idx in feature_classifications:
                            classification = feature_classifications[feature_idx]
                            if classification == "alignment-related":
                                alignment_related_steered.append(feature_idx)
                            elif classification == "not-alignment-related":
                                not_alignment_related_steered.append(feature_idx)
    
    # Calculate statistics
    unique_steered = list(set(all_steered_features))
    unique_alignment_steered = list(set(alignment_related_steered))
    unique_not_alignment_steered = list(set(not_alignment_related_steered))
    
    # Get baseline statistics from all classified features
    all_classifications = list(feature_classifications.values())
    total_alignment = sum(1 for c in all_classifications if c == "alignment-related")
    total_not_alignment = sum(1 for c in all_classifications if c == "not-alignment-related")
    baseline_alignment_rate = total_alignment / len(all_classifications) if all_classifications else 0
    
    # Calculate steering statistics
    total_classified_steered = len(unique_alignment_steered) + len(unique_not_alignment_steered)
    steering_alignment_rate = len(unique_alignment_steered) / total_classified_steered if total_classified_steered > 0 else 0
    
    results = {
        "total_steered_features": len(unique_steered),
        "alignment_related_steered": len(unique_alignment_steered),
        "not_alignment_related_steered": len(unique_not_alignment_steered),
        "total_classified_steered": total_classified_steered,
        "baseline_alignment_rate": baseline_alignment_rate,
        "steering_alignment_rate": steering_alignment_rate,
        "improvement_over_baseline": steering_alignment_rate - baseline_alignment_rate,
        "steered_feature_indices": unique_steered,
        "alignment_steered_indices": unique_alignment_steered,
        "not_alignment_steered_indices": unique_not_alignment_steered,
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
    
    args = parser.parse_args()
    
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
    print(f"Total features steered: {results['total_steered_features']}")
    print(f"Alignment-related features steered: {results['alignment_related_steered']}")
    print(f"Not alignment-related features steered: {results['not_alignment_related_steered']}")
    print(f"Total classified features steered: {results['total_classified_steered']}")
    print()
    print(f"Baseline alignment rate (all features): {results['baseline_alignment_rate']:.3f}")
    print(f"Steering alignment rate: {results['steering_alignment_rate']:.3f}")
    print(f"Improvement over baseline: {results['improvement_over_baseline']:.3f}")
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
