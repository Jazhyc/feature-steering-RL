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
        "ignore_attention_mask": True,  # If True, treat all positions as valid (ignore padding)
    }
}

# Default adapter paths - different for alignment vs formatting
DEFAULT_ALIGNMENT_ADAPTER_PATH = "models/Gemma2-2B-new-arch/pious-wildflower-11/adapter"
DEFAULT_FORMATTING_ADAPTER_PATH = "models/Gemma2-2B-new-arch/pious-wildflower-11/adapter"  # Update this when you have a formatting-trained model (OK?)

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


def analyze_features_from_cache(
    activations: np.ndarray,
    attention_mask: np.ndarray,
    feature_classifications: Dict[int, str],
    classification_mode: str,
    ignore_attention_mask: bool = True
) -> Tuple[Dict[str, any], np.ndarray]:
    """
    Analyze feature activations and their classification properties.
    
    Args:
        activations: Feature activations of shape [batch, seq_len, num_features]
        attention_mask: Attention mask of shape [batch, seq_len]
        feature_classifications: Dict mapping feature indices to labels ('related'/'not-related')
        classification_mode: The type of classification being analyzed
        ignore_attention_mask: Whether to ignore the attention mask
    
    Returns:
        Tuple of (analysis_results, feature_usage_counter)
    """
    all_steered_features = []
    related_steered = []
    not_related_steered = []
    
    # Track position-level statistics for proper averaging
    position_related_ratios = []
    position_total_steered = []
    
    # Initialize feature usage counter
    num_features = activations.shape[2]
    feature_usage_counter = np.zeros(num_features, dtype=int)
    
    if ignore_attention_mask:
        effective_attention_mask = np.ones_like(attention_mask)
    else:
        effective_attention_mask = attention_mask
    
    feature_indices = np.arange(num_features)
    # Use new standardized labels: 'related' and 'not-related'
    related_mask = np.array([feature_classifications.get(idx) == "related" for idx in feature_indices])
    not_related_mask = np.array([feature_classifications.get(idx) == "not-related" for idx in feature_indices])

    for batch_idx in range(activations.shape[0]):
        sample_len = int(effective_attention_mask[batch_idx].sum())
        if sample_len == 0: 
            continue

        sample_activations = activations[batch_idx, :sample_len, :]
        sample_active_mask = (np.abs(sample_activations) > 1e-6)
        
        if not np.any(sample_active_mask): 
            continue

        related_counts = np.sum(sample_active_mask & related_mask[None, :], axis=1)
        not_related_counts = np.sum(sample_active_mask & not_related_mask[None, :], axis=1)
        total_classified_counts = related_counts + not_related_counts
        
        valid_positions = total_classified_counts > 0
        if np.any(valid_positions):
            position_ratios = related_counts[valid_positions] / total_classified_counts[valid_positions]
            position_related_ratios.extend(position_ratios.tolist())
            position_total_steered.extend(total_classified_counts[valid_positions].tolist())
        
        active_features_in_sample = np.where(np.any(sample_active_mask, axis=0))[0]
        all_steered_features.extend(active_features_in_sample.tolist())
        
        # Update feature usage counter for this sample
        # Count how many times each feature is used (across all positions in this sample)
        feature_usage_in_sample = np.sum(sample_active_mask, axis=0)  # Shape: [num_features]
        feature_usage_counter += feature_usage_in_sample.astype(int)
        
        active_related_features = active_features_in_sample[related_mask[active_features_in_sample]]
        active_not_related_features = active_features_in_sample[not_related_mask[active_features_in_sample]]
        
        related_steered.extend(active_related_features.tolist())
        not_related_steered.extend(active_not_related_features.tolist())

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
    
    results = {
        "total_steered_features": len(unique_steered),
        "related_steered": len(unique_related_steered),
        "not_related_steered": len(unique_not_related_steered),
        "total_positions_analyzed": total_positions_analyzed,
        "mean_steered_per_position": mean_steered_per_position,
        "baseline_related_rate": float(baseline_related_rate),
        "steering_related_rate": float(steering_related_rate),
        "improvement_over_baseline": float(steering_related_rate - baseline_related_rate),
        # Raw data for statistical testing
        "position_level_data": {
            "position_related_ratios": position_related_ratios,
            "position_total_steered": position_total_steered,
            "num_positions": len(position_related_ratios)
        }
    }
    
    return results, feature_usage_counter


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
    Analyzes both SAE adapter and regular SAE activations.
    
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
    
    l0_norms_adapter = []
    l0_norms_sae = []
    
    # Initialize feature usage counters
    adapter_usage_counter = None
    sae_usage_counter = None
    
    # Per-feature activation statistics using Welford's online algorithm
    adapter_feature_count = None
    adapter_feature_mean = None
    adapter_feature_m2 = None
    
    sae_feature_count = None
    sae_feature_mean = None
    sae_feature_m2 = None
    
    # Track position-level statistics for proper averaging
    adapter_position_related_ratios = []
    adapter_position_total_steered = []
    sae_position_related_ratios = []
    sae_position_total_steered = []
    
    # Track steered features
    adapter_all_steered_features = []
    adapter_related_steered = []
    adapter_not_related_steered = []
    
    sae_all_active_features = []
    sae_related_active = []
    sae_not_related_active = []
    
    # Pre-compute classification masks ONCE outside the batch loop for maximum efficiency
    if not feature_classifications:
        raise ValueError("feature_classifications must be provided and non-empty")
    
    num_features = max(feature_classifications.keys()) + 1
    related_mask_cpu = np.array([feature_classifications.get(idx, "") == "related" for idx in range(num_features)])
    not_related_mask_cpu = np.array([feature_classifications.get(idx, "") == "not-related" for idx in range(num_features)])
    
    model.eval()
    with torch.no_grad():
        num_batches = (len(sample_indices) + batch_size - 1) // batch_size
        batch_iterator = tqdm(range(0, len(sample_indices), batch_size), desc="Processing batches", total=num_batches)
        
        # Convert classification masks to GPU tensors once
        related_mask_gpu = None
        not_related_mask_gpu = None
        
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
            
            # Get both adapter and SAE activations - keep on GPU
            adapter_activations = llm_cache["blocks.12.hook_resid_post.hook_sae_adapter"]
            sae_activations = llm_cache["blocks.12.hook_resid_post.hook_sae_acts_post"]
            
            # Keep as tensors for GPU computation, convert to float32 for consistency
            if isinstance(adapter_activations, torch.Tensor):
                adapter_activations = adapter_activations.float()
            if isinstance(sae_activations, torch.Tensor):
                sae_activations = sae_activations.float()
            
            # Initialize feature usage counters on first batch
            if adapter_usage_counter is None:
                num_features = adapter_activations.shape[2]
                adapter_usage_counter = np.zeros(num_features, dtype=int)
                sae_usage_counter = np.zeros(num_features, dtype=int)
                
                # Initialize per-feature activation statistics on GPU
                device = adapter_activations.device
                adapter_feature_count = torch.zeros(num_features, dtype=torch.long, device=device)
                adapter_feature_mean = torch.zeros(num_features, dtype=torch.float32, device=device)
                adapter_feature_m2 = torch.zeros(num_features, dtype=torch.float32, device=device)
                
                sae_feature_count = torch.zeros(num_features, dtype=torch.long, device=device)
                sae_feature_mean = torch.zeros(num_features, dtype=torch.float32, device=device)
                sae_feature_m2 = torch.zeros(num_features, dtype=torch.float32, device=device)
                
                # Initialize GPU classification masks on first batch
                actual_num_features = adapter_activations.shape[2]
                related_mask_gpu = torch.tensor(related_mask_cpu[:actual_num_features], device=device, dtype=torch.bool)
                not_related_mask_gpu = torch.tensor(not_related_mask_cpu[:actual_num_features], device=device, dtype=torch.bool)
                
                print(f"Tracking usage for {num_features} SAE features...")
            
            # --- FIX 3: Replicate the trainer's naive L0 norm calculation ---
            # The trainer's metric is a simple mean over the whole tensor, including padding.
            # Use PyTorch operations since tensors are still on GPU
            l0_per_position_batch_adapter = torch.count_nonzero(torch.abs(adapter_activations) > 1e-6, dim=-1)
            batch_l0_mean_adapter = float(torch.mean(l0_per_position_batch_adapter.float()))
            l0_norms_adapter.append(batch_l0_mean_adapter)
            
            l0_per_position_batch_sae = torch.count_nonzero(torch.abs(sae_activations) > 1e-6, dim=-1)
            batch_l0_mean_sae = float(torch.mean(l0_per_position_batch_sae.float()))
            l0_norms_sae.append(batch_l0_mean_sae)

            # Update per-feature raw value statistics using Welford's online algorithm
            # Fully vectorized GPU implementation - no loops, maximum speed
            
            # For adapter activations - batch vectorized Welford update
            adapter_flat = adapter_activations.view(-1, adapter_activations.shape[2])  # (batch*seq, features)
            n_new = adapter_flat.shape[0]
            
            # Current statistics
            n_old = adapter_feature_count.clone().float()
            n_total = n_old + n_new
            
            # Batch statistics
            batch_mean = torch.mean(adapter_flat, dim=0)
            batch_var = torch.var(adapter_flat, dim=0, unbiased=False) * n_new  # Convert back to sum of squares
            
            # Combined mean
            delta = batch_mean - adapter_feature_mean
            new_mean = adapter_feature_mean + delta * n_new / n_total
            
            # Combined M2 (sum of squared deviations)
            new_m2 = adapter_feature_m2 + batch_var + delta.pow(2) * n_old * n_new / n_total
            
            # Update statistics
            adapter_feature_count += n_new
            adapter_feature_mean = new_mean
            adapter_feature_m2 = new_m2
            
            # For SAE activations - batch vectorized Welford update
            sae_flat = sae_activations.view(-1, sae_activations.shape[2])  # (batch*seq, features)
            
            # Current statistics
            n_old_sae = sae_feature_count.clone().float()
            n_total_sae = n_old_sae + n_new
            
            # Batch statistics
            batch_mean_sae = torch.mean(sae_flat, dim=0)
            batch_var_sae = torch.var(sae_flat, dim=0, unbiased=False) * n_new
            
            # Combined mean
            delta_sae = batch_mean_sae - sae_feature_mean
            new_mean_sae = sae_feature_mean + delta_sae * n_new / n_total_sae
            
            # Combined M2 (sum of squared deviations)
            new_m2_sae = sae_feature_m2 + batch_var_sae + delta_sae.pow(2) * n_old_sae * n_new / n_total_sae
            
            # Update statistics
            sae_feature_count += n_new
            sae_feature_mean = new_mean_sae
            sae_feature_m2 = new_m2_sae

            # VECTORIZED GPU PROCESSING - Process all samples at once for massive speedup
            batch_size_dim, seq_len, num_features = adapter_activations.shape
            
            # Create effective attention mask on GPU
            attention_mask_gpu = batch_tokens["attention_mask"]
            if CONFIG["analysis"]["ignore_attention_mask"]:
                effective_mask = torch.ones((batch_size_dim, seq_len), device=device, dtype=torch.bool)
            else:
                effective_mask = attention_mask_gpu.bool()
            
            # Vectorized adapter processing on GPU
            adapter_threshold_mask = (torch.abs(adapter_activations) > 1e-6)  # Shape: [batch, seq, features]
            adapter_active_mask = adapter_threshold_mask & effective_mask[:, :, None]  # Apply attention mask
            
            # Count feature usage across all positions and samples (GPU)
            adapter_usage_batch = torch.sum(adapter_active_mask, dim=(0, 1))  # Sum over batch and sequence
            adapter_usage_counter += adapter_usage_batch.cpu().numpy().astype(int)
            
            # Find steered features across all samples (GPU)
            adapter_any_active = torch.any(adapter_active_mask, dim=(0, 1))  # Features active anywhere
            adapter_steered_features = torch.where(adapter_any_active)[0]
            adapter_all_steered_features.extend(adapter_steered_features.cpu().tolist())
            
            # Classify steered features (GPU operations)
            adapter_steered_related = adapter_steered_features[related_mask_gpu[adapter_steered_features]]
            adapter_steered_not_related = adapter_steered_features[not_related_mask_gpu[adapter_steered_features]]
            adapter_related_steered.extend(adapter_steered_related.cpu().tolist())
            adapter_not_related_steered.extend(adapter_steered_not_related.cpu().tolist())
            
            # Position-wise analysis (only convert to CPU for final aggregation)
            for batch_idx in range(batch_size_dim):
                sample_len = int(effective_mask[batch_idx].sum().item())
                if sample_len == 0:
                    continue
                    
                sample_mask = adapter_active_mask[batch_idx, :sample_len, :]
                if torch.any(sample_mask):
                    related_counts = torch.sum(sample_mask & related_mask_gpu[None, :], dim=1)
                    not_related_counts = torch.sum(sample_mask & not_related_mask_gpu[None, :], dim=1)
                    total_counts = related_counts + not_related_counts
                    valid_pos = total_counts > 0
                    if torch.any(valid_pos):
                        ratios = related_counts[valid_pos].float() / total_counts[valid_pos].float()
                        adapter_position_related_ratios.extend(ratios.cpu().tolist())
                        adapter_position_total_steered.extend(total_counts[valid_pos].cpu().tolist())

            # VECTORIZED SAE PROCESSING ON GPU
            sae_threshold_mask = (torch.abs(sae_activations) > 1e-6)
            sae_active_mask = sae_threshold_mask & effective_mask[:, :, None]
            
            # Count feature usage across all positions and samples (GPU)
            sae_usage_batch = torch.sum(sae_active_mask, dim=(0, 1))
            sae_usage_counter += sae_usage_batch.cpu().numpy().astype(int)
            
            # Find active features across all samples (GPU)
            sae_any_active = torch.any(sae_active_mask, dim=(0, 1))
            sae_active_features = torch.where(sae_any_active)[0]
            sae_all_active_features.extend(sae_active_features.cpu().tolist())
            
            # Classify active features (GPU operations)
            sae_active_related = sae_active_features[related_mask_gpu[sae_active_features]]
            sae_active_not_related = sae_active_features[not_related_mask_gpu[sae_active_features]]
            sae_related_active.extend(sae_active_related.cpu().tolist())
            sae_not_related_active.extend(sae_active_not_related.cpu().tolist())
            
            # Position-wise analysis for SAE (GPU operations)
            for batch_idx in range(batch_size_dim):
                sample_len = int(effective_mask[batch_idx].sum().item())
                if sample_len == 0:
                    continue
                    
                sample_mask = sae_active_mask[batch_idx, :sample_len, :]
                if torch.any(sample_mask):
                    related_counts = torch.sum(sample_mask & related_mask_gpu[None, :], dim=1)
                    not_related_counts = torch.sum(sample_mask & not_related_mask_gpu[None, :], dim=1)
                    total_counts = related_counts + not_related_counts
                    valid_pos = total_counts > 0
                    if torch.any(valid_pos):
                        ratios = related_counts[valid_pos].float() / total_counts[valid_pos].float()
                        sae_position_related_ratios.extend(ratios.cpu().tolist())
                        sae_position_total_steered.extend(total_counts[valid_pos].cpu().tolist())
    
    # Process results for adapter
    unique_adapter_steered = list(set(adapter_all_steered_features))
    unique_adapter_related_steered = list(set(adapter_related_steered))
    unique_adapter_not_related_steered = list(set(adapter_not_related_steered))
    
    all_classifications = list(feature_classifications.values())
    total_related = sum(1 for c in all_classifications if c == "related")
    baseline_related_rate = total_related / len(all_classifications) if all_classifications else 0
    
    if len(adapter_position_related_ratios) > 0:
        adapter_steering_related_rate = float(np.mean(adapter_position_related_ratios))
        adapter_total_positions_analyzed = len(adapter_position_related_ratios)
        adapter_mean_steered_per_position = float(np.mean(adapter_position_total_steered))
    else:
        adapter_steering_related_rate = 0.0
        adapter_total_positions_analyzed = 0
        adapter_mean_steered_per_position = 0.0
    
    # Process results for SAE
    unique_sae_active = list(set(sae_all_active_features))
    unique_sae_related_active = list(set(sae_related_active))
    unique_sae_not_related_active = list(set(sae_not_related_active))
    
    if len(sae_position_related_ratios) > 0:
        sae_related_rate = float(np.mean(sae_position_related_ratios))
        sae_total_positions_analyzed = len(sae_position_related_ratios)
        sae_mean_active_per_position = float(np.mean(sae_position_total_steered))
    else:
        sae_related_rate = 0.0
        sae_total_positions_analyzed = 0
        sae_mean_active_per_position = 0.0
    
    # Calculate L0 norms
    l0_norms_adapter_array = np.array(l0_norms_adapter)
    l0_mean_adapter = float(np.mean(l0_norms_adapter_array)) if len(l0_norms_adapter_array) > 0 else 0.0
    l0_std_adapter = float(np.std(l0_norms_adapter_array)) if len(l0_norms_adapter_array) > 0 else 0.0
    l0_stderr_adapter = float(l0_std_adapter / np.sqrt(len(l0_norms_adapter_array))) if len(l0_norms_adapter_array) > 0 else 0.0
    
    l0_norms_sae_array = np.array(l0_norms_sae)
    l0_mean_sae = float(np.mean(l0_norms_sae_array)) if len(l0_norms_sae_array) > 0 else 0.0
    l0_std_sae = float(np.std(l0_norms_sae_array)) if len(l0_norms_sae_array) > 0 else 0.0
    l0_stderr_sae = float(l0_std_sae / np.sqrt(len(l0_norms_sae_array))) if len(l0_norms_sae_array) > 0 else 0.0
    
    # Calculate raw value statistics from per-feature statistics
    # Convert GPU tensors to CPU/NumPy for statistics calculations
    if adapter_feature_count is not None:
        adapter_feature_count_cpu = adapter_feature_count.cpu().numpy()
        adapter_feature_mean_cpu = adapter_feature_mean.cpu().numpy()
        adapter_feature_m2_cpu = adapter_feature_m2.cpu().numpy()
    else:
        adapter_feature_count_cpu = None
        adapter_feature_mean_cpu = None
        adapter_feature_m2_cpu = None
    
    if sae_feature_count is not None:
        sae_feature_count_cpu = sae_feature_count.cpu().numpy()
        sae_feature_mean_cpu = sae_feature_mean.cpu().numpy()
        sae_feature_m2_cpu = sae_feature_m2.cpu().numpy()
    else:
        sae_feature_count_cpu = None
        sae_feature_mean_cpu = None
        sae_feature_m2_cpu = None
    
    # Compute global means as weighted average of per-feature means
    if adapter_feature_count_cpu is not None and np.sum(adapter_feature_count_cpu) > 0:
        # Global mean is weighted average of per-feature means
        total_adapter_count = np.sum(adapter_feature_count_cpu)
        raw_mean_adapter = float(np.sum(adapter_feature_mean_cpu * adapter_feature_count_cpu) / total_adapter_count)
        
        # For global variance, we need to combine per-feature variances
        # Using the formula for combining variances from multiple groups
        feature_variances = np.where(adapter_feature_count_cpu > 0, adapter_feature_m2_cpu / adapter_feature_count_cpu, 0.0)
        mean_of_squares = np.sum(adapter_feature_count_cpu * (feature_variances + adapter_feature_mean_cpu**2)) / total_adapter_count
        raw_variance_adapter = mean_of_squares - raw_mean_adapter**2
        raw_std_adapter = float(np.sqrt(max(0, raw_variance_adapter)))
        raw_stderr_adapter = float(raw_std_adapter / np.sqrt(total_adapter_count))
    else:
        raw_mean_adapter = 0.0
        raw_std_adapter = 0.0
        raw_stderr_adapter = 0.0
    
    if sae_feature_count_cpu is not None and np.sum(sae_feature_count_cpu) > 0:
        # Global mean is weighted average of per-feature means
        total_sae_count = np.sum(sae_feature_count_cpu)
        raw_mean_sae = float(np.sum(sae_feature_mean_cpu * sae_feature_count_cpu) / total_sae_count)
        
        # For global variance, we need to combine per-feature variances
        feature_variances = np.where(sae_feature_count_cpu > 0, sae_feature_m2_cpu / sae_feature_count_cpu, 0.0)
        mean_of_squares = np.sum(sae_feature_count_cpu * (feature_variances + sae_feature_mean_cpu**2)) / total_sae_count
        raw_variance_sae = mean_of_squares - raw_mean_sae**2
        raw_std_sae = float(np.sqrt(max(0, raw_variance_sae)))
        raw_stderr_sae = float(raw_std_sae / np.sqrt(total_sae_count))
    else:
        raw_mean_sae = 0.0
        raw_std_sae = 0.0
        raw_stderr_sae = 0.0
    
    # Create adapter results
    adapter_results = {
        "total_steered_features": len(unique_adapter_steered),
        "related_steered": len(unique_adapter_related_steered),
        "not_related_steered": len(unique_adapter_not_related_steered),
        "total_positions_analyzed": adapter_total_positions_analyzed,
        "mean_steered_per_position": adapter_mean_steered_per_position,
        "baseline_related_rate": float(baseline_related_rate),
        "steering_related_rate": float(adapter_steering_related_rate),
        "improvement_over_baseline": float(adapter_steering_related_rate - baseline_related_rate),
        # Raw data for statistical testing
        "position_level_data": {
            "position_related_ratios": adapter_position_related_ratios,
            "position_total_steered": adapter_position_total_steered,
            "num_positions": len(adapter_position_related_ratios)
        }
    }
    
    # Create SAE results
    sae_results = {
        "total_steered_features": len(unique_sae_active),
        "related_steered": len(unique_sae_related_active),
        "not_related_steered": len(unique_sae_not_related_active),
        "total_positions_analyzed": sae_total_positions_analyzed,
        "mean_steered_per_position": sae_mean_active_per_position,
        "baseline_related_rate": float(baseline_related_rate),
        "steering_related_rate": float(sae_related_rate),
        "improvement_over_baseline": float(sae_related_rate - baseline_related_rate),
        # Raw data for statistical testing
        "position_level_data": {
            "position_related_ratios": sae_position_related_ratios,
            "position_total_steered": sae_position_total_steered,
            "num_positions": len(sae_position_related_ratios)
        }
    }
    
    # Combine results
    results = {
        "configuration": {
            "append_response": CONFIG["analysis"]["append_response"],
            "ignore_attention_mask": CONFIG["analysis"]["ignore_attention_mask"],
            "classification_mode": classification_mode,
        },
        "sae_adapter": {
            **adapter_results,
            "l0_norm_mean": l0_mean_adapter,
            "l0_norm_std": l0_std_adapter,
            "l0_norm_stderr": l0_stderr_adapter,
            "raw_value_mean": raw_mean_adapter,
            "raw_value_std": raw_std_adapter,
            "raw_value_stderr": raw_stderr_adapter,
            "feature_usage_counter": adapter_usage_counter.tolist() if adapter_usage_counter is not None else [],
            "feature_statistics": {
                "feature_count": adapter_feature_count_cpu.tolist() if adapter_feature_count_cpu is not None else [],
                "feature_mean": adapter_feature_mean_cpu.tolist() if adapter_feature_mean_cpu is not None else [],
                "feature_m2": adapter_feature_m2_cpu.tolist() if adapter_feature_m2_cpu is not None else []
            }
        },
        "sae_regular": {
            **sae_results,
            "l0_norm_mean": l0_mean_sae,
            "l0_norm_std": l0_std_sae,
            "l0_norm_stderr": l0_stderr_sae,
            "raw_value_mean": raw_mean_sae,
            "raw_value_std": raw_std_sae,
            "raw_value_stderr": raw_stderr_sae,
            "feature_usage_counter": sae_usage_counter.tolist() if sae_usage_counter is not None else [],
            "feature_statistics": {
                "feature_count": sae_feature_count_cpu.tolist() if sae_feature_count_cpu is not None else [],
                "feature_mean": sae_feature_mean_cpu.tolist() if sae_feature_mean_cpu is not None else [],
                "feature_m2": sae_feature_m2_cpu.tolist() if sae_feature_m2_cpu is not None else []
            }
        }
    }
    
    return results


def save_feature_usage_analysis(feature_usage_counter, feature_classifications, output_file, analysis_type="adapter", 
                               feature_count=None, feature_mean=None, feature_m2=None):
    """
    Save detailed feature usage statistics to a separate file.
    
    Args:
        feature_usage_counter: Array of usage counts per feature
        feature_classifications: Dict mapping feature indices to classifications
        output_file: Path to save the analysis
        analysis_type: Type of analysis ("adapter" or "sae") for labeling
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
        
        # Calculate per-feature activation statistics
        feature_data = {
            "feature_index": feature_idx,
            "usage_count": usage_count,
            "usage_percentage": usage_percentage,
            "classification": classification
        }
        
        # Add per-feature activation statistics if available
        if feature_count is not None and feature_mean is not None and feature_m2 is not None:
            if feature_count[feature_idx] > 0:
                mean_activation = float(feature_mean[feature_idx])
                variance = feature_m2[feature_idx] / feature_count[feature_idx]
                std_activation = float(np.sqrt(variance))
                firing_rate = float(usage_count / feature_count[feature_idx] * 100)  # Percentage of positions where feature fired
            else:
                mean_activation = 0.0
                std_activation = 0.0
                firing_rate = 0.0
            
            feature_data.update({
                "mean_activation": mean_activation,
                "std_activation": std_activation,
                "firing_rate_percent": firing_rate,
                "total_positions": int(feature_count[feature_idx])
            })
        
        feature_usage_data.append(feature_data)
    
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
        "analysis_type": analysis_type,
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
    print(f"  - Analysis type: {analysis_type}")
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
    main_results = {k: v for k, v in results.items() if not k.endswith("feature_usage_counter")}
    # Remove feature_usage_counter from nested structures too
    if "sae_adapter" in main_results:
        main_results["sae_adapter"] = {k: v for k, v in main_results["sae_adapter"].items() if k != "feature_usage_counter"}
    if "sae_regular" in main_results:
        main_results["sae_regular"] = {k: v for k, v in main_results["sae_regular"].items() if k != "feature_usage_counter"}
    
    with open(output_file, 'w') as f:
        json.dump(main_results, f, indent=2, sort_keys=True)
    
    # Save detailed feature usage for both adapter and SAE
    base_filename = str(output_file).replace(".json", "")
    
    if "sae_adapter" in results and "feature_usage_counter" in results["sae_adapter"]:
        adapter_usage_file = f"{base_filename}_adapter_feature_usage.json"
        adapter_stats = results["sae_adapter"]["feature_statistics"]
        save_feature_usage_analysis(
            results["sae_adapter"]["feature_usage_counter"], 
            feature_classifications, 
            adapter_usage_file,
            analysis_type="adapter",
            feature_count=np.array(adapter_stats["feature_count"]),
            feature_mean=np.array(adapter_stats["feature_mean"]),
            feature_m2=np.array(adapter_stats["feature_m2"])
        )
    
    if "sae_regular" in results and "feature_usage_counter" in results["sae_regular"]:
        sae_usage_file = f"{base_filename}_sae_feature_usage.json"
        sae_stats = results["sae_regular"]["feature_statistics"]
        save_feature_usage_analysis(
            results["sae_regular"]["feature_usage_counter"], 
            feature_classifications, 
            sae_usage_file,
            analysis_type="sae",
            feature_count=np.array(sae_stats["feature_count"]),
            feature_mean=np.array(sae_stats["feature_mean"]),
            feature_m2=np.array(sae_stats["feature_m2"])
        )
    
    # Print summary for both adapter and SAE
    print(f"\n{classification_mode.upper()} STEERING ANALYSIS RESULTS")
    print(f"Append response: {append_response}, Ignore mask: {ignore_attention_mask}")
    
    if "sae_adapter" in results:
        adapter_res = results["sae_adapter"]
        print(f"\n--- SAE ADAPTER RESULTS ---")
        print(f"Total unique features steered: {adapter_res['total_steered_features']}")
        print(f"Unique {classification_mode}-related features steered: {adapter_res['related_steered']}")
        print(f"Unique not-{classification_mode}-related features steered: {adapter_res['not_related_steered']}")
        print(f"Baseline {classification_mode}-related rate: {adapter_res['baseline_related_rate']:.3f}")
        print(f"Steering {classification_mode}-related rate: {adapter_res['steering_related_rate']:.3f}")
        print(f"Improvement over baseline: {adapter_res['improvement_over_baseline']:.3f}")
        print(f"L0 norm mean: {adapter_res['l0_norm_mean']:.2f}")
        
        if adapter_res['improvement_over_baseline'] > 0:
            print(f"✅ ADAPTER POSITIVE: Steers more {classification_mode}-related features than random")
        else:
            print(f"❌ ADAPTER NEGATIVE: Does not preferentially steer {classification_mode}-related features")
    
    if "sae_regular" in results:
        sae_res = results["sae_regular"]
        print(f"\n--- REGULAR SAE RESULTS ---")
        print(f"Total unique features active: {sae_res['total_steered_features']}")
        print(f"Unique {classification_mode}-related features active: {sae_res['related_steered']}")
        print(f"Unique not-{classification_mode}-related features active: {sae_res['not_related_steered']}")
        print(f"Baseline {classification_mode}-related rate: {sae_res['baseline_related_rate']:.3f}")
        print(f"SAE {classification_mode}-related rate: {sae_res['steering_related_rate']:.3f}")
        print(f"Difference from baseline: {sae_res['improvement_over_baseline']:.3f}")
        print(f"L0 norm mean: {sae_res['l0_norm_mean']:.2f}")
        
        if sae_res['improvement_over_baseline'] > 0:
            print(f"✅ SAE POSITIVE: Activates more {classification_mode}-related features than random")
        else:
            print(f"❌ SAE NEGATIVE: Does not preferentially activate {classification_mode}-related features")
    
    print(f"\nResults saved to: {output_file}")
    
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
        
        # Clean up GPU memory between classification modes
        del model, sae, base_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"\n{'='*100}")
    print("COMPREHENSIVE ANALYSIS COMPLETE")
    print("Individual experiment results saved in output directory")
    print(f"{'='*100}")
    
    return all_results
    
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
    parser.add_argument("--use_attention_mask", action="store_true", help="Use attention mask (treat padding positions as invalid, default is to ignore mask)")
    
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
        CONFIG["analysis"]["ignore_attention_mask"] = not args.use_attention_mask  # Invert because default is to ignore mask
        
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
        
        print("="*60)
        print(f"{args.classification_mode.upper()} STEERING ANALYSIS RESULTS")
        print("="*60)
        print(f"Configuration: append_response = {CONFIG['analysis']['append_response']}")
        print(f"Classification mode: {args.classification_mode}")
        
        if "sae_adapter" in results:
            adapter_res = results["sae_adapter"]
            print(f"\n--- SAE ADAPTER RESULTS ---")
            print(f"Total unique features steered: {adapter_res['total_steered_features']}")
            print(f"Unique {args.classification_mode}-related features steered: {adapter_res['related_steered']}")
            print(f"Unique not-{args.classification_mode}-related features steered: {adapter_res['not_related_steered']}")
            print(f"Total positions analyzed: {adapter_res['total_positions_analyzed']}")
            print(f"Mean features steered per position (masked): {adapter_res['mean_steered_per_position']:.2f}")
            print(f"Baseline {args.classification_mode}-related rate (all features): {adapter_res['baseline_related_rate']:.3f}")
            print(f"Position-averaged steering {args.classification_mode}-related rate: {adapter_res['steering_related_rate']:.3f}")
            print(f"Improvement over baseline: {adapter_res['improvement_over_baseline']:.3f}")
            print(f"L0 norm mean: {adapter_res['l0_norm_mean']:.2f}")
            print(f"L0 norm std: {adapter_res['l0_norm_std']:.2f}")
            print(f"L0 norm stderr: {adapter_res['l0_norm_stderr']:.2f}")
            
            if adapter_res['improvement_over_baseline'] > 0:
                print(f"✅ ADAPTER POSITIVE: Steers more {args.classification_mode}-related features than random!")
            else:
                print(f"❌ ADAPTER NEGATIVE: Does not preferentially steer {args.classification_mode}-related features")
        
        if "sae_regular" in results:
            sae_res = results["sae_regular"]
            print(f"\n--- REGULAR SAE RESULTS ---")
            print(f"Total unique features active: {sae_res['total_steered_features']}")
            print(f"Unique {args.classification_mode}-related features active: {sae_res['related_steered']}")
            print(f"Unique not-{args.classification_mode}-related features active: {sae_res['not_related_steered']}")
            print(f"Total positions analyzed: {sae_res['total_positions_analyzed']}")
            print(f"Mean features active per position (masked): {sae_res['mean_steered_per_position']:.2f}")
            print(f"Baseline {args.classification_mode}-related rate (all features): {sae_res['baseline_related_rate']:.3f}")
            print(f"Position-averaged SAE {args.classification_mode}-related rate: {sae_res['steering_related_rate']:.3f}")
            print(f"Difference from baseline: {sae_res['improvement_over_baseline']:.3f}")
            print(f"L0 norm mean: {sae_res['l0_norm_mean']:.2f}")
            print(f"L0 norm std: {sae_res['l0_norm_std']:.2f}")
            print(f"L0 norm stderr: {sae_res['l0_norm_stderr']:.2f}")
            
            if sae_res['improvement_over_baseline'] > 0:
                print(f"✅ SAE POSITIVE: Activates more {args.classification_mode}-related features than random!")
            else:
                print(f"❌ SAE NEGATIVE: Does not preferentially activate {args.classification_mode}-related features")
        
        print("="*60)
        
        # Save main analysis results (without detailed feature usage)
        main_results = {k: v for k, v in results.items() if not k.endswith("feature_usage_counter")}
        # Remove feature_usage_counter from nested structures too
        if "sae_adapter" in main_results:
            main_results["sae_adapter"] = {k: v for k, v in main_results["sae_adapter"].items() if k != "feature_usage_counter"}
        if "sae_regular" in main_results:
            main_results["sae_regular"] = {k: v for k, v in main_results["sae_regular"].items() if k != "feature_usage_counter"}
        
        with open(args.output_file, 'w') as f:
            json.dump(main_results, f, indent=2, sort_keys=True)
        
        print(f"\nMain results saved to: {args.output_file}")
        
        # Generate feature usage filenames and save detailed feature usage
        base_name = args.output_file.replace(".json", "")
        
        if "sae_adapter" in results and "feature_usage_counter" in results["sae_adapter"]:
            adapter_usage_file = f"{base_name}_adapter_feature_usage.json"
            adapter_stats = results["sae_adapter"]["feature_statistics"]
            save_feature_usage_analysis(
                results["sae_adapter"]["feature_usage_counter"], 
                feature_classifications, 
                adapter_usage_file,
                analysis_type="adapter",
                feature_count=np.array(adapter_stats["feature_count"]),
                feature_mean=np.array(adapter_stats["feature_mean"]),
                feature_m2=np.array(adapter_stats["feature_m2"])
            )
        else:
            print("No adapter feature usage data available to save.")
        
        if "sae_regular" in results and "feature_usage_counter" in results["sae_regular"]:
            sae_usage_file = f"{base_name}_sae_feature_usage.json"
            sae_stats = results["sae_regular"]["feature_statistics"]
            save_feature_usage_analysis(
                results["sae_regular"]["feature_usage_counter"], 
                feature_classifications, 
                sae_usage_file,
                analysis_type="sae",
                feature_count=np.array(sae_stats["feature_count"]),
                feature_mean=np.array(sae_stats["feature_mean"]),
                feature_m2=np.array(sae_stats["feature_m2"])
            )
        else:
            print("No SAE feature usage data available to save.")
        
        return results


if __name__ == "__main__":
    main()