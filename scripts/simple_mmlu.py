#!/usr/bin/env python3
"""
Simple MMLU evaluation script for models trained with the training_demo.ipynb notebook.

Usage:
    python scripts/simple_mmlu.py --model_path ../logs/test_dpo
    python scripts/simple_mmlu.py --model_path ../logs/test_simpo
    python scripts/simple_mmlu.py  # Uses base model without any trained adapter
"""

import os
import sys
import torch
import argparse
import lm_eval
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fsrl import SAEAdapter, HookedModel
from transformer_lens import HookedTransformer

from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

def load_model(model_path=None, model_name="google/gemma-2-2b-it"):
    """Load either a trained model from path or base model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if model_path and os.path.exists(model_path):
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer
    else:
        model = HookedTransformer.from_pretrained_no_processing(model_name, dtype=torch.bfloat16)
        tokenizer = model.tokenizer
        
        # to recreate SAE
        try:
            adapter_kwargs = {
                "use_lora_adapter": True,
                "lora_rank": 32,
                "lora_alpha": 32, # as jeremias mentioned the heuristic is to set it to the rank
                "fusion_mode": "additive",
            }
            if "gemma" in model_name.lower():
                release = "gemma-scope-2b-pt-res"
                sae_id = "layer_12/width_65k/average_l0_21"
            elif "gpt2" in model_name.lower():
                release = "gpt2-small-res-jb"
                sae_id = "blocks.7.hook_resid_pre"
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            
            sae, cfg_dict, sparsity = SAEAdapter.from_pretrained(release, sae_id, device=device, **adapter_kwargs)
            model = HookedModel(model, sae)
        except Exception as e:
            print(f"Could not load SAE adapter: {e}")
        
        return model, tokenizer


def run_mmlu(model, tokenizer, batch_size=2, limit=None):
    """Run MMLU evaluation."""
    
    if hasattr(model, 'model') and hasattr(model, 'config'):
        hf_model = model
        print(f"Debug: Using HookedModel directly (has config)")
    
    # For HookedTransformer without HookedModel wrapper, we need to access the actual PyTorch model
    elif hasattr(model, 'cfg') and hasattr(model, 'model'):
        # This is a HookedTransformer, get the actual PyTorch model
        hf_model = model.model
    
    # Create HFLM wrapper for lm_eval
    eval_model = HFLM(pretrained=hf_model, tokenizer=tokenizer, batch_size=batch_size)
    
    # Run MMLU evaluation
    task_manager = lm_eval.tasks.TaskManager()
    results = lm_eval.simple_evaluate(
        model=eval_model,
        tasks=["mmlu"],
        task_manager=task_manager,
        limit=limit if limit is not None else 0.01,  # Use limit for faster testing, set to None for full evaluation
        apply_chat_template=True,
    )
    
    return results["results"]["mmlu"]


def main():
    parser = argparse.ArgumentParser(description="Simple MMLU evaluation")
    parser.add_argument("--model_path", type=str, default=None, 
                       help="Path to trained model directory (e.g., ../logs/test_dpo)")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b-it",
                       help="Base model name if not loading from path")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for evaluation")
    parser.add_argument("--limit", type=float, default=0.01,
                       help="Fraction of MMLU to evaluate (0.01 = 1%, None = full)")
    
    args = parser.parse_args()
    
    # Convert limit
    if args.limit == 0 or args.limit >= 1.0:
        limit = None  # Full evaluation
    else:
        limit = args.limit
    
    print("=" * 50)
    print("Simple MMLU Evaluation")
    print("=" * 50)
    
    # Load model
    model, tokenizer = load_model(args.model_path, args.model_name)
    
    # Run evaluation
    results = run_mmlu(model, tokenizer, args.batch_size, limit)
    
    # Print results
    print("\n" + "=" * 30)
    print("MMLU Results")
    print("=" * 30)
    
    accuracy = results.get('acc,none', results.get('acc_norm', 'N/A'))
    print(f"Accuracy: {accuracy}")
    
    if limit is not None:
        print(f"(Evaluated on {limit*100:.1f}% of MMLU)")
    
    # Print other available metrics
    print("\nAll metrics:")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main() 