import sys
import os
import json
import tempfile
from pathlib import Path
import argparse
import torch
import wandb
from dotenv import load_dotenv

from .. import SAEAdapter, HookedModel, BaseHookedModel
from ..utils.wandb_utils import WandBModelDownloader
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
import alpaca_eval



def load_model_and_adapter(run, with_adapter=True, full_ft=False, wandb_project="Gemma2-2B-new-arch"):
    """Load model and adapter using the same logic as evals.py"""
    root = Path(__file__).resolve().parent.parent.parent
    models_path = f"{root}/models/{wandb_project}" if not full_ft else f"{root}/models/full-gemma2_2B"
    
    load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    
    downloader = WandBModelDownloader(
        entity="feature-steering-RL",
        project=wandb_project if not full_ft else "full-gemma2_2B",
        verbose=True
    )
    
    print(f"##### Loading model for run: {run} #####")
    
    # Search in the correct project based on full_ft flag
    project_name = "feature-steering-rl/full-gemma2_2B" if full_ft else f"feature-steering-rl/{wandb_project}"
    run_objs = downloader.api.runs(project_name, filters={"display_name": run})
    
    if not run_objs:
        raise ValueError(f"No run found with name '{run}' in project '{project_name}'")
    
    downloader.download_model(run_objs[0], models_path)
    
    if full_ft:
        base_model_path = downloader.models_base_dir / "full-gemma2_2B" / run / "full_model"
        base_model = BaseHookedModel.from_pretrained(base_model_path, device="cuda", dtype=torch.bfloat16)
        hooked_model = base_model
    else:
        base_model = HookedTransformer.from_pretrained_no_processing(
            "google/gemma-2-2b-it", 
            device="cuda", 
            dtype=torch.bfloat16
        )
        
        adapter_path = downloader.models_base_dir / wandb_project / run / "adapter"
        print(f"Loading adapter from: {adapter_path}")
        sae_adapter = SAEAdapter.load_from_pretrained_adapter(adapter_path, device="cuda")
        hooked_model = HookedModel(base_model, sae_adapter)
        
        # Disable steering if not using adapter
        if not with_adapter:
            hooked_model.disable_steering()
            print(f"Evaluating model without steering")
    
    tokenizer = base_model.tokenizer
    return hooked_model, tokenizer

def generate_model_outputs(hooked_model, tokenizer, run_name, limit=None):
    """Generate outputs for AlpacaEval dataset using our model"""
    import datasets

    eval_data = datasets.load_dataset("tatsu-lab/alpaca_farm", "alpaca_farm_evaluation", trust_remote_code=True)["eval"]

    
    if limit:
        eval_data = eval_data.select(range(limit))
    
    model_outputs = []
    
    print(f"Generating outputs for {len(eval_data)} examples...")
    
    for i, example in enumerate(eval_data):
        if i % 50 == 0:
            print(f"Progress: {i}/{len(eval_data)}")
            
        instruction = example['instruction']
        
        if hasattr(tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": instruction}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"<bos><start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
        
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(hooked_model.device)
        
        with torch.no_grad():
            output_ids = hooked_model.generate(
                input_ids,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        response_ids = output_ids[0][len(input_ids[0]):]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        
        model_outputs.append({
            'instruction': instruction,
            'output': response,
            'generator': f"hooked_model_{run_name}"
        })
    
    return model_outputs

def run_alpaca_eval(runs, with_adapter=True, full_ft=False, annotator="alpaca_eval_gpt4", limit=None):
    """Run AlpacaEval evaluation on the specified runs"""

    load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(entity="feature-steering-RL", project="alpaca-eval")
    
    results = {}
    results_file = "alpaca_eval_results.json"
    
    for run in runs:
        print(f"=" * 50)
        print(f"Running AlpacaEval for: {run}")
        print(f"=" * 50)
        
        hooked_model, tokenizer = load_model_and_adapter(run, with_adapter, full_ft)
        
        model_outputs = generate_model_outputs(hooked_model, tokenizer, run, limit)
        
        eval_results = alpaca_eval.evaluate(
            model_outputs=model_outputs,
            annotators_config=annotator,
            name=f"{run}_{'with_adapter' if with_adapter else 'no_adapter'}",
            output_path=f"alpaca_eval_results_{run}",
            max_instances=limit,
            is_return_instead_of_print=True,
        )
        
        results[run] = eval_results
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        artifact = wandb.Artifact("alpaca_eval_results", type="results")
        artifact.add_file(results_file)
        wandb.log_artifact(artifact)

        wandb.log(results)
        
        del hooked_model, tokenizer
        torch.cuda.empty_cache()
    
    os.remove(results_file)

    return results 