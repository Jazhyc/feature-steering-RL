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
    
    downloader.download_model(run_objs[0], wandb_project)
    
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

def generate_model_outputs(hooked_model, tokenizer, run_name, limit=None, batch_size=16):
    """Generate outputs for AlpacaEval dataset using our model with batching"""
    import datasets

    eval_data = datasets.load_dataset("tatsu-lab/alpaca_farm", "alpaca_farm_evaluation", trust_remote_code=True)["eval"]

    
    if limit:
        eval_data = eval_data.select(range(limit))
    
    model_outputs = []
    output_filename = f"model_outputs_{run_name}.json"
    
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
                
                output_entry = {
                    'instruction': instruction,
                    'output': response,
                    'generator': f"hooked_model_{run_name}"
                }
                model_outputs.append(output_entry)
                batch_outputs.append(output_entry)
            
            batch_start = batch_end
            
            with open(output_filename, 'w') as f: # continuous saving w each batch
                json.dump(model_outputs, f, indent=2)
            
            if wandb.run is not None:
                artifact = wandb.Artifact(f"model_outputs_{run_name}", type="model_outputs")
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