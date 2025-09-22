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

def run_eval(runs, tasks, limit=0.01, with_adapter=True, full_ft=False):
    
    root = Path(__file__).resolve().parent.parent.parent  # climb up to project root
    models_path = f"{root}/models/Gemma2-2B-clean" if not full_ft else f"{root}/models/full-gemma2_2B"
    
    load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(entity="feature-steering-RL", project="lm-eval")

    downloader = WandBModelDownloader(
        entity="feature-steering-RL",
        project="Gemma2-2B-clean" if not full_ft else "full-gemma2_2B",
        verbose=True
    )

    summary_results = {}

    for run in runs:
        print(f"##### Running evaluation for run: {run} #####")
        print("=" * 30)
        run_objs = downloader.api.runs("feature-steering-rl/full-gemma2_2B", filters={"display_name": run})
        downloader.download_model(run_objs[0], models_path)
        
        # Create a fresh base model for each adapter to avoid hook point conflicts
        print(f"Loading fresh base model for run: {run}")
        if full_ft:
            base_model_path = downloader.models_base_dir / "full-gemma2_2B" / run / "full_model"
            base_model = BaseHookedModel.from_pretrained(base_model_path, device="cuda", dtype=torch.bfloat16)
            hooked_model = base_model
        else:
            base_model = HookedTransformer.from_pretrained_no_processing("google/gemma-2-2b-it", device="cuda", dtype=torch.bfloat16)

            adapter_path = downloader.models_base_dir / "Gemma2-2B-clean" / run / "adapter"

            print(f"Loading adapter from: {adapter_path}")
            sae_adapter = SAEAdapter.load_from_pretrained_adapter(adapter_path, device="cuda")
            hooked_model = HookedModel(base_model, sae_adapter)
        
            # Disable steering if not using adapter
            if not with_adapter:
                hooked_model.disable_steering()
                print(f"Evaluating model without steering")
        
        tokenizer = base_model.tokenizer

        # For HookedTransformer without HookedModel wrapper, we need to access the actual PyTorch model
        if hasattr(hooked_model, 'cfg') and hasattr(hooked_model, 'model'):
            # This is a HookedTransformer, get the actual PyTorch model
            hf_model = hooked_model.model
            print(f"Debug: Using HookedTransformer's underlying model")
        else:
            # Fallback - use model directly
            hf_model = hooked_model
            print(f"Debug: Using model directly")
        
        # start with a large batch size and reduce it if we get an error
        initial_batch_size = 16
        batch_size = initial_batch_size
        while True:
            try:
                # Create HFLM wrapper for lm_eval
                eval_model = HFLM(pretrained=hf_model, tokenizer=tokenizer, batch_size=batch_size)
                
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
    
        summary_results[run] = results["results"]

    # Save results as a JSON file and upload as WandB artifact
    results_file = "eval_results.json"
    with open(results_file, 'w') as f:
        json.dump(summary_results, f, indent=2)
    
    artifact = wandb.Artifact("eval_results", type="results")
    artifact.add_file(results_file)
    wandb.log_artifact(artifact)

    # Log summary results for dashboard viewing
    wandb.log(summary_results)

    os.remove(results_file)

    return summary_results

def pretty_results(results):
    for run, task_results in results.items():
        print(f"Run: {run}")
        for task, result in task_results.items():
            print(f"Task: {task}")
            print(f"Result: {result}")
            print("-" * 100) 