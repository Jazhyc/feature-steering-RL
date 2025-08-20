import os
import lm_eval
import torch
import wandb
from dotenv import load_dotenv
from pathlib import Path
from fsrl import SAEAdapter, HookedModel
from fsrl.utils.wandb_utils import (
    WandBModelDownloader,
    download_model_family,
    list_model_family,
)
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval.models.huggingface import HFLM

def run_eval(runs, tasks, limit=0.01):
    
    root = Path(__file__).resolve().parent.parent  # climb up to project root
    models_path = f"{root}/models/Gemma2-2B-clean"
    
    load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(entity="feature-steering-RL", project="lm-eval")

    downloader = WandBModelDownloader(
        entity="feature-steering-RL",
        project="Gemma2-2B-clean",
        verbose=True
    )

    full_results = {}

    for run in runs:
        run_objs = downloader.api.runs("feature-steering-RL/Gemma2-2B-clean", filters={"display_name": run})
        downloader.download_model(run_objs[0], models_path)
        
        adapter_path = downloader.models_base_dir / "Gemma2-2B-clean" / run / "adapter"

        print(f"Loading adapter from: {adapter_path}")
        sae_adapter = SAEAdapter.load_from_pretrained_adapter(adapter_path, device="cuda")
        
        # Create a fresh base model for each adapter to avoid hook point conflicts
        print(f"Loading fresh base model for run: {run}")
        base_model = HookedTransformer.from_pretrained("google/gemma-2-2b-it", device="cuda", dtype=torch.bfloat16)
        tokenizer = base_model.tokenizer
        
        hooked_model = HookedModel(base_model, sae_adapter)

        eval_model = HFLM(pretrained=hooked_model, tokenizer=tokenizer, batch_size=16)
            
        #task_manager = lm_eval.tasks.TaskManager()

        results = lm_eval.simple_evaluate(
            model=eval_model,
            tasks=tasks,
            #task_manager=task_manager,
            limit=limit,
            apply_chat_template=True,
        )
    
        full_results[run] = results

    wandb.log(full_results)

    return full_results

def pretty_results(results):
    for run, task_results in results.items():
        print(f"Run: {run}")
        for task, result in task_results.items():
            print(f"Task: {task}")
            print(f"Result: {result}")
            print("-" * 100)