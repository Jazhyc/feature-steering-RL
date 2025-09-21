#!/usr/bin/env python3
"""
Experiment: Evaluate the base model (without any adapters) on the ultrafeedback dataset.
This provides a baseline comparison for adapter performance.

This reuses model/dataset loading from train.py and utilities across the repo.
"""

import os
import sys
from pathlib import Path

import torch
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

from transformer_lens import HookedTransformer

from fsrl import SimPOTrainer, SimPOConfig
from fsrl.hooked_model import BaseHookedModel
from fsrl.train import load_dataset_and_tokenizer


dtype_map = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def setup_env():
    load_dotenv()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_base_model(model_cfg) -> BaseHookedModel:
    device = model_cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    dtype = dtype_map.get(model_cfg.dtype, torch.bfloat16)
    
    # Load the base HookedTransformer
    hooked_transformer = HookedTransformer.from_pretrained_no_processing(
        model_name=model_cfg.name,
        device=device,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    
    # Wrap it in BaseHookedModel for trainer compatibility
    model = BaseHookedModel(hooked_transformer)
    return model


def load_eval_dataset(dataset_cfg, tokenizer):
    # Reuse training utility to ensure identical processing
    _, eval_dataset = load_dataset_and_tokenizer(dataset_cfg, tokenizer)
    return eval_dataset


def make_trainer(model, tokenizer, eval_dataset, training_cfg) -> SimPOTrainer:
    cfg = OmegaConf.to_container(training_cfg, resolve=True)
    # Ensure batch size settings
    cfg["per_device_train_batch_size"] = cfg.get("per_device_train_batch_size", 2)
    cfg["per_device_eval_batch_size"] = cfg.get("per_device_eval_batch_size", 4)
    cfg["gradient_accumulation_steps"] = cfg.get("gradient_accumulation_steps", 1)
    cfg["eval_steps"] = cfg.get("eval_steps", 100)
    cfg["dataset_num_proc"] = cfg.get("dataset_num_proc", 20) or 20
    # Match gemma2_2B training defaults unless overridden in YAML
    cfg["max_prompt_length"] = cfg.get("max_prompt_length", 1800)
    cfg["max_length"] = cfg.get("max_length", 2048)
    # Ensure SimPO key params are present (beta and gamma ratio)
    cfg["beta"] = cfg.get("beta", 10)
    cfg["gamma_beta_ratio"] = cfg.get("gamma_beta_ratio", 0.5)
    
    args = SimPOConfig(**cfg)
    trainer = SimPOTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        train_dataset=None,
        eval_dataset=eval_dataset,
    )
    return trainer


@hydra.main(version_base=None, config_path="../../../config", config_name="base_model_eval")
def main(cfg: DictConfig) -> None:
    setup_env()

    # Load base model (no adapters)
    model = load_base_model(cfg.architecture.model)
    tokenizer = model.model.tokenizer

    # Dataset
    eval_dataset = load_eval_dataset(cfg.architecture.dataset, tokenizer)

    # Trainer for evaluation
    trainer = make_trainer(model, tokenizer, eval_dataset, cfg.training)

    # W&B init
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        dir=cfg.wandb.dir,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=cfg.wandb.get("tags", []),
        notes=cfg.wandb.get("notes", ""),
        mode=cfg.wandb.get("mode", "online"),
        name=cfg.wandb.get("name", None),
    )

    # Evaluate the base model
    print("Evaluating base model on ultrafeedback dataset...")
    metrics = trainer.evaluate()
    
    # Log the results
    wandb.log({
        "eval/loss": metrics.get("eval_loss"),
        "model_type": "base_model"
    })
    
    print(f"Base model evaluation loss: {metrics.get('eval_loss', 'N/A')}")
    
    # Log summary table
    results_table = wandb.Table(columns=["model_type", "eval_loss"])
    results_table.add_data("base_model", metrics.get("eval_loss"))
    wandb.log({"base_model_eval": results_table})

    wandb.finish()


if __name__ == "__main__":
    # Filter out deepspeed local rank if present
    args_to_pass = [arg for arg in sys.argv if not arg.startswith('--local_rank')]
    script_name = sys.argv[0]
    sys.argv = [script_name] + args_to_pass[1:]
    main()