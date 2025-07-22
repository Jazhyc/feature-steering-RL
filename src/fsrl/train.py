#!/usr/bin/env python3
"""
Feature Steering RL Training Script

This script provides a configurable training pipeline for SimPO training
using Sparse Autoencoders (SAEs) with transformer models.
"""

import os
import torch
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv
from pathlib import Path

from datasets import load_dataset
from transformer_lens import HookedTransformer
from fsrl import SAEAdapter, HookedModel, SimPOTrainer, SimPOConfig, apply_chat_template


def setup_environment(wandb_config: DictConfig) -> None:
    """Set up environment variables for training."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Set tokenizer parallelism to false for use with multiple workers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_model_and_tokenizer(model_config: DictConfig) -> tuple:
    """Load the base model and tokenizer."""
    # Auto-detect device if not specified or if cuda is requested but not available
    device = model_config.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(model_config.dtype, torch.bfloat16)
    
    model = HookedTransformer.from_pretrained(
        model_name=model_config.name,
        device=device, 
        torch_dtype=dtype,
        attn_implementation="sdpa", # Compile with SDPA is faster than FA2?
    )
    
    tokenizer = model.tokenizer
    
    return model, tokenizer, device


def load_sae_adapter(sae_config: DictConfig, device: str) -> SAEAdapter:
    """Load and configure the SAE adapter."""
    adapter_kwargs = {
        "use_lora_adapter": sae_config.use_lora_adapter,
        "lora_rank": sae_config.lora_rank,
        "lora_alpha": sae_config.lora_alpha,
        "fusion_mode": sae_config.fusion_mode,
    }
    
    sae, cfg_dict, sparsity = SAEAdapter.from_pretrained(
        sae_config.release, 
        sae_config.sae_id, 
        device=device, 
        **adapter_kwargs
    )
    
    return sae


def load_dataset_and_tokenizer(dataset_config: DictConfig, tokenizer):
    """Load the training dataset and configure the tokenizer."""
    train_dataset = load_dataset(dataset_config.name, split=dataset_config.train_split)
    eval_dataset = load_dataset(dataset_config.name, split=dataset_config.eval_split)
    
    # Limit dataset size if specified (useful for testing)
    if dataset_config.sample_size is not None:
        train_dataset = train_dataset.select(range(dataset_config.sample_size))

    column_names = list(train_dataset.features)
    
    train_dataset = train_dataset.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "simpo",
            "chat_template": dataset_config.chat_template,
        },
        num_proc=dataset_config.dataset_num_proc,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )
    
    eval_dataset = eval_dataset.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "simpo",
            "chat_template": dataset_config.chat_template,
        },
        num_proc=dataset_config.dataset_num_proc,
        remove_columns=column_names, # Removes original prompt, chosen, rejected columns
        desc="Formatting comparisons with prompt template",
    )
    
    # Convert text columns to match SimPOTrainer expectations
    mapping = {
        "text_prompt": "prompt",
        "text_chosen": "chosen",
        "text_rejected": "rejected"
    }
    
    # Rename columns to match SimPOTrainer expectations
    train_dataset = train_dataset.rename_columns(mapping)
    eval_dataset = eval_dataset.rename_columns(mapping)

    return train_dataset, eval_dataset


def create_trainer(
    model, 
    tokenizer, 
    train_dataset,
    eval_dataset: None,
    training_config: DictConfig
):
    """Create the SimPO trainer."""
    
    training_config = OmegaConf.to_container(training_config, resolve=True)
    eval_frequency = training_config.get("eval_epoch_fraction", 0.1)
    
    per_device_batch_size = training_config["per_device_train_batch_size"]
    gradient_accumulation_steps = training_config["gradient_accumulation_steps"]
    effective_batch_size = per_device_batch_size * gradient_accumulation_steps
    training_config["eval_steps"] = int(len(train_dataset) * eval_frequency / effective_batch_size)
    training_config.pop("eval_epoch_fraction", None)
    
    training_args = SimPOConfig(**training_config)
    
    trainer = SimPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    return trainer


def save_adapter_to_wandb(sae: SAEAdapter, cfg: DictConfig, run_name: str = None) -> None:
    """Save the trained adapter and configs to wandb as an artifact."""
    
    # Use run_name as the folder name, fallback to "trained_adapter" if no run_name
    folder_name = run_name if run_name else "trained_adapter"
    
    # Create a permanent model directory using the path from config
    model_dir = Path(cfg.models_dir)
    model_dir.mkdir(exist_ok=True)
    
    adapter_path = model_dir / folder_name
    
    # Save the adapter locally first
    sae.save_adapter(adapter_path)
    
    # Create wandb artifact
    artifact_name = f"adapter-{run_name}" if run_name else "trained-adapter"
    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        description="Trained SAE adapter with LoRA weights and configuration",
        metadata={
            "fusion_mode": sae.fusion_mode,
            "use_lora_adapter": sae.use_lora_adapter,
            "lora_rank": sae.lora_rank if sae.use_lora_adapter else None,
            "lora_alpha": sae.lora_alpha if sae.use_lora_adapter else None,
            "base_sae_release": sae.cfg.release,
            "base_sae_id": sae.cfg.sae_id,
            "training_config": OmegaConf.to_container(cfg.training, resolve=True),
            "architecture_config": OmegaConf.to_container(cfg.architecture, resolve=True)
        }
    )
    
    # Add all files from the adapter directory to the artifact
    artifact.add_dir(str(adapter_path), name="adapter")
    
    # Log the artifact to wandb
    wandb.log_artifact(artifact)
    
    print(f"Adapter saved to wandb as artifact: {artifact_name}")
    print(f"Local copy saved to: {adapter_path.absolute()}")


@hydra.main(version_base=None, config_path="../../config", config_name="gpt2")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    # Set up environment
    setup_environment(cfg.wandb)
    
    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(cfg.architecture.model)
    
    # Load SAE adapter
    sae = load_sae_adapter(cfg.architecture.sae, device)
    
    # Create hooked model
    sae_hooked_model = HookedModel(model, sae)
    
    # Load dataset
    train_dataset, eval_dataset = load_dataset_and_tokenizer(
        cfg.architecture.dataset, 
        tokenizer, 
    )
    
    # Create trainer
    trainer = create_trainer(
        sae_hooked_model, 
        tokenizer, 
        train_dataset,
        eval_dataset,
        cfg.training
    )
    
    # Initialize wandb just before training
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        dir=cfg.wandb.dir,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=cfg.wandb.get("tags", []),
        notes=cfg.wandb.get("notes", ""),
        mode=cfg.wandb.get("mode", "online")
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save the trained adapter to wandb
    run_name = wandb.run.name if wandb.run else None
    save_adapter_to_wandb(sae, cfg, run_name)
    
    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()