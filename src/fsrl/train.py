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
import tempfile
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv
from pathlib import Path

from datasets import load_dataset
from transformer_lens import HookedTransformer
from fsrl import SAEAdapter, HookedModel, SimPOTrainer, SimPOConfig


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


def load_dataset_and_tokenizer(dataset_config: DictConfig, tokenizer, backup_chat_template: str):
    """Load the training dataset and configure the tokenizer."""
    train_dataset = load_dataset(dataset_config.name, split=dataset_config.split)
    
    # Limit dataset size if specified (useful for testing)
    if dataset_config.sample_size is not None:
        train_dataset = train_dataset.select(range(dataset_config.sample_size))
    
    # Set chat template if the tokenizer doesn't have one
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        tokenizer.chat_template = backup_chat_template
    
    return train_dataset


def create_trainer(
    model, 
    tokenizer, 
    train_dataset, 
    training_config: DictConfig
):
    """Create the SimPO trainer."""
    
    training_args = SimPOConfig(**training_config)
    
    trainer = SimPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset
    )
    
    return trainer


def save_adapter_to_wandb(sae: SAEAdapter, cfg: DictConfig, run_name: str = None) -> None:
    """Save the trained adapter and configs to wandb as an artifact."""
    
    # Create a temporary directory to save the adapter
    with tempfile.TemporaryDirectory() as temp_dir:
        adapter_path = Path(temp_dir) / "trained_adapter"
        
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


@hydra.main(version_base=None, config_path="../../config", config_name="config")
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
    train_dataset = load_dataset_and_tokenizer(
        cfg.architecture.dataset, 
        tokenizer, 
        cfg.architecture.backup_chat_template
    )
    
    # Create trainer
    trainer = create_trainer(
        sae_hooked_model, 
        tokenizer, 
        train_dataset, 
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
    trainer.train()
    
    # Save the trained adapter to wandb
    run_name = wandb.run.name if wandb.run else None
    save_adapter_to_wandb(sae, cfg, run_name)
    
    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()