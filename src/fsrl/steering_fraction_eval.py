#!/usr/bin/env python3
"""
Experiment: Evaluate validation loss as we limit the adapter's steering vector
to the top-k features by absolute value. Sweep in intervals and log to W&B.

This reuses model/dataset loading from train.py and utilities across the repo.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

from transformer_lens import HookedTransformer

from fsrl import SAEAdapter, HookedModel, SimPOTrainer, SimPOConfig
from fsrl.train import load_dataset_and_tokenizer
from fsrl.utils.wandb_utils import WandBModelDownloader


dtype_map = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Optional progress bar for fraction sweep
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


def setup_env():
    load_dotenv()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_base_model(model_cfg) -> HookedTransformer:
    device = model_cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    dtype = dtype_map.get(model_cfg.dtype, torch.bfloat16)
    model = HookedTransformer.from_pretrained_no_processing(
        model_name=model_cfg.name,
        device=device,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    return model


def load_sae_adapter(device: str, adapter_local_path: str) -> SAEAdapter:
    adapter_path = Path(adapter_local_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"adapter_local_path '{adapter_local_path}' not found")
    sae = SAEAdapter.load_from_pretrained_adapter(adapter_path, device=device)
    return sae


def ensure_models_available(cfg: DictConfig) -> Optional[str]:
    """Optionally download a specific run's artifacts and return adapter path."""
    if not cfg.get("auto_download", False):
        return None
    family = cfg.wandb_project_family
    run_name = cfg.get("wandb_run_name")
    if not run_name:
        raise ValueError("auto_download=true but wandb_run_name is not set")
    downloader = WandBModelDownloader(entity=cfg.wandb_entity, project=family, verbose=True)
    # Find by display_name (W&B API.run expects run ID, not name)
    runs = downloader.api.runs(f"{cfg.wandb_entity}/{family}", filters={"display_name": run_name})
    if not runs:
        raise ValueError(f"Run '{run_name}' not found in {cfg.wandb_entity}/{family}")
    run = runs[0]
    download_path = downloader.download_model(run, family, force_download=cfg.get("force_download", False))
    if download_path is None:
        raise RuntimeError(f"Failed to download artifacts for run '{run_name}'")
    adapter_path = Path(download_path) / "adapter"
    if not adapter_path.exists():
        # Fallback: look for first subdir containing adapter_weights.safetensors
        for p in Path(download_path).rglob("adapter_weights.safetensors"):
            return str(p.parent)
        raise FileNotFoundError(f"Adapter folder not found under {download_path}")
    return str(adapter_path)


def load_eval_dataset(dataset_cfg, tokenizer):
    # Reuse training utility to ensure identical processing
    _, eval_dataset = load_dataset_and_tokenizer(dataset_cfg, tokenizer)
    return eval_dataset


def make_trainer(model, tokenizer, eval_dataset, training_cfg) -> SimPOTrainer:
    cfg = OmegaConf.to_container(training_cfg, resolve=True)
    # Ensure batch size 2 as requested
    cfg["per_device_train_batch_size"] = 2
    cfg["per_device_eval_batch_size"] = 2
    cfg["gradient_accumulation_steps"] = cfg.get("gradient_accumulation_steps", 1)
    cfg["eval_steps"] = cfg.get("eval_steps", 50)
    cfg["dataset_num_proc"] = cfg.get("dataset_num_proc", 1) or 1
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


def sweep_and_log(trainer: SimPOTrainer, model, fractions: List[float]):
    results = []
    for frac in tqdm(fractions, desc="Sweeping steering fractions"):
        if isinstance(model, HookedModel):
            model.set_steering_fraction(frac)
        # Evaluate; use trainer's evaluate to compute loss
        metrics = trainer.evaluate()
        metrics = {**metrics, "steering_fraction": frac}
        wandb.log({"eval/loss": metrics.get("eval_loss"), "steering_fraction": frac})
        results.append(metrics)
    return results


@hydra.main(version_base=None, config_path="../../config", config_name="steering_fraction_eval")
def main(cfg: DictConfig) -> None:
    setup_env()

    # Optional W&B artifact download (single run)
    downloaded_adapter_path = ensure_models_available(cfg)

    # Load model + SAE adapter
    base_model = load_base_model(cfg.architecture.model)
    tokenizer = base_model.tokenizer

    # Always use HookedModel with a local adapter
    adapter_local_path = cfg.architecture.get("adapter_local_path") or downloaded_adapter_path
    if not adapter_local_path:
        raise ValueError("adapter_local_path must be set or auto_download configured with wandb_run_name")
    sae = load_sae_adapter(cfg.architecture.model.device, adapter_local_path)
    model = HookedModel(base_model, sae)

    # Dataset
    eval_dataset = load_eval_dataset(cfg.architecture.dataset, tokenizer)

    # Trainer for evaluation
    trainer = make_trainer(model, tokenizer, eval_dataset, cfg.training)

    # W&B init (separate project)
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

    # Build fractions from intervals (e.g., 10 -> [0.1, 0.2, ..., 1.0])
    step = cfg.get("interval_percent", 10)
    fractions = [i / 100.0 for i in range(step, 101, step)]

    results = sweep_and_log(trainer, model, fractions)

    # Log table summary
    if results:
        table = wandb.Table(columns=list(results[0].keys()))
        for r in results:
            table.add_data(*[r.get(c) for c in table.columns])
        wandb.log({"steering_fraction_eval": table})

    wandb.finish()


if __name__ == "__main__":
    # Filter out deepspeed local rank if present
    args_to_pass = [arg for arg in sys.argv if not arg.startswith('--local_rank')]
    script_name = sys.argv[0]
    sys.argv = [script_name] + args_to_pass[1:]
    main()
