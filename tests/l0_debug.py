# src/fsrl/scripts/debug_l0_discrepancy.py (Final Corrected Version)

import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from typing import Any
from torch.amp import autocast # Use torch.amp.autocast for modern PyTorch

# Import the key function to fix the discrepancy
from trl.trainer.utils import DPODataCollatorWithPadding, disable_dropout_in_model

# --- Configuration ---
MODEL_NAME = "gemma-2-2b-it"
DEFAULT_ADAPTER_PATH = "models/Gemma2-2B-clean/mild-glade-10/adapter" # ADJUST IF NEEDED
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2

# --- Helper Functions (self-contained for portability) ---

def apply_chat_template(example, tokenizer, chat_template, **kwargs):
    """Applies the chat template to a single example from the dataset."""
    tokenizer.chat_template = chat_template
    if not all(k in example.keys() for k in ("chosen", "rejected")):
        raise ValueError("Example must have 'chosen' and 'rejected' keys.")
    
    prompt_messages = example["chosen"][:-1]
    chosen_messages = example["chosen"][-1:]
    rejected_messages = example["rejected"][-1:]
    
    example["text_prompt"] = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    example["text_chosen"] = tokenizer.apply_chat_template(
        chosen_messages, tokenize=False
    )
    if example["text_chosen"].startswith(str(tokenizer.bos_token)):
        example["text_chosen"] = example["text_chosen"][len(str(tokenizer.bos_token)):]
    
    example["text_rejected"] = tokenizer.apply_chat_template(
        rejected_messages, tokenize=False
    )
    if example["text_rejected"].startswith(str(tokenizer.bos_token)):
        example["text_rejected"] = example["text_rejected"][len(str(tokenizer.bos_token)):]
        
    return example

def load_dataset_local(dataset_config, tokenizer):
    """Loads and preprocesses the evaluation dataset."""
    eval_dataset = load_dataset(dataset_config["name"], split=dataset_config["eval_split"])
    column_names = list(eval_dataset.features)
    
    eval_dataset = eval_dataset.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "simpo",
            "chat_template": dataset_config["chat_template"],
        },
        num_proc=dataset_config["dataset_num_proc"],
        remove_columns=column_names,
    )
    
    mapping = {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
    eval_dataset = eval_dataset.rename_columns(mapping)
    return eval_dataset

def get_one_batch(dataset, tokenizer, batch_size):
    """Collates a single batch of data and safely moves tensors to the correct device."""
    data_collator = DPODataCollatorWithPadding(pad_token_id=tokenizer.pad_token_id)
    subset = dataset.select(range(batch_size))
    
    batch = data_collator([item for item in subset])
    
    prepared_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            prepared_batch[key] = value.to(DEVICE)
        else:
            prepared_batch[key] = value
    return prepared_batch

def main():
    print("--- L0 Discrepancy Debugger (Replication Test) ---")

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    from transformer_lens import HookedTransformer
    from fsrl import SAEAdapter, HookedModel
    from fsrl.simPO import SimPOTrainer, SimPOConfig

    print("Loading model and tokenizer...")
    base_model = HookedTransformer.from_pretrained_no_processing(
        MODEL_NAME, device=DEVICE, torch_dtype=torch.bfloat16
    )
    tokenizer = base_model.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading SAE adapter from: {DEFAULT_ADAPTER_PATH}")
    sae = SAEAdapter.load_from_pretrained_adapter(DEFAULT_ADAPTER_PATH, device=DEVICE)
    model = HookedModel(base_model, sae)

    print("\nDisabling dropout in model to match trainer's state...")
    disable_dropout_in_model(model)
    model.eval()

    print("Loading and preparing one evaluation batch...")
    dataset_config = {
        "name": "princeton-nlp/llama3-ultrafeedback-armorm",
        "eval_split": "test",
        "dataset_num_proc": 1,
        "chat_template": "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] | trim + '\n\n' %}{% set messages = messages[1:] %}{% else %}{% set system_message = '' %}{% endif %}{% for message in messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + content | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"
    }
    eval_dataset_raw = load_dataset_local(dataset_config, tokenizer)
    
    training_args_for_tokenization = SimPOConfig(
        output_dir="./tmp_debug", max_length=2048, max_prompt_length=1800, 
        per_device_eval_batch_size=BATCH_SIZE, remove_unused_columns=False
    )
    trainer_for_tokenization = SimPOTrainer(
        model=model, args=training_args_for_tokenization, processing_class=tokenizer
    )
    tokenized_eval_dataset = eval_dataset_raw.map(trainer_for_tokenization.tokenize_row, num_proc=1)
    
    batch = get_one_batch(tokenized_eval_dataset, tokenizer, BATCH_SIZE)

    # --- 4. RUN ANALYSIS PIPELINE (CORRECTED) ---
    print("\n--- Running Analysis Pipeline (on fixed model) ---")
    
    chosen_ids = batch['chosen_input_ids']
    rejected_ids = batch['rejected_input_ids']
    chosen_mask = batch['chosen_attention_mask']
    rejected_mask = batch['rejected_attention_mask']

    max_len = max(chosen_ids.shape[1], rejected_ids.shape[1])
    pad_token_id = tokenizer.pad_token_id
    
    if chosen_ids.shape[1] < max_len:
        pad_len = max_len - chosen_ids.shape[1]
        chosen_ids = F.pad(chosen_ids, (0, pad_len), value=pad_token_id)
        chosen_mask = F.pad(chosen_mask, (0, pad_len), value=0)
    if rejected_ids.shape[1] < max_len:
        pad_len = max_len - rejected_ids.shape[1]
        rejected_ids = F.pad(rejected_ids, (0, pad_len), value=pad_token_id)
        rejected_mask = F.pad(rejected_mask, (0, pad_len), value=0)
        
    concatenated_ids = torch.cat([chosen_ids, rejected_ids], dim=0)
    concatenated_mask = torch.cat([chosen_mask, rejected_mask], dim=0)

    with torch.no_grad(), autocast("cuda", enabled=True, dtype=torch.bfloat16):
        _, cache = model.run_with_cache(
            concatenated_ids, 
            attention_mask=concatenated_mask,
            prepend_bos=False
        )
        steering_vector_full = cache["blocks.12.hook_resid_post.hook_sae_adapter"].cpu().float().numpy()
        
        # --- START: THE FIX ---
        # To replicate the trainer, we must perform the same "naive" averaging
        # that includes padding tokens.
        active_features_per_pos = np.count_nonzero(np.abs(steering_vector_full) > 1e-6, axis=-1)
        
        # This is the trainer's calculation: a simple mean over all positions.
        analysis_l0 = np.mean(active_features_per_pos)
        # --- END: THE FIX ---

        print(f"L0 Norm from Analysis Pipeline: {analysis_l0:.4f}")

    # --- 5. RUN TRAINER PIPELINE (for verification) ---
    print("\n--- Running Trainer's Pipeline (for verification) ---")
    
    trainer = SimPOTrainer(model=model, args=training_args_for_tokenization, processing_class=tokenizer)
    with torch.no_grad():
        _, metrics = trainer.get_batch_loss_metrics(model, batch, train_eval="eval")
        trainer_l0_norm = metrics["eval_steering_vector/l0_norm_sparsity"].item()
        print(f"L0 Norm from Trainer's Pipeline: {trainer_l0_norm:.4f}")

    # --- FINAL COMPARISON ---
    print("\n" + "="*40)
    print("--- Final Comparison ---")
    print(f"Analysis L0:  {analysis_l0:.4f}")
    print(f"Trainer L0:   {trainer_l0_norm:.4f}")
    
    diff = abs(trainer_l0_norm - analysis_l0)
    if diff < 1e-3: 
        print("\n✅ SUCCESS: The results now match perfectly!")
        print("The cause was the L0 norm calculation including padding tokens in the trainer.")
    else:
        print(f"\n❌ FAILURE: Discrepancy of {diff:.4f} still exists.")
    print("="*40)

if __name__ == "__main__":
    main()