
import torch
import json
import os
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE
import gc
from dotenv import load_dotenv

load_dotenv()

def get_dataset_prompts(dataset_name, num_samples=1000):
    print(f"Loading {dataset_name}...")
    if dataset_name == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split="train")
        dataset = dataset.select(range(min(len(dataset), num_samples)))
        return [q + "\n" + a for q, a in zip(dataset['question'], dataset['answer'])]
    elif dataset_name == "mmlu":
        # Using a collection of subjects to get a representative sample if 'all' is not available directly as a split
        # Actually, cais/mmlu requires a config. Let's use a few diverse subjects.
        subjects = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
        
        # We'll take a few samples from the first few subjects to reach num_samples
        prompts = []
        for subject in subjects:
            if len(prompts) >= num_samples:
                break
            try:
                ds = load_dataset("cais/mmlu", subject, split="test")
                # Just take the question
                for item in ds:
                    prompts.append(item['question'])
                    if len(prompts) >= num_samples:
                        break
            except Exception as e:
                print(f"Error loading MMLU subject {subject}: {e}")
                continue
        return prompts
        
    elif dataset_name == "truthful_qa":
        dataset = load_dataset("truthful_qa", "generation", split="validation")
        dataset = dataset.select(range(min(len(dataset), num_samples)))
        return dataset['question']
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def run_experiment(model_name, sae_release, sae_id, formatting_json_path, cache_dir, dataset_name, batch_size=4):
    print(f"Running experiment for {model_name} on {dataset_name} with SAE {sae_id}")
    
    # Create cache directory for this model and dataset
    model_cache_dir = os.path.join(cache_dir, model_name.replace("/", "_"), dataset_name)
    os.makedirs(model_cache_dir, exist_ok=True)
    
    # Load Model
    print("Loading model...")
    model = HookedTransformer.from_pretrained(model_name, device="cuda", dtype="bfloat16")
    
    # Load SAE
    print("Loading SAE...")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device="cuda"
    )
    sae = sae.to(dtype=torch.bfloat16)
    
    hook_name = sae.cfg.hook_name
    print(f"SAE Hook Name: {hook_name}")
    
    # Load Prompts
    prompts = get_dataset_prompts(dataset_name, num_samples=1000)
    print(f"Loaded {len(prompts)} prompts.")
    
    # Prepare for accumulation
    feature_activations_sum = torch.zeros(sae.cfg.d_sae, device="cuda", dtype=torch.float32)
    feature_fire_count = torch.zeros(sae.cfg.d_sae, device="cuda", dtype=torch.float32)
    total_tokens = 0
    
    # Processing loop
    print("Processing batches...")
    
    # Batching
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]
        
        # Tokenize
        tokens = model.to_tokens(batch_prompts)
        
        # Run model and get activations
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=[hook_name],
                stop_at_layer=sae.cfg.hook_layer + 1 # Optimization
            )
            
            activations = cache[hook_name]
            
            # activations shape: [batch, seq_len, d_model]
            # Flatten batch and seq_len for SAE
            flat_activations = activations.reshape(-1, activations.shape[-1])
            
            # Run SAE
            feature_acts = sae.encode(flat_activations) # [batch*seq_len, d_sae]
            
            # Accumulate sum
            feature_activations_sum += feature_acts.sum(dim=0)
            feature_fire_count += (feature_acts > 0).float().sum(dim=0)
            total_tokens += feature_acts.shape[0]
            
            del activations
            del flat_activations
            del feature_acts
            del cache
            torch.cuda.empty_cache()

    # Calculate average
    average_feature_activations = feature_activations_sum / total_tokens
    feature_firing_frequencies = feature_fire_count / total_tokens
    
    # Save average
    avg_save_path = os.path.join(model_cache_dir, "average_feature_activations.pt")
    torch.save(average_feature_activations.cpu(), avg_save_path)
    print(f"Saved average feature activations to {avg_save_path}")

    freq_save_path = os.path.join(model_cache_dir, "feature_firing_frequencies.pt")
    torch.save(feature_firing_frequencies.cpu(), freq_save_path)
    print(f"Saved feature firing frequencies to {freq_save_path}")

    # Ranking
    print("Ranking features...")
    feature_activations_sum = average_feature_activations.cpu()
    feature_fire_count = feature_fire_count.cpu()
    
    # Get top 10 features
    top_values, top_indices = torch.topk(feature_activations_sum, 10)
    top_indices = top_indices.tolist()
    
    print(f"Top 10 active features: {top_indices}")
    
    # Load Formatting JSON
    print(f"Loading formatting JSON from {formatting_json_path}...")
    with open(formatting_json_path, 'r') as f:
        formatting_data = json.load(f)
    
    style_feature_indices = set()
    for item in formatting_data:
        if item.get("label") == "related":
            fid = item["feature_id"]
            try:
                idx = int(fid.split("-")[-1])
                style_feature_indices.add(idx)
            except ValueError:
                pass
                
    print(f"Found {len(style_feature_indices)} style features.")
    
    # Check if top 10 are style features
    print("Checking top 10 features against style features:")
    for idx in top_indices:
        is_style = idx in style_feature_indices
        print(f"Feature {idx}: Style? {is_style}")
        
    # Metric: Sum of activations of style features / Total activations
    total_activation_sum = feature_activations_sum.sum().item()
    
    style_activation_sum = 0.0
    for idx in style_feature_indices:
        if idx < len(feature_activations_sum):
            style_activation_sum += feature_activations_sum[idx].item()
            
    metric = style_activation_sum / total_activation_sum if total_activation_sum > 0 else 0
    print(f"Metric (Style Act / Total Act): {metric}")

    # Metric L0: Number of times style features fire / Total number of firings
    total_firings = feature_fire_count.sum().item()
    style_firings = 0.0
    for idx in style_feature_indices:
        if idx < len(feature_fire_count):
            style_firings += feature_fire_count[idx].item()
            
    metric_l0 = style_firings / total_firings if total_firings > 0 else 0
    print(f"Metric L0 (Style Firings / Total Firings): {metric_l0}")
    
    # Save results to JSON
    results = {
        "model": model_name,
        "dataset": dataset_name,
        "sae_id": sae_id,
        "top_10_features": top_indices,
        "metric_activation_ratio": metric,
        "metric_l0_ratio": metric_l0,
        "style_features_count": len(style_feature_indices),
        "total_tokens": total_tokens
    }
    
    results_path = os.path.join(model_cache_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")
    
    # Clean up
    del model
    del sae
    del feature_activations_sum
    torch.cuda.empty_cache()
    gc.collect()

def main():
    parser = argparse.ArgumentParser(description="Run SAE caching experiments.")
    parser.add_argument("--dataset", type=str, default="mmlu", choices=["gsm8k", "mmlu", "truthful_qa", "all"], help="Dataset to use (default: mmlu)")
    parser.add_argument("--model", type=str, default="all", choices=["gemma-2-2b", "gemma-2-9b", "all"], help="Model to use (default: all)")
    args = parser.parse_args()

    cache_dir = "/home/ubuntu/evals/feature-steering-RL/models/cache"
    
    if args.dataset == "all":
        datasets = ["gsm8k", "mmlu", "truthful_qa"]
    else:
        datasets = [args.dataset]

    models_to_run = []
    if args.model == "all" or args.model == "gemma-2-2b":
        models_to_run.append("gemma-2-2b")
    if args.model == "all" or args.model == "gemma-2-9b":
        models_to_run.append("gemma-2-9b")

    for model_key in models_to_run:
        if model_key == "gemma-2-2b":
            model_name = "gemma-2-2b-it"
            sae_release = "gemma-scope-2b-pt-res"
            sae_id = "layer_12/width_65k/average_l0_72"
            formatting_json = "/home/ubuntu/evals/feature-steering-RL/outputs/12-gemmascope-res-65k_canonical_formatting_classified_deepseek-deepseek-chat-v3-0324.json"
            batch_size = 4
        elif model_key == "gemma-2-9b":
            model_name = "gemma-2-9b-it"
            sae_release = "gemma-scope-9b-pt-res"
            sae_id = "layer_12/width_16k/average_l0_130"
            formatting_json = "/home/ubuntu/evals/feature-steering-RL/outputs/12-gemmascope-res-16k_canonical_formatting_classified_deepseek-deepseek-chat-v3-0324.json"
            batch_size = 2
        
        for dataset in datasets:
            print(f"\n{'='*20} Processing {model_name} on {dataset} {'='*20}")
            run_experiment(
                model_name=model_name,
                sae_release=sae_release,
                sae_id=sae_id,
                formatting_json_path=formatting_json,
                cache_dir=cache_dir,
                dataset_name=dataset,
                batch_size=batch_size
            )

if __name__ == "__main__":
    main()
