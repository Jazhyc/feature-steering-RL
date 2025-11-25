
import torch
import json
import os
from tqdm import tqdm
from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE
import gc
from dotenv import load_dotenv

load_dotenv()

def run_experiment(model_name, sae_release, sae_id, formatting_json_path, cache_dir, batch_size=4):
    print(f"Running experiment for {model_name} with SAE {sae_id}")
    
    # Create cache directory for this model
    model_cache_dir = os.path.join(cache_dir, model_name.replace("/", "_"))
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
    
    # Load Dataset
    print("Loading GSM8k dataset...")
    dataset = load_dataset("gsm8k", "main", split="train")
    dataset = dataset.select(range(1000))
    
    # Prepare for accumulation
    feature_activations_sum = torch.zeros(sae.cfg.d_sae, device="cuda", dtype=torch.float32)
    feature_fire_count = torch.zeros(sae.cfg.d_sae, device="cuda", dtype=torch.float32)
    total_tokens = 0
    
    # Processing loop
    print("Processing batches...")
    prompts = [q + "\n" + a for q, a in zip(dataset['question'], dataset['answer'])]
    
    # Batching
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]
        
        # Tokenize
        # We use the model's tokenizer. 
        # We need to handle padding if batch_size > 1
        # HookedTransformer handles tokenization in run_with_cache if passed strings, 
        # but for batching with padding it's better to tokenize first.
        
        # Actually HookedTransformer.to_tokens handles padding if we pass a list of strings
        tokens = model.to_tokens(batch_prompts)
        
        # Run model and get activations
        # We only need the activation at the hook point
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
            # sae(x) returns the reconstruction tensor in this version
            # sae.encode(x) returns the feature activations
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
    # feature_activations_sum is on GPU, move to CPU
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
    
    # Parse formatting features
    # The json has "feature_id": "gemma-2-9b-12-gemmascope-res-16k-8"
    # We need to extract the index.
    # Assuming format "...-{index}"
    
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
    
    # Clean up
    del model
    del sae
    del feature_activations_sum
    torch.cuda.empty_cache()
    gc.collect()

def main():
    cache_dir = "/home/ubuntu/evals/feature-steering-RL/models/cache"
    
    # Gemma 2B
    # sae_id: "layer_12/width_65k/average_l0_72"
    # formatting json: 65k
    run_experiment(
        model_name="gemma-2-2b-it",
        sae_release="gemma-scope-2b-pt-res",
        sae_id="layer_12/width_65k/average_l0_72",
        formatting_json_path="/home/ubuntu/evals/feature-steering-RL/outputs/12-gemmascope-res-65k_canonical_formatting_classified_deepseek-deepseek-chat-v3-0324.json",
        cache_dir=cache_dir,
        batch_size=4
    )
    
    print("-" * 50)
    
    # Gemma 9B
    # sae_id: "layer_12/width_16k/average_l0_130"
    # formatting json: 16k
    run_experiment(
        model_name="gemma-2-9b-it",
        sae_release="gemma-scope-9b-pt-res",
        sae_id="layer_12/width_16k/average_l0_130",
        formatting_json_path="/home/ubuntu/evals/feature-steering-RL/outputs/12-gemmascope-res-16k_canonical_formatting_classified_deepseek-deepseek-chat-v3-0324.json",
        cache_dir=cache_dir,
        batch_size=2 # 9B is larger, reduce batch size
    )

if __name__ == "__main__":
    main()
