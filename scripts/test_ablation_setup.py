#!/usr/bin/env python3
"""
Test script to verify feature ablation configuration loads correctly.
"""

import json
from pathlib import Path

def test_ablation_config(config_name: str, classification_file: str):
    """Test that the classification file can be loaded correctly."""
    
    # Path to the classification file
    file_path = Path(classification_file)
    
    if not file_path.exists():
        print(f"❌ Error: Classification file not found at {file_path}")
        return False
    
    print(f"✓ Found classification file: {file_path}")
    
    # Load and parse the file
    with open(file_path, 'r') as f:
        classifications = json.load(f)
    
    print(f"✓ Successfully loaded {len(classifications)} feature classifications")
    
    # Count features to ablate (label='related')
    ablation_features = []
    for item in classifications:
        if item.get('label') == 'related':
            feature_id = item['feature_id']
            feature_idx = int(feature_id.split('-')[-1])
            ablation_features.append(feature_idx)
    
    print(f"✓ Found {len(ablation_features)} features to ablate during training")
    print(f"  Example feature IDs: {ablation_features[:5]}...")
    
    # Verify config file exists
    config_file = Path(f"config/{config_name}.yaml")
    if not config_file.exists():
        print(f"❌ Error: Config file not found at {config_file}")
        return False
    
    print(f"✓ Found config file: {config_file}")
    
    return True


def main():
    print("=" * 70)
    print("Testing Feature Ablation Training Configurations")
    print("=" * 70)
    
    # Test 2B model config
    print("\n[1] Testing Gemma2-2B configuration:")
    print("-" * 70)
    success_2b = test_ablation_config(
        "gemma2_2B_train_ablation",
        "outputs/feature_classification/gemma-2-2b/12-gemmascope-res-65k_canonical_formatting_classified_deepseek-deepseek-chat-v3-0324.json"
    )
    
    # Test 9B model config
    print("\n[2] Testing Gemma2-9B configuration:")
    print("-" * 70)
    success_9b = test_ablation_config(
        "gemma2_9B_train_ablation",
        "outputs/feature_classification/gemma-2-9b/12-gemmascope-res-16k_canonical_formatting_classified_deepseek-deepseek-chat-v3-0324.json"
    )
    
    print("\n" + "=" * 70)
    if success_2b and success_9b:
        print("✅ All configurations are ready!")
        print("=" * 70)
        print("\n📖 USAGE INSTRUCTIONS:")
        print("-" * 70)
        print("\n1. Train Gemma2-2B with feature ablation:")
        print("   python -m fsrl.train --config-name=gemma2_2B_train_ablation")
        print("\n2. Train Gemma2-9B with feature ablation:")
        print("   python -m fsrl.train --config-name=gemma2_9B_train_ablation")
        print("\n3. Override classification file at runtime:")
        print("   python -m fsrl.train --config-name=gemma2_2B_train_ablation \\")
        print("     ablation_classification_file=/path/to/custom_classification.json")
        print("\n4. Disable ablation for a run:")
        print("   python -m fsrl.train --config-name=gemma2_2B_train_ablation \\")
        print("     ablation_classification_file=null")
        print("\n📝 NOTES:")
        print("-" * 70)
        print("• Features with label='related' in the classification file will be ablated")
        print("• Currently configured for style/formatting features")
        print("• To ablate alignment features instead, update ablation_classification_file to:")
        print("  - 2B: ...12-gemmascope-res-65k_canonical_alignment_classified_...")
        print("  - 9B: ...12-gemmascope-res-16k_canonical_alignment_classified_...")
        print("• Model will be saved to models/Gemma2-{2B|9B}-train-ablation/")
        print("• WandB tags: ['feature-ablation', 'feature-masking']")
        print("=" * 70)
    else:
        print("❌ Some configurations failed - please check the errors above")
        print("=" * 70)

if __name__ == "__main__":
    main()
