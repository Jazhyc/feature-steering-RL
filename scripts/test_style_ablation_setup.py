#!/usr/bin/env python3
"""
Test script to verify style ablation configuration loads correctly.
"""

import json
from pathlib import Path

def test_style_ablation_config():
    """Test that the style classification file can be loaded correctly."""
    
    # Path to the classification file
    style_file = Path("outputs/feature_classification/gemma-2-2b/12-gemmascope-res-65k_canonical_formatting_classified_deepseek-deepseek-chat-v3-0324.json")
    
    if not style_file.exists():
        print(f"❌ Error: Style classification file not found at {style_file}")
        return False
    
    print(f"✓ Found style classification file: {style_file}")
    
    # Load and parse the file
    with open(style_file, 'r') as f:
        classifications = json.load(f)
    
    print(f"✓ Successfully loaded {len(classifications)} feature classifications")
    
    # Count style features (label='related')
    style_features = []
    for item in classifications:
        if item.get('label') == 'related':
            feature_id = item['feature_id']
            feature_idx = int(feature_id.split('-')[-1])
            style_features.append(feature_idx)
    
    print(f"✓ Found {len(style_features)} style features to ablate")
    print(f"  Example feature IDs: {style_features[:5]}...")
    
    # Verify config file exists
    config_file = Path("config/gemma2_2B_train_ablation.yaml")
    if not config_file.exists():
        print(f"❌ Error: Config file not found at {config_file}")
        return False
    
    print(f"✓ Found config file: {config_file}")
    
    print("\n✅ Configuration is ready!")
    print(f"\nTo start training with style features ablated, run:")
    print(f"  python -m fsrl.train --config-name=gemma2_2B_train_ablation")
    
    return True

if __name__ == "__main__":
    test_style_ablation_config()
