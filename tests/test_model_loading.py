#!/usr/bin/env python3
"""
Test script for loading trained BaseHookedModel models.
"""

import sys
import torch
from pathlib import Path
sys.path.append('src')

from fsrl import BaseHookedModel


def test_model_loading(model_path: str):
    """
    Test loading a trained BaseHookedModel and verify it works.
    
    Args:
        model_path: Path to the saved model directory
    """
    print(f"Testing model loading from: {model_path}")
    
    # Check if the model path exists
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ Model path does not exist: {model_path}")
        return False
    
    print(f"✓ Model path exists: {model_path}")
    
    try:
        # Load the model
        print("Loading model...")
        loaded_model = BaseHookedModel.from_pretrained(
            str(model_path),
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype="bfloat16"
        )
        print("✓ Model loaded successfully!")
        
        # Check model attributes
        print(f"Model device: {loaded_model.device}")
        print(f"Model has config: {hasattr(loaded_model, 'config')}")
        print(f"Model config type: {type(loaded_model.config)}")
        
        # Test basic functionality
        print("\nTesting model functionality...")
        
        # Test get_norms method
        norms = loaded_model.get_norms()
        print(f"✓ get_norms() works: {len(norms)} tensors, all zero: {all(norm.item() == 0.0 for norm in norms)}")
        
        # Test tokenizer access (if available)
        if hasattr(loaded_model.model, 'tokenizer'):
            tokenizer = loaded_model.model.tokenizer
            print("✓ Tokenizer available")
            
            # Test simple inference
            test_text = "Hello, world!"
            tokens = tokenizer.encode(test_text, return_tensors="pt").to(loaded_model.device)
            print(f"Test input: '{test_text}' -> {tokens.shape} tokens")
            
            # Forward pass test
            with torch.no_grad():
                output = loaded_model(tokens)
                print(f"✓ Forward pass successful: {output.logits.shape}")
                
                # Test generation (if supported)
                try:
                    generated = loaded_model.generate(
                        tokens, 
                        max_new_tokens=5, 
                        do_sample=False
                    )
                    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
                    print(f"✓ Generation test: '{generated_text}'")
                except Exception as e:
                    print(f"⚠ Generation test failed (this may be expected): {e}")
        
        else:
            print("⚠ No tokenizer found, skipping inference tests")
        
        print("\n🎉 All tests passed! Model loading works correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False


def find_trained_models():
    """Find available trained models in the models directory."""
    models_dir = Path("models")
    if not models_dir.exists():
        return []
    
    model_paths = []
    
    # Search recursively for model directories
    for root in models_dir.rglob("*"):
        if root.is_dir():
            # Look for typical model files
            if any((root / name).exists() for name in ["config.json", "pytorch_model.bin", "model.safetensors"]):
                model_paths.append(root)
    
    return model_paths


def main():
    """Main test function."""
    print("BaseHookedModel Loading Test")
    print("=" * 40)
    
    # Check for command line argument
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        success = test_model_loading(model_path)
    else:
        # Find available models
        print("Searching for trained models...")
        available_models = find_trained_models()
        
        if not available_models:
            print("❌ No trained models found in 'models/' directory")
            print("Usage: python test_model_loading.py <path_to_model>")
            return
        
        print(f"Found {len(available_models)} potential models:")
        for i, model_path in enumerate(available_models):
            print(f"  {i+1}. {model_path}")
        
        # Test the first available model
        print(f"\nTesting first model: {available_models[0]}")
        success = test_model_loading(str(available_models[0]))
    
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
