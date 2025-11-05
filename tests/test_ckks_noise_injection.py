#!/usr/bin/env python3
"""
Test script for AttributeNoiseInjector implementation in modeling_llama_CKKS.py
"""

import torch
import sys
import os

# Add paths for imports
sys.path.append('/data/jypark/PrefixQuant10')

from src.training.modeling_llama_CKKS import LlamaForCausalLM, LlamaConfig
from src.utils.attribute_noise_injector import AttributeNoiseInjector, NoiseConfig, NoiseType

def test_noise_injection_setup():
    """Test that noise injection setup works correctly."""
    print("🧪 Testing Noise Injection Setup...")
    
    # Create a minimal config for testing
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=128,
    )
    
    # Initialize model
    model = LlamaForCausalLM(config)
    
    # Initialize R1 and R2 matrices for CKKS (normally done through rotation_utils)
    # R1 is model-level, R2 is layer-specific - they need to be nn.Linear layers to have .weight attribute
    model.R1 = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)
    model.R1.weight.data = torch.eye(config.hidden_size)
    model.R1.requires_grad_(False)
    
    # Initialize R2 for each attention layer
    for layer in model.model.layers:
        layer.self_attn.R2 = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)  
        layer.self_attn.R2.weight.data = torch.eye(config.hidden_size)
        layer.self_attn.R2.requires_grad_(False)
    
    print(f"✅ Model initialized with {config.num_hidden_layers} layers")
    
    # Test noise injection setup
    rmsnorm_config = NoiseConfig(
        noise_type=NoiseType.GAUSSIAN,
        std=0.01,
        enabled=True
    )
    
    softmax_config = NoiseConfig(
        noise_type=NoiseType.UNIFORM,
        low=-0.005,
        high=0.005,
        enabled=True
    )
    
    activation_config = NoiseConfig(
        noise_type=NoiseType.GAUSSIAN,
        std=0.02,
        enabled=True
    )
    
    # Setup noise injection
    injector = model.setup_noise_injection(
        rmsnorm_config=rmsnorm_config,
        softmax_config=softmax_config,
        activation_config=activation_config
    )
    
    print(f"✅ Noise injection setup complete")
    print(f"📊 Injector status: {injector.get_status()}")
    
    return model, injector

def test_forward_pass_with_noise():
    """Test that forward passes work with noise injection enabled."""
    print("\n🚀 Testing Forward Pass with Noise...")
    
    model, injector = test_noise_injection_setup()
    
    # Create sample input
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        try:
            output1 = model(input_ids)
            output2 = model(input_ids)
            
            print(f"✅ Forward passes completed successfully")
            print(f"📏 Output shape: {output1.logits.shape}")
            
            # Check that outputs differ due to noise
            diff = torch.abs(output1.logits - output2.logits).mean()
            print(f"🎲 Mean difference between passes: {diff:.6f}")
            
            if diff > 1e-6:
                print("✅ Noise injection working - outputs differ between runs")
            else:
                print("⚠️  Outputs are identical - noise may not be active")
                
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            return False
    
    return True

def test_noise_control():
    """Test enabling/disabling noise injection."""
    print("\n🔄 Testing Noise Control...")
    
    model, injector = test_noise_injection_setup()
    
    # Create sample input
    input_ids = torch.randint(0, 1000, (2, 16))
    
    model.eval()
    with torch.no_grad():
        # Get baseline with noise
        output_with_noise = model(input_ids)
        
        # Disable noise
        model.disable_all_noise()
        output_without_noise1 = model(input_ids)
        output_without_noise2 = model(input_ids)
        
        # Check that outputs are identical when noise is disabled
        diff_no_noise = torch.abs(output_without_noise1.logits - output_without_noise2.logits).mean()
        print(f"🔕 Difference without noise: {diff_no_noise:.8f}")
        
        if diff_no_noise < 1e-8:
            print("✅ Noise successfully disabled - deterministic outputs")
        else:
            print("⚠️  Outputs still differ when noise disabled")
        
        # Re-enable noise
        injector.enable_all_noise()
        output_noise_reenabled = model(input_ids)
        
        # Check that noise is working again
        diff_reenabled = torch.abs(output_noise_reenabled.logits - output_without_noise1.logits).mean()
        print(f"🔄 Difference after re-enabling: {diff_reenabled:.6f}")
        
        if diff_reenabled > 1e-6:
            print("✅ Noise successfully re-enabled")
        else:
            print("⚠️  Noise may not be working after re-enable")

def test_cleanup():
    """Test noise injection cleanup."""
    print("\n🧹 Testing Cleanup...")
    
    model, injector = test_noise_injection_setup()
    
    # Count modules with noise attributes before cleanup
    noise_attrs_before = 0
    for name, module in model.named_modules():
        if hasattr(module, 'inject_rmsnorm_noise') or hasattr(module, 'inject_softmax_noise') or hasattr(module, 'inject_activation_noise'):
            noise_attrs_before += 1
    
    print(f"📊 Modules with noise attributes before cleanup: {noise_attrs_before}")
    
    # Cleanup
    model.cleanup_noise()
    
    # Count modules with noise attributes after cleanup
    noise_attrs_after = 0
    for name, module in model.named_modules():
        if hasattr(module, 'inject_rmsnorm_noise') or hasattr(module, 'inject_softmax_noise') or hasattr(module, 'inject_activation_noise'):
            noise_attrs_after += 1
    
    print(f"🧹 Modules with noise attributes after cleanup: {noise_attrs_after}")
    
    if noise_attrs_after == 0:
        print("✅ Cleanup successful - all noise attributes removed")
    else:
        print(f"⚠️  {noise_attrs_after} modules still have noise attributes")
    
    # Check that model injector is reset
    if model.noise_injector is None:
        print("✅ Model noise injector properly reset to None")
    else:
        print("⚠️  Model noise injector not reset")

def main():
    """Run all tests."""
    print("🔬 Testing AttributeNoiseInjector in modeling_llama_CKKS.py")
    print("=" * 60)
    
    try:
        # Run all tests
        test_noise_injection_setup()
        
        if test_forward_pass_with_noise():
            test_noise_control()
            test_cleanup()
        
        print("\n✅ All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()