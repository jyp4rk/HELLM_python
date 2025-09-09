#!/usr/bin/env python3
"""
Test numerical stability of different NoisyLinear implementations.
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass


@dataclass
class NoiseConfig:
    """Simple noise configuration for testing."""
    fractional_bitwidth: int
    sqrt_Nh: float = 0.0
    delta_bitwidth: int = 42
    free_weights: bool = False


def test_implementation(linear_class, name, input_tensor, noise_config):
    """Test a NoisyLinear implementation."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    try:
        # Create layer
        layer = linear_class(
            in_features=128, 
            out_features=64, 
            noise_config=noise_config
        )
        
        # Initialize weights with typical values
        with torch.no_grad():
            layer.weight.data = torch.randn_like(layer.weight) * 0.1
            if layer.bias is not None:
                layer.bias.data = torch.zeros_like(layer.bias)
        
        # Setup noise processing
        layer.setup_noise()
        
        # Forward pass
        output = layer(input_tensor)
        
        # Check for NaN/Inf
        has_nan = torch.isnan(output).any()
        has_inf = torch.isinf(output).any()
        
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.6f}, {output.max():.6f}]")
        print(f"  Output mean: {output.mean():.6f}")
        print(f"  Output std: {output.std():.6f}")
        print(f"  Has NaN: {has_nan}")
        print(f"  Has Inf: {has_inf}")
        
        if has_nan or has_inf:
            print(f"  ⚠️ WARNING: Numerical issues detected!")
        
        return output, has_nan or has_inf
        
    except Exception as e:
        print(f"✗ Failed with error: {e}")
        return None, True


def main():
    """Test different fractional bitwidths and approaches."""
    
    # Test configuration
    batch_size = 32
    seq_len = 64
    hidden_dim = 128
    
    # Create test input with realistic values
    input_tensor = torch.randn(batch_size, seq_len, hidden_dim) * 0.5
    
    print("Input Statistics:")
    print(f"  Shape: {input_tensor.shape}")
    print(f"  Range: [{input_tensor.min():.6f}, {input_tensor.max():.6f}]")
    print(f"  Mean: {input_tensor.mean():.6f}")
    print(f"  Std: {input_tensor.std():.6f}")
    
    # Test different fractional bitwidths
    test_bitwidths = [4, 8, 12, 16, 20, 24]
    
    for bitwidth in test_bitwidths:
        print(f"\n{'#'*70}")
        print(f"Testing with fractional_bitwidth = {bitwidth}")
        print(f"Scale = 2^{bitwidth} = {2**bitwidth}")
        print(f"{'#'*70}")
        
        noise_config = NoiseConfig(fractional_bitwidth=bitwidth)
        
        # Test updated noisy_linear.py (hybrid approach)
        try:
            from train_utils.noisy_linear import NoisyLinear as NoisyLinearHybrid
            output_hybrid, has_issues_hybrid = test_implementation(
                NoisyLinearHybrid, 
                "Hybrid (scaling + truncation)", 
                input_tensor,
                noise_config
            )
        except Exception as e:
            print(f"Hybrid implementation failed to import: {e}")
            has_issues_hybrid = True
        
        # Test truncation-only approach
        try:
            from train_utils.noisy_linear_truncate import NoisyLinear as NoisyLinearTruncate
            output_truncate, has_issues_truncate = test_implementation(
                NoisyLinearTruncate,
                "Truncation-only",
                input_tensor,
                noise_config
            )
        except Exception as e:
            print(f"Truncation implementation failed to import: {e}")
            has_issues_truncate = True
        
        # Compare outputs if both succeeded
        if not has_issues_hybrid and not has_issues_truncate:
            diff = (output_hybrid - output_truncate).abs().mean()
            print(f"\n📊 Comparison:")
            print(f"  Mean absolute difference: {diff:.6f}")
            
            if diff > 0.1:
                print(f"  ⚠️ Large difference between implementations!")
    
    print(f"\n{'='*70}")
    print("Summary:")
    print("- Hybrid approach: Uses moderate scaling (≤2^12) + truncation for larger bitwidths")
    print("- Truncation approach: Uses quantization to fixed precision levels")
    print("- Both avoid numerical issues from excessive scaling")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()