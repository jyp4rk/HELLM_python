"""
Test script to validate the attribute-based noise injection system.
This tests the approach similar to capture_pre_softmax pattern.
"""

import torch
import torch.nn as nn
from src.utils.attribute_noise_injector import (
    AttributeNoiseInjector, NoiseConfig, NoiseType,
    create_modified_rmsnorm_forward
)


class ModifiedRMSNorm(nn.Module):
    """
    RMSNorm with noise injection support using attribute checking.
    This demonstrates how to modify existing layers.
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        output = self.weight * hidden_states.to(input_dtype)

        # Noise injection point - uses attribute checking
        if hasattr(self, "inject_rmsnorm_noise") and self.inject_rmsnorm_noise:
            if hasattr(self, "rmsnorm_noise_config") and self.rmsnorm_noise_config.enabled:
                noise = self.noise_injector_fn(output, self.rmsnorm_noise_config)
                output = output + noise

        return output


class ModifiedAttention(nn.Module):
    """
    Simplified attention with noise injection support.
    """
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape

        # Get Q, K, V
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply softmax
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # Noise injection point for softmax
        if hasattr(self, "inject_softmax_noise") and self.inject_softmax_noise:
            if hasattr(self, "softmax_noise_config") and self.softmax_noise_config.enabled:
                noise = self.noise_injector_fn(attn_weights, self.softmax_noise_config)
                attn_weights = attn_weights + noise
                # Re-normalize after noise injection
                attn_weights = torch.softmax(attn_weights, dim=-1)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        return self.out_proj(attn_output)


class ModifiedMLP(nn.Module):
    """
    MLP with activation noise injection support.
    """
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)

        # Apply SiLU activation
        activated = nn.functional.silu(gate) * up

        # Noise injection point for activation
        if hasattr(self, "inject_activation_noise") and self.inject_activation_noise:
            if hasattr(self, "activation_noise_config") and self.activation_noise_config.enabled:
                noise = self.noise_injector_fn(activated, self.activation_noise_config)
                activated = activated + noise

        return self.down_proj(activated)


class TestModel(nn.Module):
    """Test model with modified layers that support noise injection"""
    def __init__(self, hidden_size=128, intermediate_size=512):
        super().__init__()
        self.norm1 = ModifiedRMSNorm(hidden_size)
        self.attention = ModifiedAttention(hidden_size)
        self.norm2 = ModifiedRMSNorm(hidden_size)
        self.mlp = ModifiedMLP(hidden_size, intermediate_size)

    def forward(self, x):
        # Pre-attention norm
        normed = self.norm1(x)

        # Self-attention with residual
        attn_out = self.attention(normed)
        x = x + attn_out

        # Pre-MLP norm
        normed = self.norm2(x)

        # MLP with residual
        mlp_out = self.mlp(normed)
        x = x + mlp_out

        return x


def test_attribute_injection():
    """Test that attributes are correctly added and removed"""
    print("=== Testing Attribute Injection ===")

    model = TestModel()
    injector = AttributeNoiseInjector()

    # Initial state - no noise attributes
    norm_modules = [m for n, m in model.named_modules() if isinstance(m, ModifiedRMSNorm)]
    initial_attrs = sum(hasattr(m, "inject_rmsnorm_noise") for m in norm_modules)
    print(f"Initial noise attributes: {initial_attrs}")

    # Enable RMSNorm noise
    config = NoiseConfig(noise_type=NoiseType.GAUSSIAN, std=0.01)
    injector.enable_rmsnorm_noise(model, config)

    # Check attributes were added
    after_enable_attrs = sum(hasattr(m, "inject_rmsnorm_noise") for m in norm_modules)
    print(f"After enable - noise attributes: {after_enable_attrs}")
    print(f"Expected: {len(norm_modules)}, Got: {after_enable_attrs}")

    # Verify all required attributes exist
    for i, module in enumerate(norm_modules):
        has_flag = hasattr(module, "inject_rmsnorm_noise")
        has_config = hasattr(module, "rmsnorm_noise_config")
        has_fn = hasattr(module, "noise_injector_fn")
        print(f"  Norm {i}: flag={has_flag}, config={has_config}, fn={has_fn}")

    # Cleanup
    injector.cleanup()

    # Check attributes were removed
    after_cleanup_attrs = sum(hasattr(m, "inject_rmsnorm_noise") for m in norm_modules)
    print(f"After cleanup - noise attributes: {after_cleanup_attrs}")

    return after_enable_attrs == len(norm_modules) and after_cleanup_attrs == 0


def test_noise_functionality():
    """Test that noise is actually applied when attributes are set"""
    print("\n=== Testing Noise Functionality ===")

    model = TestModel()
    injector = AttributeNoiseInjector()

    # Test input
    batch_size, seq_len, hidden_size = 2, 10, 128
    test_input = torch.randn(batch_size, seq_len, hidden_size)

    # Get clean output
    with torch.no_grad():
        clean_output = model(test_input)

    print(f"Clean output: mean={clean_output.mean():.6f}, std={clean_output.std():.6f}")

    # Enable noise injection
    rmsnorm_config = NoiseConfig(noise_type=NoiseType.GAUSSIAN, std=0.05)
    softmax_config = NoiseConfig(noise_type=NoiseType.UNIFORM, low=-0.02, high=0.02)
    activation_config = NoiseConfig(noise_type=NoiseType.GAUSSIAN, std=0.03)

    injector.enable_rmsnorm_noise(model, rmsnorm_config)
    injector.enable_softmax_noise(model, softmax_config)
    injector.enable_activation_noise(model, activation_config)

    print(f"Injector status: {injector.get_status()}")

    # Get noisy output
    with torch.no_grad():
        noisy_output = model(test_input)

    print(f"Noisy output: mean={noisy_output.mean():.6f}, std={noisy_output.std():.6f}")

    # Check difference
    diff = (noisy_output - clean_output).abs()
    print(f"Absolute difference: mean={diff.mean():.6f}, max={diff.max():.6f}")

    outputs_different = not torch.allclose(clean_output, noisy_output, atol=1e-6)
    print(f"Outputs are different: {outputs_different}")

    # Test enable/disable
    injector.disable_all_noise()
    with torch.no_grad():
        disabled_output = model(test_input)

    disabled_diff = (disabled_output - clean_output).abs()
    print(f"After disable - difference: mean={disabled_diff.mean():.6f}, max={disabled_diff.max():.6f}")

    noise_disabled = torch.allclose(clean_output, disabled_output, atol=1e-6)
    print(f"Noise successfully disabled: {noise_disabled}")

    # Cleanup
    injector.cleanup()

    return outputs_different and noise_disabled


def test_individual_noise_types():
    """Test each noise type individually"""
    print("\n=== Testing Individual Noise Types ===")

    model = TestModel()
    test_input = torch.randn(2, 5, 128)

    results = {}

    # Test RMSNorm noise only
    injector = AttributeNoiseInjector()
    config = NoiseConfig(noise_type=NoiseType.GAUSSIAN, std=0.02)
    injector.enable_rmsnorm_noise(model, config)

    with torch.no_grad():
        clean_output = model(test_input)

        # Temporarily disable to get clean baseline
        injector.disable_all_noise()
        baseline_output = model(test_input)

        # Re-enable for noisy output
        injector.enable_all_noise()
        rmsnorm_noisy_output = model(test_input)

    rmsnorm_diff = (rmsnorm_noisy_output - baseline_output).abs().mean().item()
    results['rmsnorm'] = rmsnorm_diff
    print(f"RMSNorm noise difference: {rmsnorm_diff:.6f}")

    injector.cleanup()

    # Test Softmax noise only
    injector = AttributeNoiseInjector()
    config = NoiseConfig(noise_type=NoiseType.UNIFORM, low=-0.01, high=0.01)
    injector.enable_softmax_noise(model, config)

    with torch.no_grad():
        injector.disable_all_noise()
        baseline_output = model(test_input)

        injector.enable_all_noise()
        softmax_noisy_output = model(test_input)

    softmax_diff = (softmax_noisy_output - baseline_output).abs().mean().item()
    results['softmax'] = softmax_diff
    print(f"Softmax noise difference: {softmax_diff:.6f}")

    injector.cleanup()

    # Test Activation noise only
    injector = AttributeNoiseInjector()
    config = NoiseConfig(noise_type=NoiseType.GAUSSIAN, std=0.015)
    injector.enable_activation_noise(model, config)

    with torch.no_grad():
        injector.disable_all_noise()
        baseline_output = model(test_input)

        injector.enable_all_noise()
        activation_noisy_output = model(test_input)

    activation_diff = (activation_noisy_output - baseline_output).abs().mean().item()
    results['activation'] = activation_diff
    print(f"Activation noise difference: {activation_diff:.6f}")

    injector.cleanup()

    # All should be greater than zero (indicating noise was applied)
    all_working = all(diff > 1e-6 for diff in results.values())
    print(f"All noise types working: {all_working}")

    return all_working, results


def run_all_tests():
    """Run all validation tests"""
    print("Attribute-Based Noise Injection Validation")
    print("=" * 50)

    results = {}

    try:
        results['attribute_injection'] = test_attribute_injection()
        results['noise_functionality'] = test_noise_functionality()
        results['individual_types'], noise_results = test_individual_noise_types()

        print("\n" + "=" * 50)
        print("TEST RESULTS:")
        print(f"✅ Attribute injection/cleanup: {results['attribute_injection']}")
        print(f"✅ Noise functionality: {results['noise_functionality']}")
        print(f"✅ Individual noise types: {results['individual_types']}")

        if results['individual_types']:
            print("\nNoise type effectiveness:")
            for noise_type, diff in noise_results.items():
                print(f"  {noise_type}: {diff:.6f}")

        all_passed = all(results.values())
        print(f"\n{'✅ ALL TESTS PASSED!' if all_passed else '❌ SOME TESTS FAILED!'}")

        return all_passed

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
