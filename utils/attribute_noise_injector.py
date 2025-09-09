"""
Attribute-Based Noise Injection System
Using the same pattern as capture_pre_softmax for reliable integration.

This approach adds attributes directly to modules, allowing the forward methods
to check for these attributes and apply noise accordingly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum


class NoiseType(Enum):
    """Supported noise distribution types"""
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"

@dataclass
class LinearLayerNoiseConfig:
    sqrt_Nh: float
    delta_bitwidth: int
    fractional_bitwidth: int
    injector: Callable[[torch.Tensor, float], torch.Tensor]

@dataclass
class NoiseConfig:
    """Configuration for noise injection"""
    noise_type: NoiseType = NoiseType.GAUSSIAN

    # Gaussian noise parameters
    mean: float = 0.0
    std: float = None  # Will use amplitude if None

    # Uniform noise parameters
    low: float = None   # Will use -amplitude if None
    high: float = None  # Will use +amplitude if None

    injector: Callable[[torch.Tensor, float], torch.Tensor] = None

def noise_injector(tensor: torch.Tensor, amplitude: float) -> torch.Tensor:
    """Generate noise matching tensor shape according to config"""
    std = amplitude
    return torch.normal(0, std, size=tensor.shape,
                          device=tensor.device, dtype=tensor.dtype)


class AttributeNoiseInjector:

    def __init__(self):
        self.active_modules: Dict[str, nn.Module] = {}
        self.noise_configs: Dict[str, NoiseConfig] = {}
        self.original_forwards: Dict[str, Callable] = {}

    def generate_noise(self, tensor: torch.Tensor, config: NoiseConfig) -> torch.Tensor:
        """Generate noise matching tensor shape according to config"""
        if not config.enabled:
            return torch.zeros_like(tensor)

        if config.noise_type == NoiseType.GAUSSIAN:
            return torch.normal(config.mean, config.std, size=tensor.shape,
                              device=tensor.device, dtype=tensor.dtype)
        elif config.noise_type == NoiseType.UNIFORM:
            return torch.empty(tensor.shape, device=tensor.device, dtype=tensor.dtype).uniform_(
                config.low, config.high)
        else:
            raise ValueError(f"Unsupported noise type: {config.noise_type}")

    def enable_rmsnorm_noise(self, model: nn.Module, config: NoiseConfig) -> None:
        """
        Enable noise injection for RMSNorm layers by adding attributes.

        The modified forward method should check:
        if hasattr(self, "inject_rmsnorm_noise") and self.inject_rmsnorm_noise:
            noise = self.noise_injector_fn(output, self.rmsnorm_noise_config)
            output = output + noise
        """
        count = 0
        for name, module in model.named_modules():
            if "RMSNorm" in module.__class__.__name__ or isinstance(module, nn.RMSNorm):
                # Add noise injection attributes
                module.inject_rmsnorm_noise = True
                module.rmsnorm_noise_config = config
                module.noise_injector_fn = self.generate_noise

                # Store reference for cleanup
                self.active_modules[f"rmsnorm_{name}"] = module
                count += 1

        self.noise_configs["rmsnorm"] = config
        print(f"Enabled RMSNorm noise injection on {count} modules")

    def enable_softmax_noise(self, model: nn.Module, config: NoiseConfig) -> None:
        """
        Enable noise injection for attention softmax.

        The attention forward method should check:
        if hasattr(self, "inject_softmax_noise") and self.inject_softmax_noise:
            noise = self.noise_injector_fn(attn_weights, self.softmax_noise_config)
            attn_weights = attn_weights + noise
        """
        count = 0
        for name, module in model.named_modules():
            if "Attention" in module.__class__.__name__ or hasattr(module, 'attention_dropout'):
                # Add noise injection attributes
                module.inject_softmax_noise = True
                module.softmax_noise_config = config
                module.noise_injector_fn = self.generate_noise

                # Store reference for cleanup
                self.active_modules[f"softmax_{name}"] = module
                count += 1

        self.noise_configs["softmax"] = config
        print(f"Enabled softmax noise injection on {count} modules")

    def enable_activation_noise(self, model: nn.Module, config: NoiseConfig,
                               activation_type: str = "silu") -> None:
        """
        Enable noise injection for activation functions in MLP layers.

        The MLP forward method should check:
        if hasattr(self, "inject_activation_noise") and self.inject_activation_noise:
            noise = self.noise_injector_fn(activated, self.activation_noise_config)
            activated = activated + noise
        """
        count = 0
        for name, module in model.named_modules():
            if "MLP" in module.__class__.__name__ or "FeedForward" in module.__class__.__name__:
                # Add noise injection attributes
                module.inject_activation_noise = True
                module.activation_noise_config = config
                module.activation_noise_type = activation_type
                module.noise_injector_fn = self.generate_noise

                # Store reference for cleanup
                self.active_modules[f"activation_{name}"] = module
                count += 1

        self.noise_configs[f"activation_{activation_type}"] = config
        print(f"Enabled {activation_type} activation noise injection on {count} modules")

    def enable_custom_noise(self, modules: List[nn.Module], config: NoiseConfig,
                           attribute_name: str = "inject_custom_noise") -> None:
        """
        Enable noise injection on custom modules with custom attribute name.

        The target forward method should check:
        if hasattr(self, attribute_name) and getattr(self, attribute_name):
            noise = self.noise_injector_fn(output, self.custom_noise_config)
            output = output + noise
        """
        count = 0
        for i, module in enumerate(modules):
            # Add noise injection attributes
            setattr(module, attribute_name, True)
            module.custom_noise_config = config
            module.noise_injector_fn = self.generate_noise

            # Store reference for cleanup
            self.active_modules[f"custom_{i}"] = module
            count += 1

        self.noise_configs[attribute_name] = config
        print(f"Enabled custom noise injection on {count} modules with attribute '{attribute_name}'")

    def disable_noise(self, noise_type: str) -> None:
        """Disable specific noise type"""
        if noise_type in self.noise_configs:
            self.noise_configs[noise_type].enabled = False

            # Update all modules with this noise type
            for module_key, module in self.active_modules.items():
                if noise_type in module_key:
                    if hasattr(module, f"{noise_type}_noise_config"):
                        getattr(module, f"{noise_type}_noise_config").enabled = False
                    elif hasattr(module, "rmsnorm_noise_config") and noise_type == "rmsnorm":
                        module.rmsnorm_noise_config.enabled = False
                    elif hasattr(module, "softmax_noise_config") and noise_type == "softmax":
                        module.softmax_noise_config.enabled = False
                    elif hasattr(module, "activation_noise_config") and "activation" in noise_type:
                        module.activation_noise_config.enabled = False

    def enable_noise(self, noise_type: str) -> None:
        """Enable specific noise type"""
        if noise_type in self.noise_configs:
            self.noise_configs[noise_type].enabled = True

            # Update all modules with this noise type
            for module_key, module in self.active_modules.items():
                if noise_type in module_key:
                    if hasattr(module, f"{noise_type}_noise_config"):
                        getattr(module, f"{noise_type}_noise_config").enabled = True
                    elif hasattr(module, "rmsnorm_noise_config") and noise_type == "rmsnorm":
                        module.rmsnorm_noise_config.enabled = True
                    elif hasattr(module, "softmax_noise_config") and noise_type == "softmax":
                        module.softmax_noise_config.enabled = True
                    elif hasattr(module, "activation_noise_config") and "activation" in noise_type:
                        module.activation_noise_config.enabled = True

    def disable_all_noise(self) -> None:
        """Disable all noise injection by setting enabled=False on all configs"""
        for config in self.noise_configs.values():
            config.enabled = False

        # Update all module configs
        for module in self.active_modules.values():
            for attr_name in dir(module):
                if attr_name.endswith('_noise_config'):
                    config = getattr(module, attr_name)
                    if hasattr(config, 'enabled'):
                        config.enabled = False

    def enable_all_noise(self) -> None:
        """Enable all noise injection"""
        for config in self.noise_configs.values():
            config.enabled = True

        # Update all module configs
        for module in self.active_modules.values():
            for attr_name in dir(module):
                if attr_name.endswith('_noise_config'):
                    config = getattr(module, attr_name)
                    if hasattr(config, 'enabled'):
                        config.enabled = True

    def cleanup(self) -> None:
        """
        Remove all noise injection attributes from modules.
        Similar to how capture_pre_softmax is cleaned up.
        """
        for module in self.active_modules.values():
            # Remove noise injection flags
            for attr in ['inject_rmsnorm_noise', 'inject_softmax_noise',
                        'inject_activation_noise', 'inject_custom_noise']:
                if hasattr(module, attr):
                    delattr(module, attr)

            # Remove noise configs
            for attr in ['rmsnorm_noise_config', 'softmax_noise_config',
                        'activation_noise_config', 'custom_noise_config']:
                if hasattr(module, attr):
                    delattr(module, attr)

            # Remove noise function
            if hasattr(module, 'noise_injector_fn'):
                delattr(module, 'noise_injector_fn')

            # Remove activation type
            if hasattr(module, 'activation_noise_type'):
                delattr(module, 'activation_noise_type')

        # Clear our references
        self.active_modules.clear()
        self.noise_configs.clear()
        print("Cleaned up all noise injection attributes")

    def get_status(self) -> Dict[str, Any]:
        """Get current status of noise injection"""
        return {
            "active_modules": len(self.active_modules),
            "configured_noise_types": list(self.noise_configs.keys()),
            "enabled_noise_types": [name for name, config in self.noise_configs.items() if config.enabled],
            "disabled_noise_types": [name for name, config in self.noise_configs.items() if not config.enabled],
            "module_details": {key: type(module).__name__ for key, module in self.active_modules.items()}
        }


# Convenience functions for common usage patterns
def setup_llama_attribute_noise(
    model: nn.Module,
    rmsnorm_noise: Optional[NoiseConfig] = None,
    softmax_noise: Optional[NoiseConfig] = None,
    activation_noise: Optional[NoiseConfig] = None,
    activation_type: str = "silu"
) -> AttributeNoiseInjector:
    """
    Set up attribute-based noise injection for LLaMA model.

    Note: This requires that the model's forward methods have been modified
    to check for the noise injection attributes.

    Args:
        model: The LLaMA model
        rmsnorm_noise: Configuration for RMSNorm noise
        softmax_noise: Configuration for attention softmax noise
        activation_noise: Configuration for activation function noise
        activation_type: Type of activation function

    Returns:
        AttributeNoiseInjector instance
    """
    injector = AttributeNoiseInjector()

    if rmsnorm_noise:
        injector.enable_rmsnorm_noise(model, rmsnorm_noise)

    if softmax_noise:
        injector.enable_softmax_noise(model, softmax_noise)

    if activation_noise:
        injector.enable_activation_noise(model, activation_noise, activation_type)

    return injector


def create_modified_rmsnorm_forward():
    """
    Example of how to modify RMSNorm forward method to support noise injection.
    This would replace the original forward method.
    """
    def modified_forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        output = self.weight * hidden_states.to(input_dtype)

        # Noise injection point
        if hasattr(self, "inject_rmsnorm_noise") and self.inject_rmsnorm_noise:
            if hasattr(self, "rmsnorm_noise_config") and self.rmsnorm_noise_config.enabled:
                noise = self.noise_injector_fn(output, self.rmsnorm_noise_config)
                output = output + noise

        return output

    return modified_forward


# Usage example and testing
if __name__ == "__main__":
    print("Attribute-Based Noise Injection System")
    print("=" * 50)

    # Create a simple model for testing
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.RMSNorm(64)
            self.linear = nn.Linear(64, 32)

        def forward(self, x):
            x = self.norm(x)
            return self.linear(x)

    model = TestModel()
    injector = AttributeNoiseInjector()

    # Test configuration
    config = NoiseConfig(
        noise_type=NoiseType.GAUSSIAN,
        amplitude=0.01,
        enabled=True
    )

    # Enable noise injection
    injector.enable_rmsnorm_noise(model, config)

    print("Status after enabling RMSNorm noise:")
    print(injector.get_status())

    # Check that attributes were added
    for name, module in model.named_modules():
        if hasattr(module, "inject_rmsnorm_noise"):
            print(f"Module {name} has noise injection enabled: {module.inject_rmsnorm_noise}")
            print(f"  Config: {module.rmsnorm_noise_config}")

    # Cleanup
    injector.cleanup()

    print("\nAfter cleanup:")
    print(injector.get_status())

    # Verify attributes were removed
    noise_attrs_found = 0
    for name, module in model.named_modules():
        for attr in ['inject_rmsnorm_noise', 'rmsnorm_noise_config', 'noise_injector_fn']:
            if hasattr(module, attr):
                noise_attrs_found += 1

    print(f"Noise attributes remaining after cleanup: {noise_attrs_found}")
    print("✅ Test completed successfully!" if noise_attrs_found == 0 else "❌ Cleanup failed")
