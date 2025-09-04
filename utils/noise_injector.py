"""
Unified Noise Injection System for LLaMA Models

This system provides a monkey-patching based approach to inject noise into
softmax, RMSNorm, and activation functions without modifying the original model architecture.

Key Design Principles:
1. Non-invasive: Uses monkey patching to avoid architectural changes
2. Configurable: Different noise types and parameters for each target function
3. Toggleable: Easy enable/disable without model reloading
4. Extensible: Easy to add new noise types and target functions
"""

import torch
import torch.nn.functional as F
import copy
import functools
import types
from typing import Dict, Any, Callable, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


class NoiseType(Enum):
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"


@dataclass
class NoiseConfig:
    """Configuration for noise injection"""
    noise_type: NoiseType = NoiseType.GAUSSIAN
    amplitude: float = 0.01
    enabled: bool = True

    # Gaussian noise parameters
    mean: float = 0.0
    std: float = None  # Will use amplitude if None

    # Uniform noise parameters
    low: float = None   # Will use -amplitude if None
    high: float = None  # Will use +amplitude if None
    
    # Laplacian noise parameters
    scale: float = None  # Will use amplitude if None
    
    # Custom noise function
    custom_noise_fn: Optional[Callable] = None
    
    # Conditional noise application
    apply_probability: float = 1.0  # Probability of applying noise (0.0-1.0)
    layer_name_filter: Optional[str] = None  # Apply only to layers matching pattern

    def __post_init__(self):
        """Set default values based on amplitude"""
        if self.std is None:
            self.std = self.amplitude
        if self.low is None:
            self.low = -self.amplitude
        if self.high is None:
            self.high = self.amplitude
        if self.scale is None:
            self.scale = self.amplitude


class NoiseGenerator:
    """Generates different types of noise"""

    @staticmethod
    def generate_noise(tensor: torch.Tensor, config: NoiseConfig) -> torch.Tensor:
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


class NoiseInjector:
    """
    Main class for injecting noise into model functions using monkey patching
    """

    def __init__(self):
        self.noise_configs: Dict[str, NoiseConfig] = {}
        self.original_functions: Dict[str, Callable] = {}
        self.patched_modules: Dict[str, Any] = {}
        self.active_patches: Dict[str, str] = {}  # target_name -> module_name

    def copy_func_with_new_globals(self, f: Callable, globals_dict: Dict = None) -> Callable:
        """Copy function with new global namespace (from existing monkeypatch.py)"""
        if globals_dict is None:
            globals_dict = f.__globals__

        g = types.FunctionType(
            f.__code__, globals_dict, name=f.__name__,
            argdefs=f.__defaults__, closure=f.__closure__
        )
        g = functools.update_wrapper(g, f)
        g.__module__ = f.__module__
        g.__kwdefaults__ = copy.copy(f.__kwdefaults__)
        return g

    def create_noise_wrapper(self, original_fn: Callable, config: NoiseConfig,
                           target_name: str) -> Callable:
        """Create a wrapper function that adds noise to the original function output"""

        @functools.wraps(original_fn)
        def noisy_wrapper(*args, **kwargs):
            # Call original function
            output = original_fn(*args, **kwargs)

            # Apply layer name filtering if specified
            if config.layer_name_filter:
                # Try to get layer name from various sources
                layer_name = None
                if hasattr(args[0], '__class__'):
                    layer_name = args[0].__class__.__name__
                elif len(args) > 0 and hasattr(args[0], 'name'):
                    layer_name = args[0].name

                if layer_name and config.layer_name_filter not in layer_name:
                    return output

            # Generate and add noise
            if config.enabled:
                noise = NoiseGenerator.generate_noise(output, config)
                output = output + noise

            return output

        return noisy_wrapper

    def patch_function_in_module(self, module: Any, function_name: str,
                                wrapper: Callable) -> None:
        """Patch a function directly in a module"""
        if hasattr(module, function_name):
            if f"{module.__name__}.{function_name}" not in self.original_functions:
                self.original_functions[f"{module.__name__}.{function_name}"] = getattr(module, function_name)
            setattr(module, function_name, wrapper)
            self.patched_modules[f"{module.__name__}.{function_name}"] = module

    def patch_method_function_call(self, module_class: Any, method_name: str,
                                  function_name: str, wrapper_fn: Callable) -> None:
        """
        Patch a function call within a method using the existing monkeypatch approach
        Based on add_wrapper_after_function_call_in_method
        """
        target_key = f"{module_class.__name__}.{method_name}.{function_name}"

        if target_key not in self.original_functions:
            original_method = getattr(module_class, method_name).__func__
            self.original_functions[target_key] = original_method

            method_globals = dict(original_method.__globals__)
            wrapper = wrapper_fn(method_globals[function_name])
            method_globals[function_name] = wrapper

            new_method = self.copy_func_with_new_globals(original_method, globals=method_globals)
            setattr(module_class, method_name, new_method.__get__(module_class))

            self.patched_modules[target_key] = module_class

    def configure_rmsnorm_noise(self, config: NoiseConfig) -> str:
        """Configure noise injection for RMSNorm layers"""
        target_name = "rmsnorm"
        self.noise_configs[target_name] = config
        return target_name

    def configure_softmax_noise(self, config: NoiseConfig) -> str:
        """Configure noise injection for softmax operations"""
        target_name = "softmax"
        self.noise_configs[target_name] = config
        return target_name

    def configure_activation_noise(self, activation_type: str, config: NoiseConfig) -> str:
        """Configure noise injection for activation functions"""
        target_name = f"activation_{activation_type}"
        self.noise_configs[target_name] = config
        return target_name

    def configure_custom_function_noise(self, function_name: str, config: NoiseConfig) -> str:
        """Configure noise injection for any custom function"""
        target_name = f"custom_{function_name}"
        self.noise_configs[target_name] = config
        return target_name

    def apply_rmsnorm_noise(self, model: Any) -> None:
        """Apply noise injection to all RMSNorm layers in the model"""
        if "rmsnorm" not in self.noise_configs:
            raise ValueError("RMSNorm noise not configured. Call configure_rmsnorm_noise first.")

        config = self.noise_configs["rmsnorm"]

        # Find and patch RMSNorm forward methods
        for name, module in model.named_modules():
            if "RMSNorm" in module.__class__.__name__:
                if hasattr(module, 'forward'):
                    wrapper = self.create_noise_wrapper(module.forward, config, "rmsnorm")
                    original_forward = module.forward
                    module.forward = wrapper
                    self.original_functions[f"{name}.forward"] = original_forward
                    self.active_patches[f"{name}"] = "rmsnorm"

    def apply_softmax_noise(self, model_class: Any) -> None:
        """Apply noise injection to softmax operations using method patching"""
        if "softmax" not in self.noise_configs:
            raise ValueError("Softmax noise not configured. Call configure_softmax_noise first.")

        config = self.noise_configs["softmax"]

        def softmax_wrapper(original_softmax):
            return self.create_noise_wrapper(original_softmax, config, "softmax")

        # Patch F.softmax calls in attention methods
        try:
            self.patch_method_function_call(
                model_class, "forward", "F.softmax",
                lambda fn: self.create_noise_wrapper(fn, config, "softmax")
            )
            self.active_patches["softmax"] = "method_patch"
        except Exception as e:
            print(f"Warning: Could not patch softmax in {model_class.__name__}: {e}")

    def apply_activation_noise(self, model_class: Any, activation_type: str = "silu") -> None:
        """Apply noise injection to activation functions"""
        target_name = f"activation_{activation_type}"
        if target_name not in self.noise_configs:
            raise ValueError(f"Activation noise for {activation_type} not configured.")

        config = self.noise_configs[target_name]

        # Map activation types to their function names
        activation_map = {
            "silu": "F.silu",
            "gelu": "F.gelu",
            "relu": "F.relu",
            "swish": "F.silu",  # SiLU is also called Swish
        }

        if activation_type not in activation_map:
            raise ValueError(f"Unsupported activation type: {activation_type}")

        function_name = activation_map[activation_type]

        try:
            self.patch_method_function_call(
                model_class, "forward", function_name,
                lambda fn: self.create_noise_wrapper(fn, config, target_name)
            )
            self.active_patches[target_name] = "method_patch"
        except Exception as e:
            print(f"Warning: Could not patch {activation_type} activation: {e}")

    def enable_noise(self, target_name: str) -> None:
        """Enable noise for a specific target"""
        if target_name in self.noise_configs:
            self.noise_configs[target_name].enabled = True
        else:
            raise ValueError(f"No noise configuration found for {target_name}")

    def disable_noise(self, target_name: str) -> None:
        """Disable noise for a specific target"""
        if target_name in self.noise_configs:
            self.noise_configs[target_name].enabled = False
        else:
            raise ValueError(f"No noise configuration found for {target_name}")

    def disable_all_noise(self) -> None:
        """Disable all noise injection"""
        for config in self.noise_configs.values():
            config.enabled = False

    def enable_all_noise(self) -> None:
        """Enable all configured noise injection"""
        for config in self.noise_configs.values():
            config.enabled = True

    def restore_original_functions(self) -> None:
        """Restore all patched functions to their original state"""
        for target_name, module in self.patched_modules.items():
            if target_name in self.original_functions:
                # Parse the target name to restore correctly
                parts = target_name.split('.')
                if len(parts) >= 2:
                    attr_name = parts[-1]
                    if hasattr(module, attr_name):
                        setattr(module, attr_name, self.original_functions[target_name])

        self.patched_modules.clear()
        self.active_patches.clear()

    def get_status(self) -> Dict[str, Any]:
        """Get current status of noise injection system"""
        return {
            "configured_targets": list(self.noise_configs.keys()),
            "active_patches": dict(self.active_patches),
            "enabled_targets": [name for name, config in self.noise_configs.items() if config.enabled],
            "disabled_targets": [name for name, config in self.noise_configs.items() if not config.enabled],
            "total_patches": len(self.active_patches)
        }


# Convenience function for easy setup
def setup_llama_noise_injection(
    model: Any,
    model_class: Any,
    rmsnorm_config: Optional[NoiseConfig] = None,
    softmax_config: Optional[NoiseConfig] = None,
    activation_config: Optional[NoiseConfig] = None,
    activation_type: str = "silu"
) -> NoiseInjector:
    """
    Convenience function to set up noise injection for a LLaMA model

    Args:
        model: The model instance
        model_class: The model class (for method patching)
        rmsnorm_config: Configuration for RMSNorm noise
        softmax_config: Configuration for softmax noise
        activation_config: Configuration for activation noise
        activation_type: Type of activation function ("silu", "gelu", "relu")

    Returns:
        NoiseInjector instance for further control
    """
    injector = NoiseInjector()

    # Configure noise types
    if rmsnorm_config:
        injector.configure_rmsnorm_noise(rmsnorm_config)
        injector.apply_rmsnorm_noise(model)

    if softmax_config:
        injector.configure_softmax_noise(softmax_config)
        injector.apply_softmax_noise(model_class)

    if activation_config:
        injector.configure_activation_noise(activation_type, activation_config)
        injector.apply_activation_noise(model_class, activation_type)

    return injector


# Example usage and testing functions
def test_noise_injection():
    """Test the noise injection system"""

    # Create noise configurations
    rmsnorm_config = NoiseConfig(
        noise_type=NoiseType.GAUSSIAN,
        amplitude=0.01,
        enabled=True
    )

    softmax_config = NoiseConfig(
        noise_type=NoiseType.UNIFORM,
        amplitude=0.005,
        enabled=True,
    )

    activation_config = NoiseConfig(
        noise_type=NoiseType.LAPLACIAN,
        amplitude=0.02,
        enabled=True,
        layer_name_filter="MLP"  # Only apply to MLP layers
    )

    print("Noise Injection System Test:")
    print(f"RMSNorm Config: {rmsnorm_config}")
    print(f"Softmax Config: {softmax_config}")
    print(f"Activation Config: {activation_config}")

    # Test noise generation
    test_tensor = torch.randn(2, 10, 512)

    for config_name, config in [("RMSNorm", rmsnorm_config), ("Softmax", softmax_config)]:
        noise = NoiseGenerator.generate_noise(test_tensor, config)
        print(f"\n{config_name} Noise Stats:")
        print(f"  Mean: {noise.mean():.6f}")
        print(f"  Std: {noise.std():.6f}")
        print(f"  Min: {noise.min():.6f}")
        print(f"  Max: {noise.max():.6f}")


if __name__ == "__main__":
    test_noise_injection()
