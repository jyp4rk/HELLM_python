# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Noise injection utilities for SpinQuant models.
Provides noise injection capabilities for linear layers to simulate noisy execution environments.
"""

import math
import torch
import torch.nn as nn
from typing import Callable, Dict, List, Optional, Tuple, Union
from transformers import LlamaForCausalLM
import os
import logging

import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_utils.quant_linear import QuantizeLinear

logger = logging.getLogger(__name__)


class NoiseGenerator:
    """Generates different types of noise for injection into linear layers."""
    
    @staticmethod
    def gaussian(tensor: torch.Tensor, scale: float) -> torch.Tensor:
        """Generate Gaussian noise with given scale."""
        return torch.randn_like(tensor, device=tensor.device, dtype=tensor.dtype) * scale
    
    @staticmethod
    def uniform(tensor: torch.Tensor, scale: float) -> torch.Tensor:
        """Generate uniform noise in [-scale, scale]."""
        return (torch.rand_like(tensor, device=tensor.device, dtype=tensor.dtype) * 2 - 1) * scale
    
    @staticmethod
    def laplace(tensor: torch.Tensor, scale: float) -> torch.Tensor:
        """Generate Laplace noise with given scale."""
        uniform_noise = torch.rand_like(tensor, device=tensor.device, dtype=tensor.dtype)
        # Convert uniform to Laplace using inverse transform sampling
        return -scale * torch.sign(uniform_noise - 0.5) * torch.log(1 - 2 * torch.abs(uniform_noise - 0.5))


class NoisyQuantizeLinear(QuantizeLinear):
    """
    Extended QuantizeLinear with noise injection capabilities.
    
    Maintains full compatibility with existing quantization and rotation framework
    while adding controlled noise injection for robustness testing.
    """
    
    NOISE_GENERATORS = {
        'gaussian': NoiseGenerator.gaussian,
        'uniform': NoiseGenerator.uniform,
        'laplace': NoiseGenerator.laplace,
    }
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        noise_scale: float = 0.0,
        noise_type: str = 'gaussian',
        noise_enabled: bool = True,
        **kwargs
    ):
        """
        Initialize NoisyQuantizeLinear layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension  
            bias: Whether to use bias
            noise_scale: Scale of noise to inject (0.0 = no noise)
            noise_type: Type of noise ('gaussian', 'uniform', 'laplace')
            noise_enabled: Whether noise injection is enabled
            **kwargs: Additional arguments passed to QuantizeLinear
        """
        super().__init__(in_features, out_features, bias, **kwargs)
        
        self.noise_scale = noise_scale
        self.noise_type = noise_type
        self.noise_enabled = noise_enabled and noise_scale > 0.0
        
        if self.noise_type not in self.NOISE_GENERATORS:
            raise ValueError(f"Unsupported noise type: {noise_type}. "
                           f"Supported types: {list(self.NOISE_GENERATORS.keys())}")
    
    def _inject_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Inject noise into tensor if noise is enabled."""
        if not self.noise_enabled or self.noise_scale <= 0.0:
            return tensor
        
        noise_fn = self.NOISE_GENERATORS[self.noise_type]
        noise = noise_fn(tensor, self.noise_scale)
        return tensor + noise
    
    def forward(
        self,
        input: torch.Tensor,
        R1: Optional[torch.Tensor] = None,
        R2: Optional[torch.Tensor] = None,
        transpose: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with noise injection.
        
        Noise is injected after linear transformation but before quantization
        to simulate hardware noise in the computation.
        """
        # Get weight (with rotations if provided)
        if R1 is not None:
            dtype = self.weight.dtype
            if not transpose:
                weight = (self.weight.to(torch.float64) @ R1.to(torch.float64)).to(dtype)
            else:
                weight = (R1.T.to(torch.float64) @ self.weight.to(torch.float64)).to(dtype)
            
            if R2 is not None:
                # Each head dim = 128 for Llama model
                had_dim = R2.shape[0]
                dtype = weight.dtype
                if transpose:
                    W_ = weight
                    init_shape = W_.shape
                    temp = W_.reshape(-1, init_shape[-1] // had_dim, had_dim)
                    temp = temp.to(torch.float64) @ R2.to(torch.float64)
                    weight = temp.reshape(init_shape)
                else:
                    W_ = weight.t()
                    transposed_shape = W_.shape
                    temp = W_.reshape(-1, transposed_shape[-1] // had_dim, had_dim)
                    temp = temp.to(torch.float64) @ R2.to(torch.float64)
                    weight = temp.reshape(transposed_shape).t()
                weight = weight.to(dtype)
        else:
            weight = self.weight
        
        # Apply linear transformation
        output = nn.functional.linear(input, weight, self.bias)
        
        # Inject noise after linear transformation
        output = self._inject_noise(output)
        
        # Apply quantization if available
        if hasattr(self, "quantizer"):
            dtype = output.dtype
            self.quantizer.find_params(weight.data)
            # Note: We quantize the weight, not the noisy output
            quantized_weight = self.quantizer.quantize(weight).to(dtype)
            # Recompute with quantized weight and inject noise again
            output = nn.functional.linear(input, quantized_weight, self.bias)
            output = self._inject_noise(output)
        
        return output
    
    def set_noise_scale(self, scale: float):
        """Dynamically adjust noise scale."""
        self.noise_scale = scale
        self.noise_enabled = scale > 0.0
    
    def set_noise_type(self, noise_type: str):
        """Dynamically change noise type."""
        if noise_type not in self.NOISE_GENERATORS:
            raise ValueError(f"Unsupported noise type: {noise_type}")
        self.noise_type = noise_type
    
    def disable_noise(self):
        """Disable noise injection."""
        self.noise_enabled = False
    
    def enable_noise(self):
        """Enable noise injection if noise_scale > 0."""
        self.noise_enabled = self.noise_scale > 0.0


# Layer filter functions for selective noise injection
def attention_layer_filter(name: str, module: nn.Module) -> bool:
    """Filter for attention projection layers."""
    return any(proj in name for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj'])


def mlp_layer_filter(name: str, module: nn.Module) -> bool:
    """Filter for MLP layers."""
    return any(proj in name for proj in ['gate_proj', 'up_proj', 'down_proj'])


def all_linear_filter(name: str, module: nn.Module) -> bool:
    """Filter for all linear layers."""
    return isinstance(module, (nn.Linear, QuantizeLinear))


def layer_range_filter(start_layer: int, end_layer: int) -> Callable[[str, nn.Module], bool]:
    """Create filter for specific layer range."""
    def filter_fn(name: str, module: nn.Module) -> bool:
        try:
            # Extract layer number from path like 'model.layers.5.self_attn.q_proj'
            parts = name.split('.')
            if 'layers' in parts:
                layer_idx = parts.index('layers')
                if layer_idx + 1 < len(parts):
                    layer_num = int(parts[layer_idx + 1])
                    return start_layer <= layer_num <= end_layer
        except (ValueError, IndexError):
            pass
        return False
    return filter_fn


def combine_filters(*filter_funcs: Callable[[str, nn.Module], bool]) -> Callable[[str, nn.Module], bool]:
    """Combine multiple filter functions using OR logic."""
    def combined_filter(name: str, module: nn.Module) -> bool:
        return any(func(name, module) for func in filter_funcs)
    return combined_filter


def convert_to_noisy_layers(
    model: nn.Module,
    layer_filter: Callable[[str, nn.Module], bool],
    noise_scale: float = 0.01,
    noise_type: str = 'gaussian'
) -> nn.Module:
    """
    Convert QuantizeLinear layers in model to NoisyQuantizeLinear based on filter.
    
    Args:
        model: The model to modify
        layer_filter: Function to determine which layers to make noisy
        noise_scale: Scale of noise to inject
        noise_type: Type of noise to inject
    
    Returns:
        Modified model with noisy layers
    """
    def replace_layer(module: nn.Module, path: str = ""):
        for name, child in module.named_children():
            current_path = f"{path}.{name}" if path else name
            
            if layer_filter(current_path, child) and isinstance(child, QuantizeLinear):
                # Create noisy version
                noisy_layer = NoisyQuantizeLinear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    noise_scale=noise_scale,
                    noise_type=noise_type
                )
                
                # Copy weights and bias
                noisy_layer.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    noisy_layer.bias.data = child.bias.data.clone()
                
                # Copy quantizer if present
                if hasattr(child, 'quantizer'):
                    noisy_layer.quantizer = child.quantizer
                
                # Move to same device and dtype
                noisy_layer = noisy_layer.to(
                    device=child.weight.device,
                    dtype=child.weight.dtype
                )
                
                # Replace the layer
                setattr(module, name, noisy_layer)
                logger.info(f"Converted {current_path} to NoisyQuantizeLinear "
                           f"(noise_scale={noise_scale}, noise_type={noise_type})")
            
            # Recursively process child modules
            replace_layer(child, current_path)
    
    replace_layer(model)
    return model


def prepare_noisy_spinquant_model(
    model_name_or_path: str,
    filter_configs: Union[
        Tuple[Callable, float],
        List[Tuple[Callable, float]],
        Tuple[Callable, float, str],
        List[Tuple[Callable, float, str]]
    ],
    cache_dir: str = "model_cache",
    force_local: bool = False,
    **model_kwargs
) -> LlamaForCausalLM:
    """
    Load SpinQuant model and convert specified layers to noisy versions.
    
    Args:
        model_name_or_path: Path or name of model to load
        filter_configs: Layer filter configurations
            - Single: (filter_func, noise_scale) or (filter_func, noise_scale, noise_type)
            - Multiple: List of above tuples
        cache_dir: Directory for model cache
        force_local: Whether to force loading from local cache
        **model_kwargs: Additional arguments for model loading
    
    Returns:
        Model with noisy layers applied
    """
    try:
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token and not force_local:
            logger.warning("HF_TOKEN not found, using force_local=True")
            force_local = True

        os.makedirs(cache_dir, exist_ok=True)

        # Default model loading arguments
        default_kwargs = {
            'torch_dtype': torch.float16,
            'device_map': 'auto',
            'cache_dir': cache_dir,
            'local_files_only': force_local
        }
        
        if not force_local:
            default_kwargs['token'] = hf_token
        
        # Merge with user-provided kwargs
        default_kwargs.update(model_kwargs)

        # Load model
        model = LlamaForCausalLM.from_pretrained(model_name_or_path, **default_kwargs)
        
        # Normalize filter configs to list of tuples
        if not isinstance(filter_configs, list):
            filter_configs = [filter_configs]
        
        # Apply noise to each filter configuration
        for config in filter_configs:
            if len(config) == 2:
                filter_func, noise_scale = config
                noise_type = 'gaussian'
            elif len(config) == 3:
                filter_func, noise_scale, noise_type = config
            else:
                raise ValueError(f"Invalid filter config: {config}. "
                               f"Expected (filter, scale) or (filter, scale, type)")
            
            model = convert_to_noisy_layers(
                model,
                layer_filter=filter_func,
                noise_scale=noise_scale,
                noise_type=noise_type
            )
        
        logger.info(f"Loaded noisy SpinQuant model from {model_name_or_path}")
        return model
        
    except Exception as e:
        logger.error(f"Error preparing noisy SpinQuant model: {str(e)}")
        raise


def get_noise_statistics(model: nn.Module) -> Dict[str, Dict]:
    """
    Get statistics about noise injection in the model.
    
    Returns:
        Dictionary with noise statistics for each noisy layer
    """
    stats = {}
    
    def collect_stats(module: nn.Module, path: str = ""):
        for name, child in module.named_children():
            current_path = f"{path}.{name}" if path else name
            
            if isinstance(child, NoisyQuantizeLinear):
                stats[current_path] = {
                    'noise_scale': child.noise_scale,
                    'noise_type': child.noise_type,
                    'noise_enabled': child.noise_enabled,
                    'in_features': child.in_features,
                    'out_features': child.out_features,
                }
            
            collect_stats(child, current_path)
    
    collect_stats(model)
    return stats


def disable_all_noise(model: nn.Module):
    """Disable noise injection in all noisy layers."""
    def disable_noise(module: nn.Module):
        for child in module.children():
            if isinstance(child, NoisyQuantizeLinear):
                child.disable_noise()
            disable_noise(child)
    
    disable_noise(model)


def enable_all_noise(model: nn.Module):
    """Enable noise injection in all noisy layers."""
    def enable_noise(module: nn.Module):
        for child in module.children():
            if isinstance(child, NoisyQuantizeLinear):
                child.enable_noise()
            enable_noise(child)
    
    enable_noise(model)