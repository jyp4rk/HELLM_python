"""
Attention Sink Analysis Utilities

This module provides utilities for analyzing attention sink patterns in transformer models,
specifically comparing maximum attention values between prefix positions and sequence positions
on a per-token basis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Any
from pathlib import Path
import json
import pickle
from collections import defaultdict


class AttentionHook:
    """Hook class to capture pre-softmax attention weights during forward pass."""

    def __init__(self):
        self.attention_weights = {}
        self.layer_count = 0

    def __call__(self, module, input, output):
        """Hook function to capture attention weights before softmax."""
        # The input to this hook is the pre-softmax attention weights
        # This should be registered on the attention module before softmax
        if hasattr(module, 'layer_idx') and module.layer_idx is not None:
            layer_idx = module.layer_idx
        else:
            layer_idx = self.layer_count
            self.layer_count += 1

        # Store the pre-softmax attention weights
        # These are the raw attention logits before softmax normalization
        self.attention_weights[f'layer_{layer_idx}'] = input[0].detach().clone()
        return output

    def clear(self):
        """Clear stored attention weights."""
        self.attention_weights.clear()
        self.layer_count = 0


def register_attention_hooks(model) -> AttentionHook:
    """
    Register hooks to capture pre-softmax attention weights from all attention layers.

    Args:
        model: The transformer model to hook

    Returns:
        AttentionHook: Hook object containing captured attention weights
    """
    hook = AttentionHook()

    # Register hooks on attention layers
    for name, module in model.named_modules():
        if 'attention' in name.lower() and hasattr(module, 'forward'):
            # We want to hook the softmax function to capture pre-softmax weights
            if hasattr(module, 'softmax') or 'softmax' in str(type(module)).lower():
                module.register_forward_hook(hook)

    return hook


def extract_attention_with_per_token_analysis(
    model,
    input_ids: torch.Tensor,
    prefix_length: int,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Any] = None,
    use_pre_softmax: bool = True
) -> Dict[str, Any]:
    """
    Extract attention weights and perform per-token max analysis.

    Args:
        model: The transformer model
        input_ids: Input token IDs [batch_size, seq_len]
        prefix_length: Length of prefix tokens
        attention_mask: Optional attention mask
        past_key_values: Optional prefixed KV cache from prefix tokens
        use_pre_softmax: If True, capture pre-softmax logits; if False, use post-softmax

    Returns:
        Dictionary containing per-token analysis data for each layer
    """
    model.eval()

    if use_pre_softmax:
        # Try direct method first (cleaner and more reliable)
        print("Using pre-softmax attention weights (raw logits) - direct method")
        pre_softmax_data = extract_presoftmax_attention_direct(
            model, input_ids, prefix_length, attention_mask, past_key_values
        )

        # If direct method fails, try monkey-patch
        if not pre_softmax_data:
            print("Direct method failed, trying monkey-patch approach...")
            pre_softmax_data = extract_presoftmax_attention_with_monkey_patch(
                model, input_ids, prefix_length, attention_mask, past_key_values
            )

        # If still no data, fall back to post-softmax
        if not pre_softmax_data:
            print("Pre-softmax capture failed, falling back to post-softmax analysis")
            use_pre_softmax = False
        else:
            return pre_softmax_data

    if not use_pre_softmax:
        # Original post-softmax implementation
        print("Using post-softmax attention weights (normalized)")
        # Forward pass to get attention weights using output_attentions=True
        with torch.no_grad():
            if past_key_values is not None:
                # When using prefix KV cache, pass it to the model
                if attention_mask is not None:
                    outputs = model(input_ids, attention_mask=attention_mask, past_key_values=past_key_values, output_attentions=True)
                else:
                    outputs = model(input_ids, past_key_values=past_key_values, output_attentions=True)
            elif attention_mask is not None:
                outputs = model(input_ids, attention_mask=attention_mask, output_attentions=True)
            else:
                outputs = model(input_ids, output_attentions=True)

        # Extract attention weights from model outputs
        # attentions is a tuple of tensors, one for each layer
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            attention_weights = outputs.attentions
        else:
            # Fallback: try to get from the model's internal structure
            raise ValueError("Model did not return attention weights. Ensure output_attentions=True is supported.")

        # Process attention weights for each layer
        processed_data = {}

        for layer_idx, attn_weights in enumerate(attention_weights):
            layer_name = f'layer_{layer_idx}'

            # attn_weights shape: [batch, heads, seq_len, seq_len]
            # These are POST-softmax attention weights, not pre-softmax
            # For attention sink analysis, we'll analyze the maximum attention values
            if attn_weights.dim() == 4:
                layer_analysis = extract_per_token_attention_maxima(attn_weights, prefix_length)
                processed_data[layer_name] = layer_analysis

        return processed_data


def extract_presoftmax_attention_with_monkey_patch(
    model,
    input_ids: torch.Tensor,
    prefix_length: int,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Extract PRE-softmax attention weights using monkey-patching.

    This function temporarily replaces torch.nn.functional.softmax to capture
    the input tensors (pre-softmax attention logits) during model forward pass.

    Args:
        model: The transformer model
        input_ids: Input token IDs [batch_size, seq_len]
        prefix_length: Length of prefix tokens
        attention_mask: Optional attention mask
        past_key_values: Optional prefixed KV cache from prefix tokens

    Returns:
        Dictionary containing per-token analysis data for each layer with pre-softmax weights
    """
    import torch.nn as nn

    # Store original softmax function
    original_softmax = nn.functional.softmax

    # Storage for captured pre-softmax weights
    captured_weights = {}
    call_counter = [0]  # Use list to make it mutable in nested function
    total_layers = len(model.model.layers) if hasattr(model.model, 'layers') else 32

    def capturing_softmax(input_tensor, dim=-1, dtype=None):
        """Replacement softmax that captures input before applying softmax"""

        # Capture all 4D tensors that could be attention weights
        # Check for square attention matrices (seq_len x seq_len)
        if input_tensor.dim() == 4:
            batch, heads, seq1, seq2 = input_tensor.shape
            # Check if it's a square attention matrix
            if seq1 == seq2 and seq1 > 1:  # Avoid capturing single token tensors
                layer_idx = len(captured_weights)  # Use current count as layer index
                if layer_idx < total_layers:
                    layer_name = f'layer_{layer_idx}'
                    # Clone and detach to avoid affecting gradients
                    captured_weights[layer_name] = input_tensor.clone().detach()
                    print(f"Captured attention from {layer_name}: shape {input_tensor.shape}")

        # Apply original softmax
        if dtype is not None:
            return original_softmax(input_tensor, dim=dim, dtype=dtype)
        else:
            return original_softmax(input_tensor, dim=dim)

    # Temporarily replace the softmax function
    nn.functional.softmax = capturing_softmax

    try:
        # Forward pass to trigger attention computation
        with torch.no_grad():
            if past_key_values is not None:
                if attention_mask is not None:
                    _ = model(input_ids, attention_mask=attention_mask, past_key_values=past_key_values)
                else:
                    _ = model(input_ids, past_key_values=past_key_values)
            elif attention_mask is not None:
                _ = model(input_ids, attention_mask=attention_mask)
            else:
                _ = model(input_ids)

    finally:
        # Always restore the original softmax function
        nn.functional.softmax = original_softmax

    # Process captured pre-softmax weights
    processed_data = {}

    if not captured_weights:
        print("WARNING: No attention weights were captured. The model may use a different softmax pattern.")
        print("Falling back to post-softmax analysis...")
        # Return empty dict to trigger fallback
        return {}

    for layer_name, attn_weights in captured_weights.items():
        if attn_weights.dim() == 4:  # [batch, heads, seq_len, seq_len]
            # These are PRE-softmax attention logits
            layer_analysis = extract_per_token_attention_maxima(attn_weights, prefix_length)
            processed_data[layer_name] = layer_analysis

    print(f"Captured pre-softmax attention from {len(processed_data)} layers")

    return processed_data


def extract_presoftmax_attention_direct(
    model,
    input_ids: torch.Tensor,
    prefix_length: int,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Extract PRE-softmax attention weights using direct model modification.

    This function sets a flag on each attention module to capture pre-softmax weights,
    then runs a forward pass and collects the captured weights.

    Args:
        model: The transformer model
        input_ids: Input token IDs [batch_size, seq_len]
        prefix_length: Length of prefix tokens
        attention_mask: Optional attention mask
        past_key_values: Optional prefixed KV cache from prefix tokens

    Returns:
        Dictionary containing per-token analysis data for each layer with pre-softmax weights
    """
    # Enable pre-softmax capture on all attention layers
    for layer in model.model.layers:
        layer.self_attn.capture_pre_softmax = True
        layer.self_attn.pre_softmax_weights = []

    try:
        # Forward pass to trigger attention computation
        with torch.no_grad():
            if past_key_values is not None:
                if attention_mask is not None:
                    _ = model(input_ids, attention_mask=attention_mask, past_key_values=past_key_values)
                else:
                    _ = model(input_ids, past_key_values=past_key_values)
            elif attention_mask is not None:
                _ = model(input_ids, attention_mask=attention_mask)
            else:
                _ = model(input_ids)

        # Collect captured pre-softmax weights
        processed_data = {}
        for layer_idx, layer in enumerate(model.model.layers):
            if hasattr(layer.self_attn, 'pre_softmax_weights') and layer.self_attn.pre_softmax_weights:
                layer_name = f'layer_{layer_idx}'
                # Get the first (and only) captured weight for this layer
                attn_weights = layer.self_attn.pre_softmax_weights[0]

                if attn_weights.dim() == 4:  # [batch, heads, seq_len, seq_len]
                    # These are PRE-softmax attention logits
                    layer_analysis = extract_per_token_attention_maxima(attn_weights, prefix_length)
                    processed_data[layer_name] = layer_analysis

        print(f"Captured pre-softmax attention from {len(processed_data)} layers using direct method")

    finally:
        # Clean up: disable capture and clear stored weights
        for layer in model.model.layers:
            layer.self_attn.capture_pre_softmax = False
            if hasattr(layer.self_attn, 'pre_softmax_weights'):
                delattr(layer.self_attn, 'pre_softmax_weights')

    return processed_data


def extract_per_token_attention_maxima(
    attn_weights: torch.Tensor,
    prefix_length: int
) -> Dict[str, torch.Tensor]:
    """
    For each token position, extract prefix max and sequence max attention values.

    Args:
        attn_weights: Pre-softmax attention weights [batch, heads, seq_len, seq_len]
        prefix_length: Length of prefix tokens

    Returns:
        Dictionary containing per-token analysis results
    """
    batch_size, num_heads, seq_len, _ = attn_weights.shape

    # Initialize output tensors
    prefix_max_per_token = torch.zeros(batch_size, num_heads, seq_len, device=attn_weights.device)
    sequence_max_per_token = torch.zeros(batch_size, num_heads, seq_len, device=attn_weights.device)
    prefix_positions = torch.zeros(batch_size, num_heads, seq_len, dtype=torch.long, device=attn_weights.device)
    sequence_positions = torch.zeros(batch_size, num_heads, seq_len, dtype=torch.long, device=attn_weights.device)

    postfix_max_per_token = torch.zeros(batch_size, num_heads, seq_len, device=attn_weights.device)
    postfix_positions = torch.zeros(batch_size, num_heads, seq_len, dtype=torch.long, device=attn_weights.device)

    # Initialize global max trackers for prefix and postfix softmax inputs
    global_prefix_max = torch.full((batch_size, num_heads), float('-inf'), device=attn_weights.device)
    global_postfix_max = torch.full((batch_size, num_heads), float('-inf'), device=attn_weights.device)
    
    # Initialize global min trackers for prefix and postfix softmax inputs
    global_prefix_min = torch.full((batch_size, num_heads), float('inf'), device=attn_weights.device)
    global_postfix_min = torch.full((batch_size, num_heads), float('inf'), device=attn_weights.device)
    
    # Track global maximum across ALL positions (for no-prefix case)
    global_max_all = torch.full((batch_size, num_heads), float('-inf'), device=attn_weights.device)
    
    # Track global minimum across ALL positions (for no-prefix case)
    global_min_all = torch.full((batch_size, num_heads), float('inf'), device=attn_weights.device)

    # For each token position i
    for i in range(seq_len):
        # Get attention weights for token i (from all previous positions due to causal mask)
        token_attention = attn_weights[:, :, i, :i+1]  # [batch, heads, i+1]

        if token_attention.size(-1) == 0:
            continue

        # Get prefix portion (only if we have prefix tokens available)
        prefix_end = min(prefix_length, i+1)
        if prefix_end > 0:
            prefix_attention = token_attention[:, :, :prefix_end]  # [batch, heads, prefix_end]
            prefix_max_per_token[:, :, i], prefix_positions[:, :, i] = torch.max(prefix_attention, dim=-1)
            
            # Update global prefix max and min
            current_prefix_max = prefix_max_per_token[:, :, i]
            global_prefix_max = torch.max(global_prefix_max, current_prefix_max)
            global_prefix_min = torch.min(global_prefix_min, current_prefix_max)

            if token_attention.size(-1) > prefix_end:
                postfix_attention = token_attention[:, :, prefix_end:]  # [batch, heads, i+1 - prefix_end]
                postfix_max_per_token[:, :, i], postfix_positions[:, :, i] = torch.max(postfix_attention, dim=-1)
                
                # Update global postfix max and min
                current_postfix_max = postfix_max_per_token[:, :, i]
                global_postfix_max = torch.max(global_postfix_max, current_postfix_max)
                global_postfix_min = torch.min(global_postfix_min, current_postfix_max)

        # Get sequence max (from all available positions for this token)
        sequence_max_per_token[:, :, i], sequence_positions[:, :, i] = torch.max(token_attention, dim=-1)
        
        # Update global maximum and minimum across all positions
        current_max = sequence_max_per_token[:, :, i]
        global_max_all = torch.max(global_max_all, current_max)
        global_min_all = torch.min(global_min_all, current_max)

    # Calculate attention ratios (handle division by zero)
    attention_ratios = torch.zeros_like(prefix_max_per_token)
    valid_mask = sequence_max_per_token != 0
    attention_ratios[valid_mask] = prefix_max_per_token[valid_mask] / sequence_max_per_token[valid_mask]
    
    # Calculate sequence/prefix ratios (sequence_max / prefix_max)
    sequence_prefix_ratios = torch.zeros_like(sequence_max_per_token)
    valid_mask_sp = prefix_max_per_token != 0
    sequence_prefix_ratios[valid_mask_sp] = sequence_max_per_token[valid_mask_sp] / prefix_max_per_token[valid_mask_sp]
    
    # Track the maximum sequence/prefix ratio across all tokens
    max_sequence_prefix_ratio = sequence_prefix_ratios.max().item() if sequence_prefix_ratios.numel() > 0 else 0.0

    # Calculate max difference: sequence_max - prefix_max
    # Positive values mean non-prefix positions have higher attention
    # Negative values mean prefix positions have higher attention
    max_difference = sequence_max_per_token - prefix_max_per_token

    return {
        'raw_attention': attn_weights,
        'prefix_max_per_token': prefix_max_per_token,
        'postfix_max_per_token': postfix_max_per_token,
        'sequence_max_per_token': sequence_max_per_token,
        'prefix_positions': prefix_positions,
        'postfix_positions': postfix_positions,
        'sequence_positions': sequence_positions,
        'attention_ratios': attention_ratios,
        'max_difference': max_difference,
        'sequence_max_per_token': sequence_max_per_token,
        'prefix_max_per_token': prefix_max_per_token,
        'global_prefix_max': global_prefix_max,
        'global_postfix_max': global_postfix_max,
        'global_prefix_min': global_prefix_min,
        'global_postfix_min': global_postfix_min,
        'sequence_prefix_ratios': sequence_prefix_ratios,
        'max_sequence_prefix_ratio': max_sequence_prefix_ratio,
        'global_max_all': global_max_all,
        'global_min_all': global_min_all
    }


def compare_prefix_vs_sequence_per_token(per_token_data: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    """
    Perform statistical comparison of prefix max vs sequence max across all layers.

    Args:
        per_token_data: Dictionary of per-layer token analysis data

    Returns:
        Dictionary containing comparison results and statistics
    """
    all_attention_ratios = []
    all_max_differences = []
    layer_stats = {}
    
    # Global max trackers across all layers
    overall_global_prefix_max = None
    overall_global_postfix_max = None
    overall_max_sequence_prefix_ratio = 0.0
    overall_global_max_all = None
    
    # Global min trackers across all layers
    overall_global_prefix_min = None
    overall_global_postfix_min = None
    overall_global_min_all = None

    # Check if we have any data
    if not per_token_data:
        return {
            'global_stats': {
                'mean_attention_ratio': 0.0,
                'median_attention_ratio': 0.0,
                'mean_max_difference': 0.0,
                'median_max_difference': 0.0,
                'total_comparisons': 0
            },
            'layer_breakdown': {},
            'raw_ratios': torch.tensor([]),
            'raw_differences': torch.tensor([])
        }

    for layer_name, layer_data in per_token_data.items():
        attention_ratios = layer_data['attention_ratios']
        max_difference = layer_data['max_difference']

        # Track global max values across layers
        if 'global_prefix_max' in layer_data and 'global_postfix_max' in layer_data:
            layer_global_prefix_max = layer_data['global_prefix_max']
            layer_global_postfix_max = layer_data['global_postfix_max']
            
            if overall_global_prefix_max is None:
                overall_global_prefix_max = layer_global_prefix_max.clone()
                overall_global_postfix_max = layer_global_postfix_max.clone()
            else:
                # Ensure tensors are on the same device for comparison
                layer_global_prefix_max = layer_global_prefix_max.to(overall_global_prefix_max.device)
                layer_global_postfix_max = layer_global_postfix_max.to(overall_global_postfix_max.device)
                overall_global_prefix_max = torch.max(overall_global_prefix_max, layer_global_prefix_max)
                overall_global_postfix_max = torch.max(overall_global_postfix_max, layer_global_postfix_max)
        
        # Track global min values across layers
        if 'global_prefix_min' in layer_data and 'global_postfix_min' in layer_data:
            layer_global_prefix_min = layer_data['global_prefix_min']
            layer_global_postfix_min = layer_data['global_postfix_min']
            
            if overall_global_prefix_min is None:
                overall_global_prefix_min = layer_global_prefix_min.clone()
                overall_global_postfix_min = layer_global_postfix_min.clone()
            else:
                # Ensure tensors are on the same device for comparison
                layer_global_prefix_min = layer_global_prefix_min.to(overall_global_prefix_min.device)
                layer_global_postfix_min = layer_global_postfix_min.to(overall_global_postfix_min.device)
                overall_global_prefix_min = torch.min(overall_global_prefix_min, layer_global_prefix_min)
                overall_global_postfix_min = torch.min(overall_global_postfix_min, layer_global_postfix_min)
        
        # Track maximum sequence/prefix ratio
        if 'max_sequence_prefix_ratio' in layer_data:
            layer_max_ratio = layer_data['max_sequence_prefix_ratio']
            overall_max_sequence_prefix_ratio = max(overall_max_sequence_prefix_ratio, layer_max_ratio)
        
        # Track global maximum across all positions
        if 'global_max_all' in layer_data:
            layer_global_max_all = layer_data['global_max_all']
            if overall_global_max_all is None:
                overall_global_max_all = layer_global_max_all.clone()
            else:
                # Ensure tensors are on the same device for comparison
                layer_global_max_all = layer_global_max_all.to(overall_global_max_all.device)
                overall_global_max_all = torch.max(overall_global_max_all, layer_global_max_all)
        
        # Track global minimum across all positions
        if 'global_min_all' in layer_data:
            layer_global_min_all = layer_data['global_min_all']
            if overall_global_min_all is None:
                overall_global_min_all = layer_global_min_all.clone()
            else:
                # Ensure tensors are on the same device for comparison
                layer_global_min_all = layer_global_min_all.to(overall_global_min_all.device)
                overall_global_min_all = torch.min(overall_global_min_all, layer_global_min_all)

        # Calculate layer-specific statistics
        layer_mean_ratio = attention_ratios[attention_ratios > 0].mean().item() if attention_ratios.sum() > 0 else 0.0
        layer_mean_diff = max_difference.mean().item()
        layer_median_diff = max_difference.median().item()
        layer_max_diff = max_difference.max().item()
        layer_min_diff = max_difference.min().item()
        layer_std_diff = max_difference.std().item()

        # Calculate percentiles for max_difference
        layer_diff_p25 = max_difference.flatten().quantile(0.25).item()
        layer_diff_p75 = max_difference.flatten().quantile(0.75).item()
        layer_diff_p95 = max_difference.flatten().quantile(0.95).item()

        # Store position data for visualization
        if 'sequence_positions' in layer_data:
            seq_positions = layer_data['sequence_positions']
            prefix_positions = layer_data['prefix_positions']
        else:
            seq_positions = None
            prefix_positions = None

        layer_stats[layer_name] = {
            'mean_attention_ratio': layer_mean_ratio,
            'mean_max_difference': layer_mean_diff,
            'median_max_difference': layer_median_diff,
            'max_max_difference': layer_max_diff,
            'min_max_difference': layer_min_diff,
            'std_max_difference': layer_std_diff,
            'max_diff_p25': layer_diff_p25,
            'max_diff_p75': layer_diff_p75,
            'max_diff_p95': layer_diff_p95,
            'total_tokens': attention_ratios.numel(),
            'sequence_positions': seq_positions,  # Store position data
            'prefix_positions': prefix_positions
        }

        all_attention_ratios.append(attention_ratios.flatten())
        all_max_differences.append(max_difference.flatten())

    # Aggregate across all layers (only if we have data)
    if all_attention_ratios:
        # Move all tensors to the same device (use first tensor's device as reference)
        device = all_attention_ratios[0].device
        all_attention_ratios = torch.cat([t.to(device) for t in all_attention_ratios])
        all_max_differences = torch.cat([t.to(device) for t in all_max_differences])
    else:
        all_attention_ratios = torch.tensor([])
        all_max_differences = torch.tensor([])

    # Calculate global statistics
    valid_ratios = all_attention_ratios[all_attention_ratios > 0] if len(all_attention_ratios) > 0 else torch.tensor([])
    global_mean_ratio = valid_ratios.mean().item() if len(valid_ratios) > 0 else 0.0
    global_median_ratio = valid_ratios.median().item() if len(valid_ratios) > 0 else 0.0
    global_mean_diff = all_max_differences.mean().item() if len(all_max_differences) > 0 else 0.0
    global_median_diff = all_max_differences.median().item() if len(all_max_differences) > 0 else 0.0
    global_max_diff = all_max_differences.max().item() if len(all_max_differences) > 0 else 0.0  # Maximum of all max differences

    # Calculate overall global max statistics
    overall_prefix_max_value = overall_global_prefix_max.max().item() if overall_global_prefix_max is not None else 0.0
    overall_postfix_max_value = overall_global_postfix_max.max().item() if overall_global_postfix_max is not None else 0.0
    overall_max_all_value = overall_global_max_all.max().item() if overall_global_max_all is not None else 0.0
    
    # Calculate overall global min statistics
    overall_prefix_min_value = overall_global_prefix_min.min().item() if overall_global_prefix_min is not None else 0.0
    overall_postfix_min_value = overall_global_postfix_min.min().item() if overall_global_postfix_min is not None else 0.0
    overall_min_all_value = overall_global_min_all.min().item() if overall_global_min_all is not None else 0.0

    return {
        'global_stats': {
            'mean_attention_ratio': global_mean_ratio,
            'median_attention_ratio': global_median_ratio,
            'mean_max_difference': global_mean_diff,
            'median_max_difference': global_median_diff,
            'max_max_difference': global_max_diff,  # Added max of max differences
            'total_comparisons': len(all_attention_ratios),
            'global_prefix_max': overall_prefix_max_value,
            'global_postfix_max': overall_postfix_max_value,
            'global_prefix_min': overall_prefix_min_value,
            'global_postfix_min': overall_postfix_min_value,
            'max_sequence_prefix_ratio': overall_max_sequence_prefix_ratio,
            'global_max_all_positions': overall_max_all_value,
            'global_min_all_positions': overall_min_all_value
        },
        'layer_breakdown': layer_stats,
        'raw_ratios': all_attention_ratios,
        'raw_differences': all_max_differences
    }


def analyze_per_token_sink_patterns(
    comparison_results: Dict[str, Any],
    per_token_data: Dict[str, Dict[str, torch.Tensor]],
    prefix_length: int
) -> Dict[str, Any]:
    """
    Analyze patterns in per-token sink behavior across positions and layers.

    Args:
        comparison_results: Results from compare_prefix_vs_sequence_per_token
        per_token_data: Raw per-token data
        prefix_length: Length of prefix tokens

    Returns:
        Dictionary containing detailed pattern analysis
    """
    # Aggregate attention ratios by position
    position_attention_ratios = defaultdict(list)
    position_prefix_max = defaultdict(list)
    position_postfix_max = defaultdict(list)

    for layer_name, layer_data in per_token_data.items():
        attention_ratios = layer_data['attention_ratios']
        prefix_max_per_token = layer_data['prefix_max_per_token']  # [batch, heads, seq_len]
        postfix_max_per_token = layer_data['postfix_max_per_token']  # [batch, heads, seq_len]

        seq_len = attention_ratios.size(-1)

        # Calculate position-wise statistics
        for pos in range(seq_len):
            pos_ratio = attention_ratios[:, :, pos][attention_ratios[:, :, pos] > 0].mean().item()
            # Use max instead of mean to get the global maximum for each position
            pos_prefix_max = prefix_max_per_token[:, :, pos].float().max().item()
            pos_postfix_max = postfix_max_per_token[:, :, pos].float().max().item()

            position_prefix_max[pos].append(pos_prefix_max)
            position_postfix_max[pos].append(pos_postfix_max)
            
            if not torch.isnan(torch.tensor(pos_ratio)):
                position_attention_ratios[pos].append(pos_ratio)

    # Average across layers for each position (but use max for prefix/postfix max values)
    position_analysis = {}
    for pos in position_prefix_max.keys():
        position_analysis[pos] = {
            'mean_attention_ratio': np.mean(position_attention_ratios[pos]) if position_attention_ratios[pos] else 0.0,
            'prefix_max': np.max(position_prefix_max[pos]) if position_prefix_max[pos] else 0.0,
            'postfix_max': np.max(position_postfix_max[pos]) if position_postfix_max[pos] else 0.0,
        }

    # Categorize positions
    seq_len = max(position_analysis.keys()) + 1 if position_analysis else 0
    early_positions = list(range(min(prefix_length, seq_len)))
    mid_positions = list(range(prefix_length, min(2 * prefix_length, seq_len)))
    late_positions = list(range(2 * prefix_length, seq_len))

    def aggregate_positions(positions):
        if not positions:
            return {'mean_attention_ratio': 0.0, 'prefix_max': 0.0, 'postfix_max': 0.0}
        ratios = [position_analysis[p]['mean_attention_ratio'] for p in positions if p in position_analysis]
        prefix_maxs = [position_analysis[p]['prefix_max'] for p in positions if p in position_analysis]
        postfix_maxs = [position_analysis[p]['postfix_max'] for p in positions if p in position_analysis]
        return {
            'mean_attention_ratio': np.mean(ratios) if ratios else 0.0,
            'prefix_max': np.max(prefix_maxs) if prefix_maxs else 0.0,  # Use max for global maximum
            'postfix_max': np.max(postfix_maxs) if postfix_maxs else 0.0  # Use max for global maximum
        }

    positional_stats = {
        'early_tokens': aggregate_positions(early_positions),
        'mid_tokens': aggregate_positions(mid_positions),
        'late_tokens': aggregate_positions(late_positions)
    }

    return {
        'global_stats': comparison_results['global_stats'],
        'positional_stats': positional_stats,
        'position_analysis': position_analysis,
        'layer_breakdown': comparison_results['layer_breakdown']
    }


def visualize_per_token_sink_analysis(
    per_token_data: Dict[str, Dict[str, torch.Tensor]],
    analysis_results: Dict[str, Any],
    prefix_length: int,
    output_dir: Path
) -> None:
    """
    Generate visualizations for per-token sink analysis.

    Args:
        per_token_data: Raw per-token analysis data
        analysis_results: Statistical analysis results
        prefix_length: Length of prefix tokens
        output_dir: Directory to save visualizations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get layer names and sequence length
    layer_names = list(per_token_data.keys())
    if not layer_names:
        return
    seq_len = per_token_data[layer_names[0]]['attention_ratios'].size(-1)

    # 1. Max difference heatmap across layers and positions
    plt.figure(figsize=(12, 8))

    # Create heatmap data for max differences
    max_diff_heatmap = np.zeros((len(layer_names), seq_len))

    for i, layer_name in enumerate(layer_names):
        max_difference = per_token_data[layer_name]['max_difference']
        # Average across batch and heads
        layer_max_diff = max_difference.mean(dim=(0, 1)).cpu().numpy()
        max_diff_heatmap[i, :] = layer_max_diff

    ax = sns.heatmap(max_diff_heatmap,
                     xticklabels=range(seq_len) if seq_len <= 50 else range(0, seq_len, seq_len//50),
                     yticklabels=layer_names,
                     cmap='coolwarm',
                     center=0,  # Center colormap at 0
                     cbar_kws={'label': 'Max Difference (Sequence - Prefix)'})

    ax.axvline(x=prefix_length, color='black', linestyle='--', alpha=0.7, label='Prefix End')
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Layer')
    ax.set_title('Max Attention Difference (Sequence - Prefix) Across Layers')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'max_difference_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Layer-wise statistics plot
    if 'layer_breakdown' in analysis_results and analysis_results['layer_breakdown']:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        layer_stats = analysis_results['layer_breakdown']
        sorted_layers = sorted(layer_stats.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
        layer_indices = [int(name.split('_')[1]) if '_' in name else i for i, name in enumerate(sorted_layers)]

        # Extract statistics per layer
        mean_ratios = [layer_stats[layer]['mean_attention_ratio'] for layer in sorted_layers]
        mean_diffs = [layer_stats[layer]['mean_max_difference'] for layer in sorted_layers]
        max_diffs = [layer_stats[layer].get('max_max_difference', 0) for layer in sorted_layers]

        # Plot 1: Mean attention ratio by layer
        axes[0, 0].plot(layer_indices, mean_ratios, 'b-o', linewidth=2, markersize=6)
        axes[0, 0].set_xlabel('Layer Index')
        axes[0, 0].set_ylabel('Mean Attention Ratio')
        axes[0, 0].set_title('Mean Attention Ratio by Layer')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 1.05])

        # Plot 2: Mean attention ratio by layer
        axes[0, 1].plot(layer_indices, mean_ratios, 'g-o', linewidth=2, markersize=6)
        axes[0, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Equal attention')
        axes[0, 1].set_xlabel('Layer Index')
        axes[0, 1].set_ylabel('Mean Attention Ratio')
        axes[0, 1].set_title('Mean Prefix/Sequence Attention Ratio by Layer')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        # Plot 3: Mean max difference by layer
        axes[1, 0].plot(layer_indices, mean_diffs, 'r-o', linewidth=2, markersize=6)
        axes[1, 0].axhline(y=0.0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Layer Index')
        axes[1, 0].set_ylabel('Mean Max Difference')
        axes[1, 0].set_title('Mean Max Attention Difference by Layer')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Max max difference by layer
        axes[1, 1].plot(layer_indices, max_diffs, 'm-o', linewidth=2, markersize=6)
        axes[1, 1].set_xlabel('Layer Index')
        axes[1, 1].set_ylabel('Max Max Difference')
        axes[1, 1].set_title('Maximum Attention Difference by Layer')
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle('Layer-wise Attention Sink Analysis', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / 'layer_wise_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 5. Maximum attention position heatmap
    if 'layer_breakdown' in analysis_results and analysis_results['layer_breakdown']:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        layer_stats = analysis_results['layer_breakdown']
        sorted_layers = sorted(layer_stats.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x else 0)

        # Collect position data for each layer
        max_seq_len = 0
        for layer_name in sorted_layers:
            if layer_stats[layer_name]['sequence_positions'] is not None:
                seq_positions = layer_stats[layer_name]['sequence_positions']
                max_seq_len = max(max_seq_len, seq_positions.max().item() + 1)

        # Create heatmap data for sequence max positions
        position_heatmap = np.zeros((len(sorted_layers), int(max_seq_len)))
        prefix_position_heatmap = np.zeros((len(sorted_layers), int(max_seq_len)))

        for i, layer_name in enumerate(sorted_layers):
            if layer_stats[layer_name]['sequence_positions'] is not None:
                seq_positions = layer_stats[layer_name]['sequence_positions'].cpu().numpy().flatten()
                prefix_positions = layer_stats[layer_name]['prefix_positions'].cpu().numpy().flatten()

                # Count occurrences of each position
                for pos in seq_positions:
                    if 0 <= pos < max_seq_len:
                        position_heatmap[i, int(pos)] += 1

                for pos in prefix_positions:
                    if 0 <= pos < max_seq_len:
                        prefix_position_heatmap[i, int(pos)] += 1

        # Normalize by row (each layer) to show distribution
        row_sums = position_heatmap.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        position_heatmap_norm = position_heatmap / row_sums

        row_sums_prefix = prefix_position_heatmap.sum(axis=1, keepdims=True)
        row_sums_prefix[row_sums_prefix == 0] = 1
        prefix_position_heatmap_norm = prefix_position_heatmap / row_sums_prefix

        # Plot 1: Overall maximum attention positions
        im1 = ax1.imshow(position_heatmap_norm, aspect='auto', cmap='hot', interpolation='nearest')
        ax1.set_xlabel('Token Position (where maximum attention occurs)')
        ax1.set_ylabel('Layer')
        ax1.set_title('Distribution of Maximum Attention Positions by Layer')
        ax1.axvline(x=prefix_length, color='cyan', linestyle='--', alpha=0.7, label='Prefix End')

        # Set y-tick labels
        ax1.set_yticks(range(len(sorted_layers)))
        ax1.set_yticklabels([f"L{name.split('_')[1]}" if '_' in name else name for name in sorted_layers])

        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Frequency (normalized per layer)')
        ax1.legend()

        # Plot 2: Prefix-only maximum positions
        im2 = ax2.imshow(prefix_position_heatmap_norm, aspect='auto', cmap='cool', interpolation='nearest')
        ax2.set_xlabel('Token Position (where prefix maximum occurs)')
        ax2.set_ylabel('Layer')
        ax2.set_title('Distribution of Prefix Maximum Positions by Layer')
        ax2.axvline(x=prefix_length, color='red', linestyle='--', alpha=0.7, label='Prefix End')

        # Set y-tick labels
        ax2.set_yticks(range(len(sorted_layers)))
        ax2.set_yticklabels([f"L{name.split('_')[1]}" if '_' in name else name for name in sorted_layers])

        # Add colorbar
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Frequency (normalized per layer)')
        ax2.legend()

        plt.suptitle('Maximum Attention Position Analysis', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / 'max_attention_positions.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 6. Attention ratio and max difference distributions
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Overall attention ratio distribution
    all_ratios = []
    all_differences = []
    for layer_data in per_token_data.values():
        ratios = layer_data['attention_ratios'][layer_data['attention_ratios'] > 0]
        all_ratios.extend(ratios.cpu().numpy())
        differences = layer_data['max_difference']
        all_differences.extend(differences.cpu().numpy().flatten())

    axes[0, 0].hist(all_ratios, bins=50, alpha=0.7, color='blue')
    axes[0, 0].axvline(x=1.0, color='red', linestyle='--', label='Equal Weight')
    axes[0, 0].set_xlabel('Prefix/Sequence Attention Ratio')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Overall Attention Ratio Distribution')
    axes[0, 0].legend()

    # Max difference distribution
    axes[0, 1].hist(all_differences, bins=50, alpha=0.7, color='green')
    axes[0, 1].axvline(x=0.0, color='red', linestyle='--', label='Zero Difference')
    axes[0, 1].set_xlabel('Max Difference (Sequence - Prefix)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Max Attention Difference Distribution')
    axes[0, 1].legend()

    # Distribution by position categories
    categories = ['early_tokens', 'mid_tokens', 'late_tokens']
    colors = ['green', 'orange', 'purple']

    for i, (category, color) in enumerate(zip(categories, colors)):
        ax = axes[0, 2] if i == 0 else axes[1, i]

        # Extract ratios for this category
        if category == 'early_tokens':
            pos_range = range(min(prefix_length, seq_len))
        elif category == 'mid_tokens':
            pos_range = range(prefix_length, min(2 * prefix_length, seq_len))
        else:
            pos_range = range(2 * prefix_length, seq_len)

        category_ratios = []
        for layer_data in per_token_data.values():
            for pos in pos_range:
                if pos < layer_data['attention_ratios'].size(-1):
                    ratios = layer_data['attention_ratios'][:, :, pos]
                    valid_ratios = ratios[ratios > 0]
                    category_ratios.extend(valid_ratios.cpu().numpy())

        if category_ratios:
            ax.hist(category_ratios, bins=30, alpha=0.7, color=color)
            ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.7)
            ax.set_xlabel('Prefix/Sequence Attention Ratio')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{category.replace("_", " ").title()} Distribution')

    plt.tight_layout()
    plt.savefig(output_dir / 'attention_ratio_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_analysis_results(
    per_token_data: Dict[str, Dict[str, torch.Tensor]],
    analysis_results: Dict[str, Any],
    output_dir: Path
) -> None:
    """
    Save analysis results to files.

    Args:
        per_token_data: Raw per-token analysis data
        analysis_results: Statistical analysis results
        output_dir: Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw data as pickle
    with open(output_dir / 'per_token_analysis.pkl', 'wb') as f:
        pickle.dump(per_token_data, f)

    # Save statistics as JSON (convert tensors to lists recursively)
    def convert_to_json_serializable(obj):
        """Recursively convert tensors and arrays to JSON-serializable format."""
        if isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist() if hasattr(obj, 'tolist') else obj
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        else:
            return obj

    json_results = convert_to_json_serializable(analysis_results)

    with open(output_dir / 'per_token_statistics.json', 'w') as f:
        json.dump(json_results, f, indent=2)


def run_complete_sink_analysis(
    model,
    input_ids: torch.Tensor,
    prefix_length: int,
    output_dir: Path,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Any] = None,
    use_pre_softmax: bool = True
) -> Dict[str, Any]:
    """
    Run complete attention sink analysis pipeline.

    Args:
        model: The transformer model
        input_ids: Input token IDs
        prefix_length: Length of prefix tokens
        output_dir: Directory to save results
        attention_mask: Optional attention mask
        past_key_values: Optional prefixed KV cache from prefix tokens
        use_pre_softmax: If True, analyze pre-softmax logits; if False, use post-softmax

    Returns:
        Complete analysis results
    """
    print("Extracting attention weights and performing per-token analysis...")
    per_token_data = extract_attention_with_per_token_analysis(
        model, input_ids, prefix_length, attention_mask, past_key_values, use_pre_softmax
    )

    print("Comparing prefix vs sequence attention patterns...")
    comparison_results = compare_prefix_vs_sequence_per_token(per_token_data)

    print("Analyzing sink patterns across positions and layers...")
    analysis_results = analyze_per_token_sink_patterns(
        comparison_results, per_token_data, prefix_length
    )

    print("Generating visualizations...")
    visualize_per_token_sink_analysis(
        per_token_data, analysis_results, prefix_length, output_dir
    )

    print("Saving results...")
    save_analysis_results(per_token_data, analysis_results, output_dir)

    # Print summary
    global_stats = analysis_results['global_stats']
    print(f"\n{'='*60}")
    print(f"GLOBAL ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Median attention ratio: {global_stats['median_attention_ratio']:.3f}")
    print(f"Median max difference: {global_stats['median_max_difference']:.3f}")
    if 'max_max_difference' in global_stats:
        print(f"Max max difference: {global_stats['max_max_difference']:.3f}")
    
    # Print global max values for prefix and postfix
    if 'global_prefix_max' in global_stats and 'global_postfix_max' in global_stats:
        print(f"Global prefix max: {global_stats['global_prefix_max']:.3f}")
        print(f"Global postfix max: {global_stats['global_postfix_max']:.3f}")
    
    # Print global min values for prefix and postfix
    if 'global_prefix_min' in global_stats and 'global_postfix_min' in global_stats:
        print(f"Global prefix min: {global_stats['global_prefix_min']:.3f}")
        print(f"Global postfix min: {global_stats['global_postfix_min']:.3f}")
    
    # Print maximum sequence/prefix ratio
    if 'max_sequence_prefix_ratio' in global_stats:
        print(f"Max sequence/prefix ratio: {global_stats['max_sequence_prefix_ratio']:.3f}")
    
    # Print global maximum across all positions (useful for no-prefix case)
    if 'global_max_all_positions' in global_stats:
        print(f"Global max (all positions): {global_stats['global_max_all_positions']:.3f}")
    
    # Print global minimum across all positions (useful for no-prefix case)
    if 'global_min_all_positions' in global_stats:
        print(f"Global min (all positions): {global_stats['global_min_all_positions']:.3f}")

    # Print layer-wise statistics
    if 'layer_breakdown' in analysis_results and analysis_results['layer_breakdown']:
        print(f"\n{'='*60}")
        print(f"LAYER-WISE ANALYSIS")
        print(f"{'='*60}")

        layer_stats = analysis_results['layer_breakdown']
        # Sort layers by their numeric index
        sorted_layers = sorted(layer_stats.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x else 0)

        for layer_name in sorted_layers:
            stats = layer_stats[layer_name]
            layer_idx = layer_name.split('_')[1] if '_' in layer_name else layer_name
            print(f"\nLayer {layer_idx}:")
            print(f"  Max difference stats:")
            if 'median_max_difference' in stats:
                print(f"    Median: {stats['median_max_difference']:.3f}")
            if 'max_max_difference' in stats:
                print(f"    Max:    {stats['max_max_difference']:.3f}")
            if 'max_diff_p95' in stats:
                print(f"    P95:    {stats['max_diff_p95']:.3f}")

    # Additional info for pre-softmax values
    if 'raw_differences' in analysis_results and len(analysis_results['raw_differences']) > 0:
        raw_diffs = analysis_results['raw_differences']
        if isinstance(raw_diffs, torch.Tensor):
            print(f"\nPre-softmax logit statistics:")
            print(f"Max logit difference range: [{raw_diffs.min().item():.2f}, {raw_diffs.max().item():.2f}]")
            print(f"Std of max differences: {raw_diffs.std().item():.3f}")

    return analysis_results
