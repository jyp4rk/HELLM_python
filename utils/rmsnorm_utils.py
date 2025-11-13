"""
RMSNorm Input Range Analysis Utilities

This module provides utilities for analyzing input range of RMSNorm layers in transformer models.
Tracks the input range (min/max) per layer and position (pre-attention vs post-attention).
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


def extract_rmsnorm_inputs(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Extract RMSNorm variance statistics across all layers and positions.

    RMSNorm computes: variance = (1/d) * sum(x_i^2)
    We track the min and max of this variance term per layer and position.

    RMSNorm appears at two positions per layer:
    1. input_layernorm: Before attention (pre-attention)
    2. post_attention_layernorm: After attention, before MLP (post-attention)

    Args:
        model: The transformer model
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Optional attention mask

    Returns:
        Dictionary containing RMSNorm variance statistics per layer and position
    """
    model.eval()

    # Storage for variance statistics (memory efficient)
    # Structure: {layer_idx: {position: {'variance_min': tensor, 'variance_max': tensor}}}
    rmsnorm_data = defaultdict(lambda: defaultdict(dict))

    # Register hooks on all RMSNorm layers
    hooks = []

    def create_hook(layer_idx, position):
        """Create a hook function for capturing RMSNorm variance statistics"""
        def hook_fn(module, input, output):
            # input is a tuple, get the actual tensor
            input_tensor = input[0].detach()  # [batch_size, seq_len, hidden_size]

            # Calculate variance as RMSNorm does: (1/d) * sum(x_i^2)
            # This is the mean of squared values along the hidden dimension
            variance = input_tensor.pow(2).mean(-1)  # [batch_size, seq_len]

            # Update min/max for this layer-position
            if position not in rmsnorm_data[layer_idx] or 'variance_min' not in rmsnorm_data[layer_idx][position]:
                # First time seeing this layer-position
                rmsnorm_data[layer_idx][position]['variance_min'] = variance.clone()
                rmsnorm_data[layer_idx][position]['variance_max'] = variance.clone()
            else:
                # Update min/max
                rmsnorm_data[layer_idx][position]['variance_min'] = torch.min(
                    rmsnorm_data[layer_idx][position]['variance_min'],
                    variance
                )
                rmsnorm_data[layer_idx][position]['variance_max'] = torch.max(
                    rmsnorm_data[layer_idx][position]['variance_max'],
                    variance
                )
        return hook_fn

    # Register hooks on each decoder layer's RMSNorm modules
    for layer_idx, layer in enumerate(model.model.layers):
        # Pre-attention RMSNorm (input_layernorm)
        hook = layer.input_layernorm.register_forward_hook(
            create_hook(layer_idx, 'pre_attention')
        )
        hooks.append(hook)

        # Post-attention RMSNorm (post_attention_layernorm)
        hook = layer.post_attention_layernorm.register_forward_hook(
            create_hook(layer_idx, 'post_attention')
        )
        hooks.append(hook)

    try:
        # Forward pass to trigger RMSNorm computation
        with torch.no_grad():
            if attention_mask is not None:
                _ = model(input_ids, attention_mask=attention_mask)
            else:
                _ = model(input_ids)

        # Process captured data
        processed_data = {}

        for layer_idx in sorted(rmsnorm_data.keys()):
            layer_data = {}

            for position in ['pre_attention', 'post_attention']:
                if position in rmsnorm_data[layer_idx]:
                    variance_min = rmsnorm_data[layer_idx][position]['variance_min']
                    variance_max = rmsnorm_data[layer_idx][position]['variance_max']

                    # Calculate statistics from variance min/max
                    stats = calculate_variance_stats(variance_min, variance_max)
                    layer_data[position] = stats

            processed_data[f'layer_{layer_idx}'] = layer_data

        print(f"Captured RMSNorm variance statistics from {len(processed_data)} layers")

    finally:
        # Remove all hooks
        for hook in hooks:
            hook.remove()

    return processed_data


def calculate_variance_stats(variance_min: torch.Tensor, variance_max: torch.Tensor) -> Dict[str, Any]:
    """
    Calculate statistics from RMSNorm variance min/max tensors.

    Args:
        variance_min: Minimum variance per position [batch_size, seq_len]
        variance_max: Maximum variance per position [batch_size, seq_len]

    Returns:
        Dictionary containing variance statistics
    """
    batch_size, seq_len = variance_min.shape

    # Global statistics (across all positions and batches)
    global_variance_min = variance_min.min().item()
    global_variance_max = variance_max.max().item()

    # Average variance across positions
    avg_variance_min = variance_min.mean().item()
    avg_variance_max = variance_max.mean().item()

    # Per-position statistics (across batches)
    # Min/max variance at each position
    position_variance_min = variance_min.min(dim=0)[0]  # [seq_len]
    position_variance_max = variance_max.max(dim=0)[0]  # [seq_len]

    return {
        'global_variance_min': global_variance_min,
        'global_variance_max': global_variance_max,
        'global_variance_range': global_variance_max - global_variance_min,
        'avg_variance_min': avg_variance_min,
        'avg_variance_max': avg_variance_max,
        'position_variance_min': position_variance_min,
        'position_variance_max': position_variance_max,
        'seq_len': seq_len,
        'batch_size': batch_size,
    }


def analyze_rmsnorm_patterns(rmsnorm_data: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Analyze patterns in RMSNorm variance across layers and positions.

    Args:
        rmsnorm_data: Dictionary of per-layer, per-position RMSNorm variance statistics

    Returns:
        Dictionary containing aggregated analysis results
    """
    # Aggregate statistics
    layer_stats = {}

    # Track overall variance max/min across all layers and positions
    overall_variance_max = float('-inf')
    overall_variance_min = float('inf')

    # Track variance max/min by position type
    pre_attention_variance_max = float('-inf')
    pre_attention_variance_min = float('inf')
    post_attention_variance_max = float('-inf')
    post_attention_variance_min = float('inf')

    # Layer-wise tracking
    for layer_name, layer_data in rmsnorm_data.items():
        layer_idx = int(layer_name.split('_')[1]) if '_' in layer_name else 0

        layer_stats[layer_name] = {}

        for position in ['pre_attention', 'post_attention']:
            if position in layer_data:
                stats = layer_data[position]

                # Update overall statistics
                overall_variance_max = max(overall_variance_max, stats['global_variance_max'])
                overall_variance_min = min(overall_variance_min, stats['global_variance_min'])

                # Update position-specific statistics
                if position == 'pre_attention':
                    pre_attention_variance_max = max(pre_attention_variance_max, stats['global_variance_max'])
                    pre_attention_variance_min = min(pre_attention_variance_min, stats['global_variance_min'])
                else:
                    post_attention_variance_max = max(post_attention_variance_max, stats['global_variance_max'])
                    post_attention_variance_min = min(post_attention_variance_min, stats['global_variance_min'])

                # Store layer-position statistics
                layer_stats[layer_name][position] = {
                    'global_variance_max': stats['global_variance_max'],
                    'global_variance_min': stats['global_variance_min'],
                    'global_variance_range': stats['global_variance_range'],
                    'avg_variance_min': stats['avg_variance_min'],
                    'avg_variance_max': stats['avg_variance_max'],
                }

    return {
        'overall_stats': {
            'variance_max': overall_variance_max,
            'variance_min': overall_variance_min,
            'variance_range': overall_variance_max - overall_variance_min,
        },
        'pre_attention_stats': {
            'variance_max': pre_attention_variance_max,
            'variance_min': pre_attention_variance_min,
            'variance_range': pre_attention_variance_max - pre_attention_variance_min,
        },
        'post_attention_stats': {
            'variance_max': post_attention_variance_max,
            'variance_min': post_attention_variance_min,
            'variance_range': post_attention_variance_max - post_attention_variance_min,
        },
        'layer_breakdown': layer_stats,
    }


def visualize_rmsnorm_analysis(
    rmsnorm_data: Dict[str, Dict[str, Dict[str, Any]]],
    analysis_results: Dict[str, Any],
    output_dir: Path
) -> None:
    """
    Generate visualizations for RMSNorm input analysis.

    Args:
        rmsnorm_data: Raw RMSNorm input data
        analysis_results: Statistical analysis results
        output_dir: Directory to save visualizations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_names = sorted(rmsnorm_data.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
    num_layers = len(layer_names)

    # Get sequence length from first layer
    seq_len = None
    for layer_data in rmsnorm_data.values():
        for position_data in layer_data.values():
            if 'seq_len' in position_data:
                seq_len = position_data['seq_len']
                break
        if seq_len:
            break

    # 1. Layer-wise max/min heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Prepare data for heatmap
    pre_attn_data = np.zeros((num_layers, 3))  # [variance_max, variance_min, avg_variance]
    post_attn_data = np.zeros((num_layers, 3))

    for i, layer_name in enumerate(layer_names):
        layer_data = rmsnorm_data[layer_name]

        if 'pre_attention' in layer_data:
            pre_attn_data[i, 0] = layer_data['pre_attention']['global_variance_max']
            pre_attn_data[i, 1] = layer_data['pre_attention']['global_variance_min']
            pre_attn_data[i, 2] = (layer_data['pre_attention']['avg_variance_max'] +
                                   layer_data['pre_attention']['avg_variance_min']) / 2

        if 'post_attention' in layer_data:
            post_attn_data[i, 0] = layer_data['post_attention']['global_variance_max']
            post_attn_data[i, 1] = layer_data['post_attention']['global_variance_min']
            post_attn_data[i, 2] = (layer_data['post_attention']['avg_variance_max'] +
                                    layer_data['post_attention']['avg_variance_min']) / 2

    # Plot pre-attention
    sns.heatmap(pre_attn_data,
                xticklabels=['Var Max', 'Var Min', 'Avg Var'],
                yticklabels=[f"L{i}" for i in range(num_layers)],
                annot=True, fmt='.3f', cmap='viridis', ax=ax1)
    ax1.set_title('Pre-Attention RMSNorm Variance Statistics')
    ax1.set_ylabel('Layer')

    # Plot post-attention
    sns.heatmap(post_attn_data,
                xticklabels=['Var Max', 'Var Min', 'Avg Var'],
                yticklabels=[f"L{i}" for i in range(num_layers)],
                annot=True, fmt='.3f', cmap='viridis', ax=ax2)
    ax2.set_title('Post-Attention RMSNorm Variance Statistics')
    ax2.set_ylabel('Layer')

    plt.tight_layout()
    plt.savefig(output_dir / 'rmsnorm_layer_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Range comparison by layer
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    layer_indices = list(range(num_layers))

    # Extract variance data for plotting
    pre_var_max = [rmsnorm_data[ln]['pre_attention']['global_variance_max'] if 'pre_attention' in rmsnorm_data[ln] else 0 for ln in layer_names]
    pre_var_min = [rmsnorm_data[ln]['pre_attention']['global_variance_min'] if 'pre_attention' in rmsnorm_data[ln] else 0 for ln in layer_names]
    post_var_max = [rmsnorm_data[ln]['post_attention']['global_variance_max'] if 'post_attention' in rmsnorm_data[ln] else 0 for ln in layer_names]
    post_var_min = [rmsnorm_data[ln]['post_attention']['global_variance_min'] if 'post_attention' in rmsnorm_data[ln] else 0 for ln in layer_names]

    # Plot 1: Variance Max values
    axes[0, 0].plot(layer_indices, pre_var_max, 'b-o', label='Pre-Attention', linewidth=2)
    axes[0, 0].plot(layer_indices, post_var_max, 'r-s', label='Post-Attention', linewidth=2)
    axes[0, 0].set_xlabel('Layer Index')
    axes[0, 0].set_ylabel('Variance Maximum')
    axes[0, 0].set_title('RMSNorm Variance Maximum by Layer')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Variance Min values
    axes[0, 1].plot(layer_indices, pre_var_min, 'b-o', label='Pre-Attention', linewidth=2)
    axes[0, 1].plot(layer_indices, post_var_min, 'r-s', label='Post-Attention', linewidth=2)
    axes[0, 1].set_xlabel('Layer Index')
    axes[0, 1].set_ylabel('Variance Minimum')
    axes[0, 1].set_title('RMSNorm Variance Minimum by Layer')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Variance Range (max - min)
    pre_var_range = [pre_var_max[i] - pre_var_min[i] for i in range(num_layers)]
    post_var_range = [post_var_max[i] - post_var_min[i] for i in range(num_layers)]
    axes[1, 0].plot(layer_indices, pre_var_range, 'b-o', label='Pre-Attention', linewidth=2)
    axes[1, 0].plot(layer_indices, post_var_range, 'r-s', label='Post-Attention', linewidth=2)
    axes[1, 0].set_xlabel('Layer Index')
    axes[1, 0].set_ylabel('Variance Range')
    axes[1, 0].set_title('RMSNorm Variance Range by Layer')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Average Variance values
    pre_avg_var = [(rmsnorm_data[ln]['pre_attention']['avg_variance_max'] + rmsnorm_data[ln]['pre_attention']['avg_variance_min'])/2 if 'pre_attention' in rmsnorm_data[ln] else 0 for ln in layer_names]
    post_avg_var = [(rmsnorm_data[ln]['post_attention']['avg_variance_max'] + rmsnorm_data[ln]['post_attention']['avg_variance_min'])/2 if 'post_attention' in rmsnorm_data[ln] else 0 for ln in layer_names]
    axes[1, 1].plot(layer_indices, pre_avg_var, 'b-o', label='Pre-Attention', linewidth=2)
    axes[1, 1].plot(layer_indices, post_avg_var, 'r-s', label='Post-Attention', linewidth=2)
    axes[1, 1].set_xlabel('Layer Index')
    axes[1, 1].set_ylabel('Average Variance')
    axes[1, 1].set_title('RMSNorm Average Variance by Layer')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('RMSNorm Input Analysis Across Layers', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'rmsnorm_layer_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Position-wise analysis (if sequence length is available)
    if seq_len:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Get first and last layer for comparison
        first_layer = layer_names[0]
        last_layer = layer_names[-1]

        positions = list(range(seq_len))

        # First layer pre-attention
        if 'pre_attention' in rmsnorm_data[first_layer]:
            pos_var_max = rmsnorm_data[first_layer]['pre_attention']['position_variance_max'].cpu().numpy()
            pos_var_min = rmsnorm_data[first_layer]['pre_attention']['position_variance_min'].cpu().numpy()
            axes[0, 0].plot(positions, pos_var_max, 'b-', label='Var Max', linewidth=2)
            axes[0, 0].plot(positions, pos_var_min, 'r-', label='Var Min', linewidth=2)
            axes[0, 0].fill_between(positions, pos_var_min, pos_var_max, alpha=0.3)
            axes[0, 0].set_xlabel('Position')
            axes[0, 0].set_ylabel('Variance')
            axes[0, 0].set_title(f'{first_layer} Pre-Attention')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # First layer post-attention
        if 'post_attention' in rmsnorm_data[first_layer]:
            pos_var_max = rmsnorm_data[first_layer]['post_attention']['position_variance_max'].cpu().numpy()
            pos_var_min = rmsnorm_data[first_layer]['post_attention']['position_variance_min'].cpu().numpy()
            axes[0, 1].plot(positions, pos_var_max, 'b-', label='Var Max', linewidth=2)
            axes[0, 1].plot(positions, pos_var_min, 'r-', label='Var Min', linewidth=2)
            axes[0, 1].fill_between(positions, pos_var_min, pos_var_max, alpha=0.3)
            axes[0, 1].set_xlabel('Position')
            axes[0, 1].set_ylabel('Variance')
            axes[0, 1].set_title(f'{first_layer} Post-Attention')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # Last layer pre-attention
        if 'pre_attention' in rmsnorm_data[last_layer]:
            pos_var_max = rmsnorm_data[last_layer]['pre_attention']['position_variance_max'].cpu().numpy()
            pos_var_min = rmsnorm_data[last_layer]['pre_attention']['position_variance_min'].cpu().numpy()
            axes[1, 0].plot(positions, pos_var_max, 'b-', label='Var Max', linewidth=2)
            axes[1, 0].plot(positions, pos_var_min, 'r-', label='Var Min', linewidth=2)
            axes[1, 0].fill_between(positions, pos_var_min, pos_var_max, alpha=0.3)
            axes[1, 0].set_xlabel('Position')
            axes[1, 0].set_ylabel('Variance')
            axes[1, 0].set_title(f'{last_layer} Pre-Attention')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Last layer post-attention
        if 'post_attention' in rmsnorm_data[last_layer]:
            pos_var_max = rmsnorm_data[last_layer]['post_attention']['position_variance_max'].cpu().numpy()
            pos_var_min = rmsnorm_data[last_layer]['post_attention']['position_variance_min'].cpu().numpy()
            axes[1, 1].plot(positions, pos_var_max, 'b-', label='Var Max', linewidth=2)
            axes[1, 1].plot(positions, pos_var_min, 'r-', label='Var Min', linewidth=2)
            axes[1, 1].fill_between(positions, pos_var_min, pos_var_max, alpha=0.3)
            axes[1, 1].set_xlabel('Position')
            axes[1, 1].set_ylabel('Variance')
            axes[1, 1].set_title(f'{last_layer} Post-Attention')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle('RMSNorm Input Range by Position', fontsize=14, y=1.00)
        plt.tight_layout()
        plt.savefig(output_dir / 'rmsnorm_position_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()


def save_analysis_results(
    rmsnorm_data: Dict[str, Dict[str, Dict[str, Any]]],
    analysis_results: Dict[str, Any],
    output_dir: Path
) -> None:
    """
    Save RMSNorm analysis results to files.

    Args:
        rmsnorm_data: Raw RMSNorm input data
        analysis_results: Statistical analysis results
        output_dir: Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw data as pickle (excluding large tensors)
    data_to_save = {}
    for layer_name, layer_data in rmsnorm_data.items():
        data_to_save[layer_name] = {}
        for position, stats in layer_data.items():
            # Exclude raw_tensor from pickle to save space
            data_to_save[layer_name][position] = {
                k: v for k, v in stats.items() if k != 'raw_tensor'
            }

    with open(output_dir / 'rmsnorm_analysis.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)

    # Convert to JSON-serializable format
    def convert_to_json_serializable(obj):
        """Recursively convert tensors and arrays to JSON-serializable format."""
        if isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist() if hasattr(obj, 'tolist') else float(obj)
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

    with open(output_dir / 'rmsnorm_statistics.json', 'w') as f:
        json.dump(json_results, f, indent=2)


def run_complete_rmsnorm_analysis(
    model,
    input_ids: torch.Tensor,
    output_dir: Path,
    attention_mask: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Run complete RMSNorm input range analysis pipeline.

    Args:
        model: The transformer model
        input_ids: Input token IDs
        output_dir: Directory to save results
        attention_mask: Optional attention mask

    Returns:
        Complete analysis results
    """
    print("Extracting RMSNorm inputs across all layers...")
    rmsnorm_data = extract_rmsnorm_inputs(model, input_ids, attention_mask)

    print("Analyzing RMSNorm input patterns...")
    analysis_results = analyze_rmsnorm_patterns(rmsnorm_data)

    print("Generating visualizations...")
    visualize_rmsnorm_analysis(rmsnorm_data, analysis_results, output_dir)

    print("Saving results...")
    save_analysis_results(rmsnorm_data, analysis_results, output_dir)

    # Print summary
    print(f"\n{'='*60}")
    print(f"RMSNORM VARIANCE ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Note: Variance = (1/d) * sum(x_i^2)")
    print(f"      This is the term RMSNorm uses before sqrt")

    overall = analysis_results['overall_stats']
    print(f"\nOverall Statistics (across all layers and positions):")
    print(f"  Variance Maximum: {overall['variance_max']:.6f}")
    print(f"  Variance Minimum: {overall['variance_min']:.6f}")
    print(f"  Variance Range:   {overall['variance_range']:.6f}")

    pre_attn = analysis_results['pre_attention_stats']
    print(f"\nPre-Attention RMSNorm (input_layernorm):")
    print(f"  Variance Maximum: {pre_attn['variance_max']:.6f}")
    print(f"  Variance Minimum: {pre_attn['variance_min']:.6f}")
    print(f"  Variance Range:   {pre_attn['variance_range']:.6f}")

    post_attn = analysis_results['post_attention_stats']
    print(f"\nPost-Attention RMSNorm (post_attention_layernorm):")
    print(f"  Variance Maximum: {post_attn['variance_max']:.6f}")
    print(f"  Variance Minimum: {post_attn['variance_min']:.6f}")
    print(f"  Variance Range:   {post_attn['variance_range']:.6f}")

    # Layer-wise breakdown
    if 'layer_breakdown' in analysis_results:
        print(f"\n{'='*60}")
        print(f"LAYER-WISE BREAKDOWN")
        print(f"{'='*60}")

        layer_stats = analysis_results['layer_breakdown']
        sorted_layers = sorted(layer_stats.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x else 0)

        for layer_name in sorted_layers[:5]:  # Show first 5 layers
            layer_idx = layer_name.split('_')[1] if '_' in layer_name else layer_name
            print(f"\nLayer {layer_idx}:")

            if 'pre_attention' in layer_stats[layer_name]:
                pre = layer_stats[layer_name]['pre_attention']
                print(f"  Pre-Attention:  var_max={pre['global_variance_max']:.6f}, var_min={pre['global_variance_min']:.6f}, range={pre['global_variance_range']:.6f}")

            if 'post_attention' in layer_stats[layer_name]:
                post = layer_stats[layer_name]['post_attention']
                print(f"  Post-Attention: var_max={post['global_variance_max']:.6f}, var_min={post['global_variance_min']:.6f}, range={post['global_variance_range']:.6f}")

        if len(sorted_layers) > 5:
            print(f"\n... and {len(sorted_layers) - 5} more layers")

    print(f"\n{'='*60}")

    return analysis_results
