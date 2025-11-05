import os
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
import logging
from utils.hadamard_utils import get_hadK

logger = logging.getLogger(__name__)


def load_model(
    model_name: str, cache_dir: str = "model_cache", force_local: bool = False
) -> tuple[LlamaForCausalLM, LlamaTokenizer]:
    """
    Load the model and tokenizer safely, using local cache if available.

    Args:
        model_name: Name/path of the model to load
        cache_dir: Directory to cache the model
    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token and not force_local:
            raise ValueError("Please set the HF_TOKEN environment variable")

        os.makedirs(cache_dir, exist_ok=True)

        tokenizer = LlamaTokenizer.from_pretrained(
            model_name,
            token=hf_token if not force_local else None,
            cache_dir=cache_dir,
            local_files_only=force_local,
        )
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            token=hf_token if not force_local else None,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory={0: "24GiB"},
            cache_dir=cache_dir,
            local_files_only=force_local,
        )
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


class ActivationHook:
    def __init__(self):
        self.activations = defaultdict(list)
        self.hooks = []

    def hook_fn(self, name):
        def hook(module, input, output):
            # Convert output to float32 for analysis
            if isinstance(output, torch.Tensor):
                tensor = output.detach().float()
            else:
                tensor = output[0].detach().float()

            min_val = tensor.min().item()
            max_val = tensor.max().item()
            num_elements = tensor.numel()

            # Store sample of values for distribution plotting
            sample = tensor.flatten().cpu().numpy()

            self.activations[name].append(
                {
                    "min": min_val,
                    "max": max_val,
                    "num_elements": num_elements,
                    "sample": sample,
                }
            )

        return hook

    def register_hooks(self, model):
        # Register hooks for each transformer layer
        for name, module in model.named_modules():
            if "layers" in name:
                if any(subname in name for subname in ["self_attn", "mlp"]):
                    hook = self.hook_fn(name)
                    handle = module.register_forward_hook(hook)
                    self.hooks.append(handle)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


def plot_activation_distributions(activations, output_dir="activation_plots"):
    os.makedirs(output_dir, exist_ok=True)

    # Plot distributions for each layer
    for layer_name, layer_activations in activations.items():
        plt.figure(figsize=(10, 6))

        # Combine samples from all inputs
        all_samples = np.concatenate([act["sample"] for act in layer_activations])

        # Plot distribution
        sns.histplot(all_samples, bins=50)
        plt.title(f"Activation Distribution: {layer_name}")
        plt.xlabel("Activation Value")
        plt.ylabel("Count")

        # Add min/max annotations
        min_vals = [act["min"] for act in layer_activations]
        max_vals = [act["max"] for act in layer_activations]
        plt.annotate(
            f"Min: {np.mean(min_vals):.2f}\nMax: {np.mean(max_vals):.2f}",
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            bbox=dict(facecolor="white", alpha=0.8),
            ha="right",
            va="top",
        )

        plt.savefig(os.path.join(output_dir, f'{layer_name.replace(".", "_")}.png'))
        plt.close()


class RangeCollector:
    def __init__(self):
        self.min_vals = {}
        self.max_vals = {}
        self.handles = []

    def hook_fn(self, name):
        def hook(module, input, output):
            t = output[0] if isinstance(output, tuple) else output
            t = t.detach().float()
            cur_min = t.min()
            cur_max = t.max()
            if name not in self.min_vals:
                self.min_vals[name] = cur_min
                self.max_vals[name] = cur_max
            else:
                self.min_vals[name] = torch.min(self.min_vals[name], cur_min)
                self.max_vals[name] = torch.max(self.max_vals[name], cur_max)

        return hook

    def register_hooks(self, model):
        for name, m in model.named_modules():
            if "layers" in name and any(k in name for k in ["self_attn", "mlp"]):
                h = m.register_forward_hook(self.hook_fn(name))
                self.handles.append(h)

    def remove_hooks(self):
        for h in self.handles:
            h.remove()

    def get_min_max_ranges(self):
        # Return a dictionary of {name: (min_val, max_val)} pairs
        return {
            name: (self.min_vals[name].item(), self.max_vals[name].item())
            for name in self.min_vals
        }

class RangeCollectorPerDim:
    """
    Collects min/max ranges for each dimension (feature) in activations.

    Unlike RangeCollector which tracks global min/max across all elements,
    this class tracks min/max for each individual dimension/feature, which is
    useful for per-channel quantization or dimension-specific analysis.
    """

    def __init__(self, device="cuda"):
        self.min_vals = {}  # {layer_name: torch.Tensor of shape [hidden_dim]}
        self.max_vals = {}  # {layer_name: torch.Tensor of shape [hidden_dim]}
        self.handles = []
        self.device = device

    def hook_fn(self, name):
        def hook(module, input, output):
            # Extract tensor from output
            t = output[0] if isinstance(output, tuple) else output
            t = t.detach().float().to(self.device)

            # Get per-dimension min/max across batch and sequence dimensions
            # t shape is typically [batch_size, seq_len, hidden_dim]
            if t.dim() >= 2:
                # Flatten all dimensions except the last (feature dimension)
                t_flat = t.view(-1, t.shape[-1])  # [batch*seq, hidden_dim]

                # Get min/max per dimension
                cur_min = t_flat.min(dim=0)[0]  # [hidden_dim]
                cur_max = t_flat.max(dim=0)[0]  # [hidden_dim]

                if name not in self.min_vals:
                    self.min_vals[name] = cur_min
                    self.max_vals[name] = cur_max
                else:
                    # Update per-dimension min/max
                    self.min_vals[name] = torch.min(self.min_vals[name], cur_min)
                    self.max_vals[name] = torch.max(self.max_vals[name], cur_max)
            else:
                # Handle 1D case (shouldn't happen in typical transformer layers)
                logger.warning(f"Unexpected tensor shape in {name}: {t.shape}")

        return hook

    def register_hooks(self, model):
        """Register hooks on transformer layers for activation collection"""
        for name, m in model.named_modules():
            if (
                "module" in name
                or "rotary_emb" in name
                or "q_proj.1" in name
                or "k_proj.1" in name
                or "0.linear_layer" in name
                or "proj_module" in name
                or "linear_layer" in name
                or "act_fn" in name
                or "gate_proj" in name
                or "up_proj" in name
                # or any(f"{i}.mlp" in name for i in range(100))  # Skip {number}.mlp
                # or any(f"{i}.self_attn" in name for i in range(100))  # Skip {number}.self_attn
            ):
                continue
            if "layers" in name and any(k in name for k in ["self_attn", "mlp"]):
                h = m.register_forward_hook(self.hook_fn(name))
                self.handles.append(h)

    def remove_hooks(self):
        """Remove all registered hooks"""
        for h in self.handles:
            h.remove()
        self.handles = []

    def get_per_dim_ranges(self):
        """
        Get per-dimension min/max ranges.

        Returns:
            dict: {layer_name: {'min': torch.Tensor, 'max': torch.Tensor}}
                  where tensors have shape [hidden_dim]
        """
        return {
            name: {
                'min': self.min_vals[name].clone(),
                'max': self.max_vals[name].clone()
            }
            for name in self.min_vals
        }

    def get_min_max_ranges(self):
        """
        Get global min/max ranges (for compatibility with existing code).

        Returns:
            dict: {layer_name: (global_min, global_max)} pairs
        """
        return {
            name: (self.min_vals[name].min().item(), self.max_vals[name].max().item())
            for name in self.min_vals
        }

    def get_per_dim_ranges_cpu(self):
        """
        Get per-dimension ranges as CPU numpy arrays.

        Returns:
            dict: {layer_name: {'min': np.ndarray, 'max': np.ndarray}}
        """
        return {
            name: {
                'min': self.min_vals[name].cpu().numpy(),
                'max': self.max_vals[name].cpu().numpy()
            }
            for name in self.min_vals
        }

    def get_per_dim_ranges_filtered(self, threshold=8.0, criterion='range', include_indices=True):
        """
        Get per-dimension ranges filtered to show only outlier dimensions outside threshold.

        This method is useful for identifying problematic dimensions in quantization that have
        extreme values outside acceptable ranges.

        Args:
            threshold (float): Absolute threshold value for filtering
            criterion (str): Filtering criterion:
                - 'max_abs': Filter dimensions where max(|min|, |max|) > threshold
                - 'range': Filter dimensions where (max - min) > threshold
            include_indices (bool): If True, include dimension indices in output

        Returns:
            dict: {layer_name: filtered_data} where filtered_data contains:
                - 'min': filtered min values (torch.Tensor or dict with indices)
                - 'max': filtered max values (torch.Tensor or dict with indices)
                - 'outlier_count': number of outlier dimensions
                - 'total_count': total number of dimensions
                - 'outlier_ratio': ratio of outliers (outlier_count / total_count)
        """
        if not self.min_vals:
            return {}

        filtered_results = {}

        for layer_name in self.min_vals:
            min_vals = self.min_vals[layer_name]
            max_vals = self.max_vals[layer_name]

            # Apply filtering criterion
            if criterion == 'max_abs':
                # Filter where max(|min|, |max|) > threshold
                mask = (torch.maximum(torch.abs(min_vals), torch.abs(max_vals)) > threshold)
            elif criterion == 'range':
                # Filter where (max - min) > threshold
                mask = ((max_vals - min_vals) > threshold)
            else:
                raise ValueError(f"Unknown criterion: {criterion}. "
                               f"Use 'max_abs', 'range', 'min_abs', 'max_only', or 'min_only'")

            # Get outlier indices
            outlier_indices = torch.where(mask)[0]
            outlier_count = outlier_indices.numel()
            total_count = min_vals.numel()

            if include_indices and outlier_count > 0:
                # Return as dict with indices
                filtered_min = {
                    int(idx): float(min_vals[idx])
                    for idx in outlier_indices
                }
                filtered_max = {
                    int(idx): float(max_vals[idx])
                    for idx in outlier_indices
                }
            else:
                # Return as tensors (empty if no outliers)
                filtered_min = min_vals[mask].clone() if outlier_count > 0 else torch.tensor([])
                filtered_max = max_vals[mask].clone() if outlier_count > 0 else torch.tensor([])

            filtered_results[layer_name] = {
                'min': filtered_min,
                'max': filtered_max,
                'outlier_indices': outlier_indices.cpu().numpy() if outlier_count > 0 else [],
                'outlier_count': outlier_count,
                'total_count': total_count,
                'outlier_ratio': float(outlier_count) / total_count if total_count > 0 else 0.0,
                'criterion': criterion,
                'threshold': threshold
            }

        return filtered_results

    def get_per_dim_ranges_filtered_cpu(self, threshold=5.0, criterion='max_abs', include_indices=False):
        """
        CPU version of get_per_dim_ranges_filtered returning numpy arrays.

        Args:
            threshold (float): Absolute threshold value for filtering
            criterion (str): Filtering criterion (see get_per_dim_ranges_filtered)
            include_indices (bool): If True, include dimension indices in output

        Returns:
            dict: {layer_name: filtered_data} with numpy arrays/dicts
        """
        if not self.min_vals:
            return {}

        filtered_results = {}

        for layer_name in self.min_vals:
            min_vals = self.min_vals[layer_name].cpu().numpy()
            max_vals = self.max_vals[layer_name].cpu().numpy()

            # Apply filtering criterion
            if criterion == 'max_abs':
                mask = (np.maximum(np.abs(min_vals), np.abs(max_vals)) > threshold)
            elif criterion == 'range':
                mask = ((max_vals - min_vals) > threshold)
            elif criterion == 'min_abs':
                mask = (np.minimum(np.abs(min_vals), np.abs(max_vals)) > threshold)
            elif criterion == 'max_only':
                mask = (max_vals > threshold)
            elif criterion == 'min_only':
                mask = (min_vals < -threshold)
            else:
                raise ValueError(f"Unknown criterion: {criterion}")

            # Get outlier indices
            outlier_indices = np.where(mask)[0]
            outlier_count = len(outlier_indices)
            total_count = len(min_vals)

            if include_indices and outlier_count > 0:
                # Return as dict with indices
                filtered_min = {
                    int(idx): float(min_vals[idx])
                    for idx in outlier_indices
                }
                filtered_max = {
                    int(idx): float(max_vals[idx])
                    for idx in outlier_indices
                }
            else:
                # Return as numpy arrays
                filtered_min = min_vals[mask] if outlier_count > 0 else np.array([])
                filtered_max = max_vals[mask] if outlier_count > 0 else np.array([])

            filtered_results[layer_name] = {
                'min': filtered_min,
                'max': filtered_max,
                'outlier_indices': outlier_indices,
                'outlier_count': outlier_count,
                'total_count': total_count,
                'outlier_ratio': float(outlier_count) / total_count if total_count > 0 else 0.0,
                'criterion': criterion,
                'threshold': threshold
            }

        return filtered_results

    def save_per_dim_ranges(self, output_path):
        """
        Save per-dimension ranges to a file.

        Args:
            output_path (str): Path to save the ranges
        """
        ranges_cpu = self.get_per_dim_ranges_cpu()
        torch.save(ranges_cpu, output_path)
        logger.info(f"Saved per-dimension ranges to {output_path}")

    def load_per_dim_ranges(self, input_path):
        """
        Load per-dimension ranges from a file.

        Args:
            input_path (str): Path to load the ranges from
        """
        ranges_cpu = torch.load(input_path, map_location='cpu')

        self.min_vals = {}
        self.max_vals = {}

        for name, ranges in ranges_cpu.items():
            self.min_vals[name] = torch.from_numpy(ranges['min']).to(self.device)
            self.max_vals[name] = torch.from_numpy(ranges['max']).to(self.device)

        logger.info(f"Loaded per-dimension ranges from {input_path}")

    def get_stats(self):
        """
        Get statistics about the collected ranges.

        Returns:
            dict: Statistics including layer count, dimension info, etc.
        """
        if not self.min_vals:
            return {"layers": 0, "total_dims": 0}

        stats = {
            "layers": len(self.min_vals),
            "layer_names": list(self.min_vals.keys()),
            "dimensions_per_layer": {},
            "total_dims": 0
        }

        for name in self.min_vals:
            dim_count = self.min_vals[name].numel()
            stats["dimensions_per_layer"][name] = dim_count
            stats["total_dims"] += dim_count

        return stats

    def log_filtered_ranges(self, threshold=5.0, criterion='max_abs',
                           logger=None, title=None, show_indices=True,
                           summary_only=False, include_recommendations=True):
        """
        Convenience method to filter and log outlier dimensions in one call.

        Args:
            threshold: Absolute threshold for outlier detection
            criterion: Filtering criterion ('max_abs', 'range', etc.)
            logger: Logger instance (uses module logger if None)
            title: Title for log output (auto-generated if None)
            show_indices: Whether to show dimension indices
            summary_only: If True, only show summary; if False, show detailed results
            include_recommendations: Whether to include quantization recommendations

        Example:
            collector.log_filtered_ranges(threshold=10.0, criterion='max_abs')
            collector.log_filtered_ranges(threshold=5.0, summary_only=True)
        """
        # Get filtered results
        filtered_results = self.get_per_dim_ranges_filtered_cpu(
            threshold=threshold, criterion=criterion, include_indices=show_indices
        )

        if not filtered_results:
            if logger is None:
                import logging
                logger = logging.getLogger(__name__)
            logger.info(f"No outlier dimensions found with threshold={threshold}, criterion={criterion}")
            return

        # Generate title if not provided
        if title is None:
            title = f"OUTLIER DIMENSIONS (threshold={threshold}, criterion={criterion})"

        if summary_only:
            # Import here to avoid circular imports
            log_filtered_ranges_summary(filtered_results, logger, include_recommendations)
        else:
            # Import here to avoid circular imports
            log_filtered_ranges(filtered_results, logger, title, show_indices)

            # Also show summary with recommendations
            if include_recommendations:
                log_filtered_ranges_summary(filtered_results, logger, include_recommendations)

    def log_threshold_comparison(self, thresholds=[1.0, 2.0, 5.0, 10.0, 20.0],
                                criteria=['max_abs', 'range'], logger=None):
        """
        Convenience method to log threshold comparison for outlier detection.

        Args:
            thresholds: List of threshold values to compare
            criteria: List of criteria to compare
            logger: Logger instance (uses module logger if None)

        Example:
            collector.log_threshold_comparison()
            collector.log_threshold_comparison(thresholds=[1.0, 5.0, 10.0])
        """
        # Import here to avoid circular imports
        log_filtered_ranges_comparison(self, thresholds, criteria, logger)


def analyze_per_dim_ranges(range_collector: RangeCollectorPerDim, output_file=None):
    """
    Analyze per-dimension ranges and provide quantization insights.

    Args:
        range_collector: RangeCollectorPerDim instance
        output_file: Optional path to save analysis results

    Returns:
        dict: Analysis results with statistics and recommendations
    """
    import numpy as np

    ranges = range_collector.get_per_dim_ranges_cpu()
    if not ranges:
        logger.warning("No ranges collected")
        return {}

    analysis = {
        "layers": {},
        "global_stats": {},
        "recommendations": []
    }

    all_ranges = []
    all_min_vals = []
    all_max_vals = []

    # Analyze each layer
    for layer_name, layer_ranges in ranges.items():
        min_vals = layer_ranges['min']
        max_vals = layer_ranges['max']
        layer_range_spans = max_vals - min_vals

        layer_analysis = {
            "dim_count": len(min_vals),
            "min_range": float(np.min(layer_range_spans)),
            "max_range": float(np.max(layer_range_spans)),
            "mean_range": float(np.mean(layer_range_spans)),
            "std_range": float(np.std(layer_range_spans)),
            "global_min": float(np.min(min_vals)),
            "global_max": float(np.max(max_vals)),
            "outlier_dims": {
                "large_range": int(np.sum(layer_range_spans > np.mean(layer_range_spans) + 2*np.std(layer_range_spans))),
                "negative_heavy": int(np.sum(np.abs(min_vals) > np.abs(max_vals))),
                "positive_heavy": int(np.sum(np.abs(max_vals) > np.abs(min_vals)))
            }
        }

        analysis["layers"][layer_name] = layer_analysis
        all_ranges.extend(layer_range_spans)
        all_min_vals.extend(min_vals)
        all_max_vals.extend(max_vals)

    # Global statistics
    all_ranges = np.array(all_ranges)
    all_min_vals = np.array(all_min_vals)
    all_max_vals = np.array(all_max_vals)

    analysis["global_stats"] = {
        "total_dimensions": len(all_ranges),
        "range_stats": {
            "min": float(np.min(all_ranges)),
            "max": float(np.max(all_ranges)),
            "mean": float(np.mean(all_ranges)),
            "std": float(np.std(all_ranges)),
            "p95": float(np.percentile(all_ranges, 95)),
            "p99": float(np.percentile(all_ranges, 99))
        },
        "activation_stats": {
            "global_min": float(np.min(all_min_vals)),
            "global_max": float(np.max(all_max_vals)),
            "symmetry_ratio": float(np.mean(np.abs(all_min_vals)) / np.mean(np.abs(all_max_vals)))
        }
    }

    # Generate recommendations
    mean_range = analysis["global_stats"]["range_stats"]["mean"]
    large_range_threshold = mean_range + 2 * analysis["global_stats"]["range_stats"]["std"]

    large_range_count = np.sum(all_ranges > large_range_threshold)
    total_dims = len(all_ranges)

    if large_range_count / total_dims > 0.1:
        analysis["recommendations"].append("High range variance detected - consider per-channel quantization")

    if analysis["global_stats"]["activation_stats"]["symmetry_ratio"] > 2.0:
        analysis["recommendations"].append("Asymmetric activations - consider asymmetric quantization")
    elif analysis["global_stats"]["activation_stats"]["symmetry_ratio"] < 0.5:
        analysis["recommendations"].append("Asymmetric activations (positive-heavy) - consider asymmetric quantization")

    # Save analysis if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write("🔍 PER-DIMENSION RANGE ANALYSIS\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Total layers: {len(analysis['layers'])}\n")
            f.write(f"Total dimensions: {analysis['global_stats']['total_dimensions']}\n\n")

            f.write("GLOBAL RANGE STATISTICS\n")
            f.write("-" * 30 + "\n")
            for stat, value in analysis['global_stats']['range_stats'].items():
                f.write(f"{stat:>10}: {value:.6f}\n")

            f.write("\nACTIVATION STATISTICS\n")
            f.write("-" * 30 + "\n")
            for stat, value in analysis['global_stats']['activation_stats'].items():
                f.write(f"{stat:>15}: {value:.6f}\n")

            f.write("\nLAYER-WISE ANALYSIS\n")
            f.write("-" * 30 + "\n")
            for layer_name, layer_stats in analysis['layers'].items():
                short_name = ".".join(layer_name.split(".")[-2:]) if "." in layer_name else layer_name
                f.write(f"\n{short_name}:\n")
                f.write(f"  Dimensions: {layer_stats['dim_count']}\n")
                f.write(f"  Range span: {layer_stats['min_range']:.4f} - {layer_stats['max_range']:.4f} (μ={layer_stats['mean_range']:.4f})\n")
                f.write(f"  Global range: [{layer_stats['global_min']:.4f}, {layer_stats['global_max']:.4f}]\n")
                f.write(f"  Outlier dims: {layer_stats['outlier_dims']['large_range']} large ranges\n")

            f.write("\nRECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")
            for rec in analysis['recommendations']:
                f.write(f"• {rec}\n")

            f.write("\n" + "=" * 60 + "\n")

        logger.info(f"Saved per-dimension analysis to {output_file}")

    return analysis


def compare_per_dim_ranges(range_collector1: RangeCollectorPerDim,
                          range_collector2: RangeCollectorPerDim,
                          output_file=None):
    """
    Compare two per-dimension range collectors (e.g., before/after rotation).

    Args:
        range_collector1: First RangeCollectorPerDim (e.g., original)
        range_collector2: Second RangeCollectorPerDim (e.g., rotated)
        output_file: Optional path to save comparison results

    Returns:
        dict: Comparison results
    """
    import numpy as np

    ranges1 = range_collector1.get_per_dim_ranges_cpu()
    ranges2 = range_collector2.get_per_dim_ranges_cpu()

    if not ranges1 or not ranges2:
        logger.warning("One or both range collectors are empty")
        return {}

    comparison = {
        "common_layers": [],
        "layer_comparisons": {},
        "global_comparison": {}
    }

    # Find common layers
    common_layers = set(ranges1.keys()) & set(ranges2.keys())
    comparison["common_layers"] = sorted(common_layers)

    if not common_layers:
        logger.warning("No common layers found between collectors")
        return comparison

    # Compare each layer
    for layer_name in common_layers:
        r1 = ranges1[layer_name]
        r2 = ranges2[layer_name]

        range_spans1 = r1['max'] - r1['min']
        range_spans2 = r2['max'] - r2['min']

        layer_comp = {
            "range_reduction": {
                "mean": float(np.mean(range_spans1) - np.mean(range_spans2)),
                "median": float(np.median(range_spans1) - np.median(range_spans2)),
                "max": float(np.max(range_spans1) - np.max(range_spans2)),
                "percentage": float((np.mean(range_spans1) - np.mean(range_spans2)) / np.mean(range_spans1) * 100)
            },
            "dimensions_improved": int(np.sum(range_spans2 < range_spans1)),
            "dimensions_worsened": int(np.sum(range_spans2 > range_spans1)),
            "total_dimensions": len(range_spans1)
        }

        comparison["layer_comparisons"][layer_name] = layer_comp

    # Global comparison
    all_improvements = [comp["range_reduction"]["percentage"]
                       for comp in comparison["layer_comparisons"].values()]

    comparison["global_comparison"] = {
        "average_range_reduction_pct": float(np.mean(all_improvements)),
        "layers_improved": int(np.sum(np.array(all_improvements) > 0)),
        "layers_worsened": int(np.sum(np.array(all_improvements) < 0)),
        "total_layers": len(all_improvements)
    }

    # Save comparison if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write("🔍 PER-DIMENSION RANGE COMPARISON\n")
            f.write("=" * 60 + "\n\n")

            gc = comparison["global_comparison"]
            f.write("GLOBAL COMPARISON SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average range reduction: {gc['average_range_reduction_pct']:.2f}%\n")
            f.write(f"Layers improved: {gc['layers_improved']}/{gc['total_layers']}\n")
            f.write(f"Layers worsened: {gc['layers_worsened']}/{gc['total_layers']}\n\n")

            f.write("LAYER-WISE COMPARISON\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Layer':<30} {'Range Reduction':<15} {'Dims Improved':<15}\n")
            f.write("-" * 60 + "\n")

            for layer_name, layer_comp in comparison["layer_comparisons"].items():
                short_name = ".".join(layer_name.split(".")[-2:]) if "." in layer_name else layer_name
                f.write(f"{short_name:<30} {layer_comp['range_reduction']['percentage']:>13.2f}% "
                       f"{layer_comp['dimensions_improved']:>6}/{layer_comp['total_dimensions']:<6}\n")

            f.write("\n" + "=" * 60 + "\n")

        logger.info(f"Saved per-dimension comparison to {output_file}")

    return comparison


def analyze_filtered_outliers(range_collector: RangeCollectorPerDim,
                              threshold=5.0,
                              criterion='max_abs',
                              output_file=None):
    """
    Analyze filtered outlier dimensions and provide detailed insights.

    Args:
        range_collector: RangeCollectorPerDim instance
        threshold: Absolute threshold for outlier detection
        criterion: Filtering criterion to use
        output_file: Optional path to save analysis results

    Returns:
        dict: Analysis results with outlier statistics and recommendations
    """
    import numpy as np

    filtered_results = range_collector.get_per_dim_ranges_filtered_cpu(
        threshold=threshold, criterion=criterion, include_indices=True
    )

    if not filtered_results:
        logger.warning("No filtered results available")
        return {}

    analysis = {
        "summary": {
            "threshold": threshold,
            "criterion": criterion,
            "total_layers": len(filtered_results),
            "layers_with_outliers": 0,
            "total_outlier_dims": 0,
            "total_dims": 0
        },
        "layer_analysis": {},
        "global_outlier_stats": {},
        "recommendations": []
    }

    all_outlier_ratios = []
    layers_with_outliers = []

    # Analyze each layer
    for layer_name, results in filtered_results.items():
        outlier_count = results['outlier_count']
        total_count = results['total_count']
        outlier_ratio = results['outlier_ratio']

        analysis["layer_analysis"][layer_name] = {
            "outlier_count": outlier_count,
            "total_count": total_count,
            "outlier_ratio": outlier_ratio,
            "outlier_indices": results['outlier_indices'].tolist() if len(results['outlier_indices']) > 0 else []
        }

        # Add extreme values if there are outliers
        if outlier_count > 0:
            if isinstance(results['min'], dict):
                min_vals = list(results['min'].values())
                max_vals = list(results['max'].values())
                analysis["layer_analysis"][layer_name].update({
                    "extreme_min": float(np.min(min_vals)),
                    "extreme_max": float(np.max(max_vals)),
                    "mean_outlier_range": float(np.mean([max_vals[i] - min_vals[i] for i in range(len(min_vals))]))
                })
            layers_with_outliers.append(layer_name)

        analysis["summary"]["total_outlier_dims"] += outlier_count
        analysis["summary"]["total_dims"] += total_count
        all_outlier_ratios.append(outlier_ratio)

    analysis["summary"]["layers_with_outliers"] = len(layers_with_outliers)

    # Global statistics
    if all_outlier_ratios:
        analysis["global_outlier_stats"] = {
            "mean_outlier_ratio": float(np.mean(all_outlier_ratios)),
            "max_outlier_ratio": float(np.max(all_outlier_ratios)),
            "std_outlier_ratio": float(np.std(all_outlier_ratios)),
            "layers_above_1pct": int(np.sum(np.array(all_outlier_ratios) > 0.01)),
            "layers_above_5pct": int(np.sum(np.array(all_outlier_ratios) > 0.05))
        }

    # Generate recommendations
    total_outlier_ratio = analysis["summary"]["total_outlier_dims"] / analysis["summary"]["total_dims"]

    if total_outlier_ratio > 0.05:
        analysis["recommendations"].append(f"High outlier rate ({total_outlier_ratio:.1%}) - consider outlier-aware quantization")

    if len(layers_with_outliers) > len(filtered_results) * 0.5:
        analysis["recommendations"].append("Majority of layers have outliers - systematic issue may exist")

    if analysis["global_outlier_stats"].get("layers_above_5pct", 0) > 0:
        analysis["recommendations"].append("Some layers have >5% outlier dimensions - focus quantization attention here")

    # Save analysis if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write("🎯 FILTERED OUTLIER DIMENSION ANALYSIS\n")
            f.write("=" * 60 + "\n\n")

            # Summary
            s = analysis["summary"]
            f.write("ANALYSIS SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Threshold: {s['threshold']} (criterion: {s['criterion']})\n")
            f.write(f"Total layers: {s['total_layers']}\n")
            f.write(f"Layers with outliers: {s['layers_with_outliers']}\n")
            f.write(f"Total outlier dimensions: {s['total_outlier_dims']:,}/{s['total_dims']:,} ({total_outlier_ratio:.2%})\n\n")

            # Global stats
            if "global_outlier_stats" in analysis:
                gs = analysis["global_outlier_stats"]
                f.write("GLOBAL OUTLIER STATISTICS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Mean outlier ratio: {gs['mean_outlier_ratio']:.3f}\n")
                f.write(f"Max outlier ratio: {gs['max_outlier_ratio']:.3f}\n")
                f.write(f"Std outlier ratio: {gs['std_outlier_ratio']:.3f}\n")
                f.write(f"Layers >1% outliers: {gs['layers_above_1pct']}\n")
                f.write(f"Layers >5% outliers: {gs['layers_above_5pct']}\n\n")

            # Layer details
            f.write("LAYER-WISE OUTLIER DETAILS\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Layer':<35} {'Outliers':<12} {'Ratio':<8} {'Extreme Range':<15}\n")
            f.write("-" * 60 + "\n")

            # Sort by outlier ratio (highest first)
            sorted_layers = sorted(analysis["layer_analysis"].items(),
                                 key=lambda x: x[1]['outlier_ratio'], reverse=True)

            for layer_name, layer_data in sorted_layers:
                if layer_data['outlier_count'] > 0:
                    short_name = ".".join(layer_name.split(".")[-2:]) if "." in layer_name else layer_name
                    outliers_str = f"{layer_data['outlier_count']}/{layer_data['total_count']}"
                    ratio_str = f"{layer_data['outlier_ratio']:.3f}"

                    if 'extreme_min' in layer_data:
                        range_str = f"[{layer_data['extreme_min']:.2f}, {layer_data['extreme_max']:.2f}]"
                    else:
                        range_str = "N/A"

                    f.write(f"{short_name:<35} {outliers_str:<12} {ratio_str:<8} {range_str:<15}\n")

            # Recommendations
            f.write("\nRECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")
            for rec in analysis["recommendations"]:
                f.write(f"• {rec}\n")

            f.write("\n" + "=" * 60 + "\n")

        logger.info(f"Saved filtered outlier analysis to {output_file}")

    return analysis


def compare_outlier_filtering(range_collector: RangeCollectorPerDim,
                             thresholds=[1.0, 2.0, 5.0, 10.0, 20.0],
                             criteria=['max_abs', 'range', 'max_only'],
                             output_file=None):
    """
    Compare outlier detection across different thresholds and criteria.

    Args:
        range_collector: RangeCollectorPerDim instance
        thresholds: List of threshold values to test
        criteria: List of filtering criteria to test
        output_file: Optional path to save comparison results

    Returns:
        dict: Comparison results across thresholds and criteria
    """
    import numpy as np

    comparison = {
        "thresholds": thresholds,
        "criteria": criteria,
        "results": {},
        "summary": {}
    }

    # Test each combination
    for criterion in criteria:
        comparison["results"][criterion] = {}

        for threshold in thresholds:
            filtered_results = range_collector.get_per_dim_ranges_filtered_cpu(
                threshold=threshold, criterion=criterion
            )

            if not filtered_results:
                continue

            total_outliers = sum(r['outlier_count'] for r in filtered_results.values())
            total_dims = sum(r['total_count'] for r in filtered_results.values())
            layers_with_outliers = sum(1 for r in filtered_results.values() if r['outlier_count'] > 0)

            comparison["results"][criterion][threshold] = {
                "total_outliers": total_outliers,
                "total_dims": total_dims,
                "outlier_ratio": float(total_outliers) / total_dims if total_dims > 0 else 0.0,
                "layers_with_outliers": layers_with_outliers,
                "total_layers": len(filtered_results)
            }

    # Find optimal threshold (highest threshold with reasonable outlier detection)
    for criterion in criteria:
        if criterion not in comparison["results"]:
            continue

        ratios = [(t, r['outlier_ratio']) for t, r in comparison["results"][criterion].items()]
        ratios.sort(key=lambda x: x[0])  # Sort by threshold

        # Find threshold where outlier ratio drops below 5%
        optimal_threshold = thresholds[-1]  # Default to highest
        for threshold, ratio in ratios:
            if ratio < 0.05:  # Less than 5% outliers
                optimal_threshold = threshold
                break

        comparison["summary"][criterion] = {
            "optimal_threshold": optimal_threshold,
            "outlier_ratios": {t: r for t, r in ratios}
        }

    # Save comparison if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write("📊 OUTLIER FILTERING THRESHOLD COMPARISON\n")
            f.write("=" * 60 + "\n\n")

            f.write("COMPARISON MATRIX\n")
            f.write("-" * 60 + "\n")

            # Header
            f.write(f"{'Criterion':<12} {'Threshold':<10}")
            f.write(f" {'Outliers':<12} {'Ratio':<8} {'Layers':<8}\n")
            f.write("-" * 60 + "\n")

            for criterion in criteria:
                if criterion not in comparison["results"]:
                    continue

                for threshold in thresholds:
                    if threshold not in comparison["results"][criterion]:
                        continue

                    r = comparison["results"][criterion][threshold]
                    f.write(f"{criterion:<12} {threshold:<10.1f}")
                    f.write(f" {r['total_outliers']:>8,}/{r['total_dims']:<8,}")
                    f.write(f" {r['outlier_ratio']:<8.3f}")
                    f.write(f" {r['layers_with_outliers']}/{r['total_layers']}\n")
                f.write("\n")

            # Optimal thresholds
            f.write("OPTIMAL THRESHOLDS (for <5% outlier ratio)\n")
            f.write("-" * 40 + "\n")
            for criterion, summary in comparison["summary"].items():
                f.write(f"{criterion:<12}: {summary['optimal_threshold']}\n")

            f.write("\n" + "=" * 60 + "\n")

        logger.info(f"Saved outlier filtering comparison to {output_file}")

    return comparison


def log_filtered_ranges(filtered_results, logger=None, title="FILTERED OUTLIER DIMENSIONS",
                       show_indices=True, max_indices_per_layer=10, sort_by='outlier_ratio'):
    """
    Log filtered per-dimension range results in a readable format.

    Args:
        filtered_results: Output from get_per_dim_ranges_filtered() or get_per_dim_ranges_filtered_cpu()
        logger: Logger instance to use (uses module logger if None)
        title: Title for the log output
        show_indices: Whether to show individual outlier dimension indices
        max_indices_per_layer: Maximum number of indices to show per layer
        sort_by: Sort layers by 'outlier_ratio', 'outlier_count', or 'layer_name'

    Example:
        filtered = collector.get_per_dim_ranges_filtered(threshold=5.0, criterion='max_abs')
        log_filtered_ranges(filtered, title="Outliers above |5.0|")
    """
    import numpy as np

    if logger is None:
        import logging
        logger = logging.getLogger(__name__)

    if not filtered_results:
        logger.info("No filtered results to log")
        return

    # Extract configuration from first layer
    sample_layer = next(iter(filtered_results.values()))
    threshold = sample_layer.get('threshold', 'N/A')
    criterion = sample_layer.get('criterion', 'N/A')

    # Calculate totals
    total_outliers = sum(r['outlier_count'] for r in filtered_results.values())
    total_dims = sum(r['total_count'] for r in filtered_results.values())
    layers_with_outliers = sum(1 for r in filtered_results.values() if r['outlier_count'] > 0)

    # Header
    logger.info("=" * 80)
    logger.info(f"{title}")
    logger.info("=" * 80)
    logger.info(f"Configuration: threshold={threshold}, criterion={criterion}")
    logger.info(f"Summary: {total_outliers:,} outlier dims / {total_dims:,} total dims ({total_outliers/total_dims*100:.2f}%)")
    logger.info(f"Layers with outliers: {layers_with_outliers}/{len(filtered_results)}")
    logger.info("-" * 80)

    # Sort layers
    if sort_by == 'outlier_ratio':
        sorted_layers = sorted(filtered_results.items(), key=lambda x: x[1]['outlier_ratio'], reverse=True)
    elif sort_by == 'outlier_count':
        sorted_layers = sorted(filtered_results.items(), key=lambda x: x[1]['outlier_count'], reverse=True)
    else:  # layer_name
        sorted_layers = sorted(filtered_results.items())

    # Log each layer
    for layer_name, results in sorted_layers:
        outlier_count = results['outlier_count']
        total_count = results['total_count']
        outlier_ratio = results['outlier_ratio']

        # Skip layers with no outliers unless specifically requested
        if outlier_count == 0:
            continue

        # Short layer name for display
        short_name = ".".join(layer_name.split(".")[-2:]) if "." in layer_name else layer_name

        # Basic layer info
        logger.info(f"📍 {short_name}")
        logger.info(f"   Outliers: {outlier_count:,}/{total_count:,} ({outlier_ratio*100:.3f}%)")

        # Show extreme values if available
        if 'min' in results and 'max' in results:
            min_vals = results['min']
            max_vals = results['max']

            if isinstance(min_vals, dict) and min_vals:  # Index-based format
                min_values = list(min_vals.values())
                max_values = list(max_vals.values())
                extreme_min = float(np.min(min_values))
                extreme_max = float(np.max(max_values))
                logger.info(f"   Range: [{extreme_min:.4f}, {extreme_max:.4f}]")

                # Show indices if requested
                if show_indices and len(min_vals) > 0:
                    indices = list(min_vals.keys())[:max_indices_per_layer]
                    if len(min_vals) > max_indices_per_layer:
                        indices_str = f"{indices} ... (+{len(min_vals) - max_indices_per_layer} more)"
                    else:
                        indices_str = str(indices)
                    logger.info(f"   Indices: {indices_str}")

                    # Show detailed values for first few indices
                    if len(indices) <= 5:
                        for idx in indices:
                            logger.info(f"     dim[{idx}]: [{min_vals[idx]:.4f}, {max_vals[idx]:.4f}]")

            elif hasattr(min_vals, '__len__') and len(min_vals) > 0:  # Array format
                if hasattr(min_vals, 'min'):  # numpy array
                    extreme_min = float(min_vals.min())
                    extreme_max = float(max_vals.max())
                else:  # torch tensor or list
                    extreme_min = float(np.min(min_vals))
                    extreme_max = float(np.max(max_vals))
                logger.info(f"   Range: [{extreme_min:.4f}, {extreme_max:.4f}]")

                # Show indices from outlier_indices if available
                if show_indices and 'outlier_indices' in results:
                    indices = results['outlier_indices']
                    if hasattr(indices, 'tolist'):
                        indices = indices.tolist()
                    indices = indices[:max_indices_per_layer]
                    if len(results['outlier_indices']) > max_indices_per_layer:
                        indices_str = f"{indices} ... (+{len(results['outlier_indices']) - max_indices_per_layer} more)"
                    else:
                        indices_str = str(indices)
                    logger.info(f"   Indices: {indices_str}")

    logger.info("=" * 80)


def log_filtered_ranges_summary(filtered_results, logger=None, include_recommendations=True):
    """
    Log a concise summary of filtered range results.

    Args:
        filtered_results: Output from get_per_dim_ranges_filtered()
        logger: Logger instance to use (uses module logger if None)
        include_recommendations: Whether to include quantization recommendations
    """
    import numpy as np

    if logger is None:
        import logging
        logger = logging.getLogger(__name__)

    if not filtered_results:
        logger.info("No filtered results to summarize")
        return

    # Extract configuration
    sample_layer = next(iter(filtered_results.values()))
    threshold = sample_layer.get('threshold', 'N/A')
    criterion = sample_layer.get('criterion', 'N/A')

    # Calculate statistics
    total_outliers = sum(r['outlier_count'] for r in filtered_results.values())
    total_dims = sum(r['total_count'] for r in filtered_results.values())
    layers_with_outliers = sum(1 for r in filtered_results.values() if r['outlier_count'] > 0)

    outlier_ratios = [r['outlier_ratio'] for r in filtered_results.values() if r['outlier_count'] > 0]

    logger.info("📊 FILTERED RANGES SUMMARY")
    logger.info("-" * 40)
    logger.info(f"Threshold: {threshold} (criterion: {criterion})")
    logger.info(f"Total outlier dimensions: {total_outliers:,}/{total_dims:,} ({total_outliers/total_dims*100:.2f}%)")
    logger.info(f"Layers with outliers: {layers_with_outliers}/{len(filtered_results)}")

    if outlier_ratios:
        logger.info(f"Outlier ratio stats: mean={np.mean(outlier_ratios)*100:.3f}%, max={np.max(outlier_ratios)*100:.3f}%")

        # Count layers by outlier severity
        high_outlier_layers = sum(1 for r in outlier_ratios if r > 0.05)  # >5%
        medium_outlier_layers = sum(1 for r in outlier_ratios if 0.01 < r <= 0.05)  # 1-5%
        low_outlier_layers = sum(1 for r in outlier_ratios if 0 < r <= 0.01)  # <1%

        logger.info(f"Outlier severity: {high_outlier_layers} high (>5%), {medium_outlier_layers} medium (1-5%), {low_outlier_layers} low (<1%)")

    if include_recommendations:
        logger.info("\n💡 RECOMMENDATIONS:")

        total_outlier_ratio = total_outliers / total_dims if total_dims > 0 else 0

        if total_outlier_ratio > 0.05:
            logger.info(f"   • High outlier rate ({total_outlier_ratio:.1%}) - consider outlier-aware quantization")
        elif total_outlier_ratio > 0.01:
            logger.info(f"   • Moderate outlier rate ({total_outlier_ratio:.1%}) - monitor quantization quality")
        else:
            logger.info(f"   • Low outlier rate ({total_outlier_ratio:.1%}) - standard quantization should work well")

        if layers_with_outliers > len(filtered_results) * 0.5:
            logger.info("   • Majority of layers have outliers - systematic issue may exist")

        if outlier_ratios and max(outlier_ratios) > 0.1:
            logger.info("   • Some layers have >10% outliers - focus calibration on these layers")

        if layers_with_outliers > 0:
            worst_layer = max(filtered_results.items(), key=lambda x: x[1]['outlier_ratio'])
            worst_name = ".".join(worst_layer[0].split(".")[-2:])
            worst_ratio = worst_layer[1]['outlier_ratio']
            logger.info(f"   • Worst layer: {worst_name} ({worst_ratio*100:.2f}% outliers)")


def log_filtered_ranges_comparison(collector: 'RangeCollectorPerDim',
                                  thresholds=[1.0, 2.0, 5.0, 10.0, 20.0],
                                  criteria=['max_abs', 'range'],
                                  logger=None):
    """
    Log a comparison of outlier detection across different thresholds and criteria.

    Args:
        collector: RangeCollectorPerDim instance with collected data
        thresholds: List of threshold values to compare
        criteria: List of criteria to compare
        logger: Logger instance to use
    """
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)

    logger.info("🔍 OUTLIER THRESHOLD COMPARISON")
    logger.info("=" * 60)

    # Header
    logger.info(f"{'Criterion':<12} {'Threshold':<10} {'Outliers':<15} {'Ratio':<8} {'Layers':<10}")
    logger.info("-" * 60)

    for criterion in criteria:
        for threshold in thresholds:
            try:
                filtered_results = collector.get_per_dim_ranges_filtered_cpu(
                    threshold=threshold, criterion=criterion
                )

                if not filtered_results:
                    continue

                total_outliers = sum(r['outlier_count'] for r in filtered_results.values())
                total_dims = sum(r['total_count'] for r in filtered_results.values())
                layers_with_outliers = sum(1 for r in filtered_results.values() if r['outlier_count'] > 0)
                outlier_ratio = total_outliers / total_dims if total_dims > 0 else 0.0

                outliers_str = f"{total_outliers:,}/{total_dims:,}"
                layers_str = f"{layers_with_outliers}/{len(filtered_results)}"

                logger.info(f"{criterion:<12} {threshold:<10.1f} {outliers_str:<15} {outlier_ratio:<8.3f} {layers_str:<10}")

            except Exception as e:
                logger.warning(f"Error with {criterion}@{threshold}: {e}")

        logger.info("")  # Blank line between criteria

    logger.info("=" * 60)

    # Find optimal thresholds
    logger.info("🎯 OPTIMAL THRESHOLDS (for <5% outlier ratio):")
    for criterion in criteria:
        try:
            optimal_threshold = None
            for threshold in sorted(thresholds):
                filtered_results = collector.get_per_dim_ranges_filtered_cpu(
                    threshold=threshold, criterion=criterion
                )
                if not filtered_results:
                    continue

                total_outliers = sum(r['outlier_count'] for r in filtered_results.values())
                total_dims = sum(r['total_count'] for r in filtered_results.values())
                outlier_ratio = total_outliers / total_dims if total_dims > 0 else 0.0

                if outlier_ratio < 0.05:
                    optimal_threshold = threshold
                    break

            if optimal_threshold:
                logger.info(f"   {criterion:<12}: {optimal_threshold}")
            else:
                logger.info(f"   {criterion:<12}: >{max(thresholds)} (high outlier rate)")

        except Exception as e:
            logger.warning(f"Error finding optimal threshold for {criterion}: {e}")


class HistogramAccumulator:
    def __init__(self, layer_ranges: dict, bins=50, device="cuda"):
        """
        layer_ranges: dict[name] = (torch.Tensor global_min, torch.Tensor global_max)
        """
        self.device = device
        self.handles = []
        # pre‑allocate one histogram per layer on GPU
        self.histograms = {}
        self.layer_ranges = {}
        for name, (min_val, max_val) in layer_ranges.items():
            self.layer_ranges[name] = (min_val, max_val)
            bins = 100
            self.histograms[name] = torch.zeros(
                bins, dtype=torch.float32, device=device
            )

    def hook_fn(self, name):
        mn, mx = self.layer_ranges[name]
        ## get integer bins from mn to mx
        bins = 100

        def hook(module, input, output):
            t = output[0] if isinstance(output, tuple) else output
            t = t.detach().float().to(self.device)
            # always use the fixed mn/mx
            hist = torch.histc(t, bins=bins, min=mn, max=mx)
            self.histograms[name] += hist

        return hook

    def register_hooks(self, model):
        for name, m in model.named_modules():
            if (
                "module" in name
                or "rotary_emb" in name
                or "q_proj.1" in name
                or "k_proj.1" in name
                or "0.linear_layer" in name
                or "proj_module" in name
                or "linear_layer" in name
                # or "self_attn.k_proj" in name
                # or "self_attn.q_proj" in name
            ):
                continue
            if name in self.histograms:
                h = m.register_forward_hook(self.hook_fn(name))
                self.handles.append(h)

    def remove_hooks(self):
        for h in self.handles:
            h.remove()

    def get_cpu_histograms(self):
        return {n: h.cpu().numpy() for n, h in self.histograms.items()}


def plot_histograms(histograms, layer_ranges, output_dir="activation_plots"):
    """
    Plot activation histograms gathered by HistogramAccumulator

    Args:
        histograms: Dict mapping layer names to numpy arrays of histogram counts
        layer_ranges: Dict mapping layer names to (min, max) pairs defining histogram ranges
        output_dir: Directory to save plots
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    for layer_name, counts in histograms.items():
        plt.figure(figsize=(10, 6))

        # Get the min/max for this layer
        min_val, max_val = layer_ranges[layer_name]

        # Create bin edges for plotting
        bin_edges = np.linspace(min_val, max_val, len(counts) + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Plot the histogram using bar plot
        plt.bar(bin_centers, counts, width=(max_val - min_val) / len(counts))

        plt.title(f"Activation Distribution: {layer_name}")
        plt.xlabel("Activation Value")
        plt.ylabel("Count")

        # Add min/max annotations
        plt.annotate(
            f"Min: {min_val:.2f}\nMax: {max_val:.2f}",
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            bbox=dict(facecolor="white", alpha=0.8),
            ha="right",
            va="top",
        )

        plt.savefig(os.path.join(output_dir, f'{layer_name.replace(".", "_")}.png'))
        plt.close()


class OutlierStatsCollector:
    def __init__(self, threshold_r, device="cuda"):
        """
        Collector for outlier statistics for activations

        Args:
            threshold_r: The threshold value to consider outliers (activations outside [-r, r])
            device: Device to perform calculations on
        """
        self.threshold_r = threshold_r
        self.device = device
        self.handles = []
        # Statistics dictionaries
        self.outlier_counts = defaultdict(int)
        self.total_counts = defaultdict(int)
        self.outlier_magnitude_sum = defaultdict(float)
        self.nonoutlier_magnitude_sum = defaultdict(float)

    def hook_fn(self, name):
        def hook(module, input, output):
            # Get tensor from output
            if isinstance(output, torch.Tensor):
                t = output.detach().float()
            else:
                t = output[0].detach().float()

            # Move to specified device if needed
            t = t.to(self.device)

            # Create masks for outliers (outside [-r, r])
            outlier_mask = (t < -self.threshold_r) | (t > self.threshold_r)
            nonoutlier_mask = ~outlier_mask

            # Count statistics
            outlier_count = outlier_mask.sum().item()
            total_count = t.numel()

            # Calculate sum of absolute values
            outlier_sum = (
                torch.sum(torch.abs(t[outlier_mask])).item() if outlier_count > 0 else 0
            )
            nonoutlier_sum = torch.sum(torch.abs(t[nonoutlier_mask])).item()

            # Accumulate statistics
            self.outlier_counts[name] += outlier_count
            self.total_counts[name] += total_count
            self.outlier_magnitude_sum[name] += outlier_sum
            self.nonoutlier_magnitude_sum[name] += nonoutlier_sum

        return hook

    def register_hooks(self, model):
        for name, m in model.named_modules():
            if (
                "module" in name
                or "rotary_emb" in name
                or "q_proj.1" in name
                or "k_proj.1" in name
                or "0.linear_layer" in name
            ):
                continue
            if "layers" in name and any(k in name for k in ["self_attn", "mlp"]):
                h = m.register_forward_hook(self.hook_fn(name))
                self.handles.append(h)

    def remove_hooks(self):
        for h in self.handles:
            h.remove()

    def get_stats(self):
        """
        Returns a dictionary containing outlier statistics for each layer
        """
        stats = {}
        for name in self.total_counts.keys():
            outlier_count = self.outlier_counts[name]
            total_count = self.total_counts[name]
            outlier_ratio = outlier_count / total_count if total_count > 0 else 0

            outlier_sum = self.outlier_magnitude_sum[name]
            nonoutlier_sum = self.nonoutlier_magnitude_sum[name]
            magnitude_ratio = (
                outlier_sum / nonoutlier_sum if nonoutlier_sum > 0 else float("inf")
            )

            stats[name] = {
                "outlier_count": outlier_count,
                "total_count": total_count,
                "outlier_percentage": outlier_ratio * 100,
                "outlier_nonoutlier_magnitude_ratio": magnitude_ratio,
            }

        return stats


def log_outlier_stats(logger, stats, threshold_r):
    """
    Log outlier statistics in a human-readable format

    Args:
        logger: The logger instance to use
        stats: Dictionary from OutlierStatsCollector.get_stats()
        threshold_r: The threshold value used
    """
    # Log separator and title
    logger.info("\n" + "=" * 80)
    logger.info(f"OUTLIER STATISTICS (threshold |x| > {threshold_r})")
    logger.info("=" * 80)

    # Sort layers by name for consistent output
    layer_names = sorted(stats.keys())
    if not layer_names:
        logger.info("No outlier statistics collected.")
        logger.info("=" * 80)
        return

    # Find longest layer name for alignment
    try:
        max_name_len = max(len(name) for name in layer_names) if layer_names else 20
    except ValueError:
        max_name_len = 20  # Default length if something goes wrong

    # Log header
    header = f"{'Layer':<{max_name_len+5}} {'Outliers':<18} {'Percentage':<12} {'|Outliers|/|Non-outliers|':<25}"
    logger.info(header)
    logger.info("-" * len(header))

    # Function to get layer number for sorting
    def get_layer_num(name):
        parts = name.split(".")
        if "layers" in parts:
            try:
                idx = parts.index("layers")
                if idx + 1 < len(parts):
                    seg = parts[idx + 1]
                    if seg.isdigit():
                        return int(seg)
            except ValueError:
                pass  # Handle cases where 'layers' is not found or index is out of bounds
        return -1  # Default sort value if number not found

    # Function to log a group of layers
    def log_layer_group(group_name, layer_group):
        logger.info(f"\n--- {group_name} Layers ---")

        # Sort by the extracted layer number
        layer_group.sort(key=get_layer_num)

        for name in layer_group:
            if name not in stats:
                continue  # Skip if somehow name isn't in stats
            layer_stats = stats[name]
            # Format counts with commas
            count_str = f"{layer_stats.get('outlier_count', 0):,}/{layer_stats.get('total_count', 0):,}"
            pct_str = f"{layer_stats.get('outlier_percentage', 0.0):.4f}%"
            # Format ratio carefully, handling potential inf values
            ratio = layer_stats.get("outlier_nonoutlier_magnitude_ratio", float("nan"))
            if ratio == float("inf"):
                ratio_str = "inf"
            elif ratio is None or ratio != ratio:  # Check for NaN
                ratio_str = "nan"
            else:
                ratio_str = f"{ratio:.6f}"

            logger.info(
                f"{name:<{max_name_len+5}} {count_str:<18} {pct_str:<12} {ratio_str:<25}"
            )

    # Group layers by type
    mlp_layers = [name for name in layer_names if "mlp" in name]
    attn_layers = [name for name in layer_names if "self_attn" in name]
    other_layers = [
        name for name in layer_names if "mlp" not in name and "self_attn" not in name
    ]

    # Log MLP layers
    if mlp_layers:
        log_layer_group("MLP", mlp_layers)

    # Log Attention layers
    if attn_layers:
        log_layer_group("Attention", attn_layers)

    # Log any other layers found
    if other_layers:
        log_layer_group("Other", other_layers)

    # Log Summary statistics
    logger.info("\n--- Summary ---")
    total_outlier_count = sum(s.get("outlier_count", 0) for s in stats.values())
    total_element_count = sum(s.get("total_count", 0) for s in stats.values())
    overall_pct = (
        (total_outlier_count / total_element_count * 100)
        if total_element_count > 0
        else 0.0
    )

    # Calculate average ratio, excluding inf and nan, handling potential division by zero
    valid_ratios = [
        s.get("outlier_nonoutlier_magnitude_ratio", float("nan"))
        for s in stats.values()
        if s.get("outlier_nonoutlier_magnitude_ratio") not in [float("inf"), None]
        and s.get("outlier_nonoutlier_magnitude_ratio")
        == s.get("outlier_nonoutlier_magnitude_ratio")
    ]  # Check for NaN

    avg_ratio = sum(valid_ratios) / len(valid_ratios) if valid_ratios else float("nan")
    avg_ratio_str = (
        f"{avg_ratio:.6f}" if avg_ratio == avg_ratio else "nan"
    )  # Format NaN nicely

    logger.info(
        f"Overall outlier percentage:  {overall_pct:.4f}% ({total_outlier_count:,}/{total_element_count:,})"
    )
    logger.info(f"Average magnitude ratio:     {avg_ratio_str}")
    logger.info("=" * 80)


def generate_normalized_hadamard(dim, device="cuda"):
    """
    Generates a normalized Hadamard matrix of the smallest power of 2 >= dim.
    Returns None if dim is not suitable (e.g., dim=0) or generation fails.
    Pads with identity if dim is not a power of 2.
    """
    if dim <= 0:
        return None

    # Find the next power of 2
    n = 1
    while n < dim:
        n *= 2

    # Generate Hadamard matrix recursively
    H = torch.tensor([[1]], dtype=torch.float32, device=device)
    i = 1
    while i < n:
        H = torch.cat((torch.cat((H, H), dim=1), torch.cat((H, -H), dim=1)), dim=0)
        i *= 2

    # Normalize
    H /= torch.sqrt(torch.tensor(n, dtype=torch.float32, device=device))

    # If dim is not a power of 2, handle padding/truncation (simple approach: identity padding)
    if n != dim:
        # This is a basic way; more sophisticated padding/selection might be needed
        # Here we select the top-left dim x dim block
        H_final = H[:dim, :dim]
        # Ensure it's still orthogonal-ish after truncation (may not be perfect)
        # Alternatively, pad with identity, but requires careful slicing:
        # H_padded = torch.eye(dim, device=device)
        # H_padded[:n, :n] = H # This only works if n <= dim
        # H_final = H[:dim, :dim] # Or just truncate for simplicity
    else:
        H_final = H

    return H_final


def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def matmul_hadU(X, transpose=False):
    n = X.shape[-1]
    hadK, K = get_hadK(n, transpose)
    input = X.clone().view(-1, n, 1)
    output = input.clone()
    while input.shape[1] > K:
        input = input.view(input.shape[0], input.shape[1] // 2, 2, input.shape[2])
        output = output.view(input.shape)
        output[:, :, 0, :] = input[:, :, 0, :] + input[:, :, 1, :]
        output[:, :, 1, :] = input[:, :, 0, :] - input[:, :, 1, :]
        output = output.view(input.shape[0], input.shape[1], -1)
        (input, output) = (output, input)
    del output
    if K > 1:
        # Do not explicitly repeat - OOM
        # input = torch.bmm(
        #     hadK.repeat(len(input), 1, 1).to(input.device).to(input.dtype), input)
        # Use bcast instead
        input = hadK.view(1, K, K).to(input) @ input

    return input.view(X.shape) / torch.tensor(n).sqrt()


def random_hadamard_matrix(size, device):
    # See https://cornell-relaxml.github.io/quip-sharp/ , Section "Randomized Hadamard Transformation"
    Q = torch.randint(low=0, high=2, size=(size,)).to(torch.float32)
    Q = Q * 2 - 1
    Q = torch.diag(Q)
    return matmul_hadU(Q).to(device)


def get_orthogonal_matrix(size, mode, device="cuda"):
    if mode == "random":
        return random_orthogonal_matrix(size, device)
    elif mode == "hadamard":
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f"Unknown mode {mode}")


class RotatedOutlierStatsCollector:
    def __init__(self, threshold_r, device="cuda"):
        self.threshold_r = threshold_r
        self.device = device
        self.handles = []
        self.outlier_counts = defaultdict(int)
        self.total_counts = defaultdict(int)
        self.outlier_magnitude_sum = defaultdict(float)
        self.nonoutlier_magnitude_sum = defaultdict(float)
        self._hadamard_matrices = {}  # Cache generated matrices

    def _get_rotation_matrix(self, dim):
        if dim not in self._hadamard_matrices:
            logger.info(f"Generating Hadamard matrix for dim={dim}")
            H = get_orthogonal_matrix(dim, "hadamard", device=self.device)
            if H is None:
                logger.error(
                    f"Could not generate Hadamard matrix for dim={dim}. Skipping rotation for this layer."
                )
            self._hadamard_matrices[dim] = H
        return self._hadamard_matrices[dim]

    def hook_fn(self, name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                t = output.detach().float()
            elif (
                isinstance(output, tuple)
                and len(output) > 0
                and isinstance(output[0], torch.Tensor)
            ):
                t = output[0].detach().float()
            else:
                return  # Skip if output format is unexpected

            t = t.to(self.device)

            # --- Apply Rotation ---
            dim = t.shape[-1]
            H = self._get_rotation_matrix(dim)

            t_rotated = torch.matmul(t, H)

            # Calculate stats on the (potentially) rotated tensor
            outlier_mask = (t_rotated < -self.threshold_r) | (
                t_rotated > self.threshold_r
            )
            nonoutlier_mask = ~outlier_mask

            outlier_count = outlier_mask.sum().item()
            total_count = t_rotated.numel()

            outlier_sum = (
                torch.sum(torch.abs(t_rotated[outlier_mask])).item()
                if outlier_count > 0
                else 0
            )
            nonoutlier_sum = torch.sum(torch.abs(t_rotated[nonoutlier_mask])).item()

            # Accumulate statistics
            self.outlier_counts[name] += outlier_count
            self.total_counts[name] += total_count
            self.outlier_magnitude_sum[name] += outlier_sum
            self.nonoutlier_magnitude_sum[name] += nonoutlier_sum

        return hook

    def register_hooks(self, model):
        # Check if the module has parameters before hooking - avoids hooking containers
        for name, m in model.named_modules():
            if list(
                m.parameters(recurse=False)
            ):  # Hook only modules with direct parameters
                if "layers" in name and any(
                    k in name
                    for k in [
                        "self_attn",
                        "mlp",
                        "input_layernorm",
                        "post_attention_layernorm",
                    ]
                ):  # Adjust hooks as needed
                    h = m.register_forward_hook(self.hook_fn(name))
                    self.handles.append(
                        (name, h)
                    )  # Store name with handle for debugging
            # Also consider hooking specific leaf modules if needed, e.g., Linear
            # elif isinstance(m, torch.nn.Linear) and "layers" in name:
            #      h = m.register_forward_hook(self.hook_fn(name))
            #      self.handles.append((name, h))

    def remove_hooks(self):
        for name, h in self.handles:
            try:
                h.remove()
            except Exception as e:
                logger.error(f"Error removing hook for {name}: {e}")
        self.handles = []  # Clear handles

    def get_stats(self):
        # Same structure as OutlierStatsCollector
        stats = {}
        for name in self.total_counts.keys():
            # ... (copy the get_stats logic from OutlierStatsCollector) ...
            outlier_count = self.outlier_counts[name]
            total_count = self.total_counts[name]
            outlier_percentage = (
                (outlier_count / total_count * 100) if total_count > 0 else 0
            )

            outlier_sum = self.outlier_magnitude_sum[name]
            non_outlier_sum = self.nonoutlier_magnitude_sum[name]
            magnitude_ratio = (
                outlier_sum / non_outlier_sum if non_outlier_sum > 0 else float("inf")
            )

            stats[name] = {
                "outlier_count": outlier_count,
                "total_count": total_count,
                "outlier_percentage": outlier_percentage,
                "outlier_nonoutlier_magnitude_ratio": magnitude_ratio,
                "nonoutlier_magnitude_sum": non_outlier_sum,  # Include for ratio calculation robustness
            }
        return stats


def save_histogram_summary(
    histograms, layer_ranges, output_file="activation_summary.txt"
):
    """
    Save all layers' activation histogram information to a single text file with improved readability.

    Args:
        histograms: Dict mapping layer names to numpy arrays of histogram counts
        layer_ranges: Dict mapping layer names to (min, max) pairs defining histogram ranges
        output_file: Path to the output text file
    """
    import os
    import numpy as np

    # Create directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Helper functions
    def get_layer_num(name):
        parts = name.split(".")
        if "layers" in parts:
            try:
                idx = parts.index("layers")
                if idx + 1 < len(parts) and parts[idx + 1].isdigit():
                    return int(parts[idx + 1])
            except ValueError:
                pass
        return -1

    def calculate_layer_stats(counts, min_val, max_val):
        """Calculate comprehensive statistics for a layer"""
        total_count = np.sum(counts)
        if total_count == 0:
            return None

        bin_edges = np.linspace(min_val, max_val, len(counts) + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Basic stats
        mean = np.sum(bin_centers * counts) / total_count

        # Variance and std dev
        variance = np.sum(counts * (bin_centers - mean) ** 2) / total_count
        std_dev = np.sqrt(variance)

        # Percentiles
        cumsum = np.cumsum(counts)
        cumsum_norm = cumsum / total_count

        percentiles = {}
        for p in [1, 5, 25, 50, 75, 95, 99]:
            idx = np.searchsorted(cumsum_norm, p / 100.0)
            percentiles[p] = bin_centers[min(idx, len(bin_centers) - 1)]

        # Peak and mode
        peak_idx = np.argmax(counts)
        peak_value = bin_centers[peak_idx]

        # Range and outlier info
        range_val = max_val - min_val
        outlier_ratio = (
            cumsum_norm[-1] - cumsum_norm[np.searchsorted(cumsum_norm, 0.99)]
        ) * 100

        return {
            "total_count": total_count,
            "mean": mean,
            "std_dev": std_dev,
            "variance": variance,
            "peak_value": peak_value,
            "percentiles": percentiles,
            "range": range_val,
            "outlier_ratio": outlier_ratio,
        }

    # Analyze all layers
    layer_names = sorted(histograms.keys())
    layer_stats = {}

    for name in layer_names:
        counts = histograms[name]
        min_val, max_val = layer_ranges[name]
        stats = calculate_layer_stats(counts, min_val, max_val)
        if stats:
            layer_stats[name] = stats

    # Group layers
    mlp_layers = [name for name in layer_names if "mlp" in name and name in layer_stats]
    attn_layers = [
        name for name in layer_names if "self_attn" in name and name in layer_stats
    ]
    other_layers = [
        name
        for name in layer_names
        if "mlp" not in name and "self_attn" not in name and name in layer_stats
    ]

    # Sort layers by range for analysis
    all_layers_by_range = sorted(
        layer_stats.items(), key=lambda x: x[1]["range"], reverse=True
    )

    with open(output_file, "w") as f:
        # Header
        f.write("🔍 LLAMA ACTIVATION ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Executive Summary
        f.write("📊 EXECUTIVE SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total layers analyzed: {len(layer_stats)}\n")
        f.write(
            f"MLP layers: {len(mlp_layers)} | Attention layers: {len(attn_layers)} | Other: {len(other_layers)}\n\n"
        )

        # Global statistics
        all_ranges = [
            layer_ranges[name][1] - layer_ranges[name][0] for name in layer_stats
        ]
        all_stds = [layer_stats[name]["std_dev"] for name in layer_stats]
        all_outliers = [layer_stats[name]["outlier_ratio"] for name in layer_stats]

        f.write(
            f"Global activation range: [{min(layer_ranges[name][0] for name in layer_stats):.4f}, {max(layer_ranges[name][1] for name in layer_stats):.4f}]\n"
        )
        f.write(
            f"Average layer range: {np.mean(all_ranges):.4f} ± {np.std(all_ranges):.4f}\n"
        )
        f.write(f"Average std deviation: {np.mean(all_stds):.4f}\n")
        f.write(f"Average outlier ratio: {np.mean(all_outliers):.2f}%\n\n")

        # Top layers by range
        f.write("⚠️  TOP 5 LAYERS WITH LARGEST ACTIVATION RANGES:\n")
        for i, (name, stats) in enumerate(all_layers_by_range[:5]):
            f.write(
                f"{i+1:2d}. {name:<45} | Range: {stats['range']:.4f} | Outliers: {stats['outlier_ratio']:.2f}%\n"
            )
        f.write("\n")
        high_range = np.mean(all_ranges) + np.std(all_ranges)
        # Quick recommendations based on range and outliers
        high_range_count = sum(
            1
            for _, stats in layer_stats.items()
            if stats["range"] > high_range
        )
        high_outlier_count = sum(
            1 for _, stats in layer_stats.items() if stats["outlier_ratio"] > 1.0
        )

        f.write("💡 QUANTIZATION OBSERVATIONS:\n")
        f.write(
            f"   • {high_range_count} layers have significantly large activation ranges of > {high_range:.4f}\n"
        )
        f.write(f"   • {high_outlier_count} layers have >1% outliers beyond P99\n")
        f.write(
            f"   • {len(layer_stats) - high_range_count - high_outlier_count} layers appear well-behaved for quantization\n\n"
        )

        # Detailed layer analysis in tabular format
        f.write("📋 DETAILED LAYER ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        # Table header
        f.write(
            f"{'Layer':<35} {'Range':<12} {'StdDev':<10} {'P99':<10} {'Mean':<10} {'Outliers%':<10}\n"
        )
        f.write("-" * 90 + "\n")

        def write_layer_table(layer_group, group_name):
            if not layer_group:
                return

            f.write(f"\n--- {group_name} Layers ---\n")
            layer_group.sort(key=get_layer_num)

            for name in layer_group:
                stats = layer_stats[name]
                min_val, max_val = layer_ranges[name]

                # Extract short name for display
                short_name = name.split(".")[-2:] if "." in name else [name]
                display_name = ".".join(short_name)

                f.write(
                    f"{display_name:<35} {stats['range']:<12.4f} {stats['std_dev']:<10.4f} "
                    f"{stats['percentiles'][99]:<10.4f} {stats['mean']:<10.4f} {stats['outlier_ratio']:<10.2f}\n"
                )

        # Write tables for each group
        write_layer_table(mlp_layers, "MLP")
        write_layer_table(attn_layers, "Attention")
        write_layer_table(other_layers, "Other")

        # Detailed statistics for layers with largest ranges
        f.write("\n\n🔬 DETAILED ANALYSIS OF TOP 3 LAYERS WITH LARGEST RANGES\n")
        f.write("=" * 80 + "\n")

        for i, (name, stats) in enumerate(all_layers_by_range[:3]):
            min_val, max_val = layer_ranges[name]
            f.write(f"\n{i+1}. {name}\n")
            f.write(
                f"   Range: [{min_val:.6f}, {max_val:.6f}] (span: {stats['range']:.6f})\n"
            )
            f.write(f"   Statistics: μ={stats['mean']:.6f}, σ={stats['std_dev']:.6f}\n")
            f.write(
                f"   Percentiles: P1={stats['percentiles'][1]:.4f}, P50={stats['percentiles'][50]:.4f}, P99={stats['percentiles'][99]:.4f}\n"
            )
            f.write(f"   Outlier ratio: {stats['outlier_ratio']:.2f}%\n")

            # Mini histogram using Unicode blocks
            counts = histograms[name]
            if len(counts) > 0:
                # Downsample for display
                step = max(1, len(counts) // 40)
                mini_counts = counts[::step][:40]
                max_count = np.max(mini_counts) if len(mini_counts) > 0 else 1

                f.write("   Distribution: ")
                blocks = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
                for count in mini_counts:
                    block_idx = (
                        min(7, int(8 * count / max_count)) if max_count > 0 else 0
                    )
                    f.write(blocks[block_idx])
                f.write("\n")

        # Layer type comparisons
        f.write("\n\n📈 LAYER TYPE COMPARISON\n")
        f.write("=" * 80 + "\n")

        def analyze_group(group, group_name):
            if not group:
                return

            group_stats = [layer_stats[name] for name in group]
            ranges = [layer_stats[name]["range"] for name in group]
            stds = [layer_stats[name]["std_dev"] for name in group]
            outliers = [layer_stats[name]["outlier_ratio"] for name in group]

            f.write(f"\n{group_name} Layers ({len(group)} total):\n")
            f.write(f"  Average range: {np.mean(ranges):.4f} ± {np.std(ranges):.4f}\n")
            f.write(f"  Average std dev: {np.mean(stds):.4f}\n")
            f.write(f"  Average outlier ratio: {np.mean(outliers):.2f}%\n")
            f.write(
                f"  Well-behaved layers: {sum(1 for r in ranges if r < np.mean(all_ranges))}/{len(group)} have below-average ranges\n"
            )

        analyze_group(mlp_layers, "MLP")
        analyze_group(attn_layers, "Attention")
        analyze_group(other_layers, "Other")

        # Footer
        f.write("\n" + "=" * 80 + "\n")
        f.write("📝 Report generated by SpinQuant activation analysis\n")
        f.write(
            "   Focus on layers with large ranges and high outlier ratios for quantization\n"
        )
        f.write("=" * 80 + "\n")


def log_histogram_summary(logger, histograms, layer_ranges):
    """
    Log a condensed version of histogram summary to the logger.

    Args:
        logger: Logger instance to write to
        histograms: Dict mapping layer names to numpy arrays of histogram counts
        layer_ranges: Dict mapping layer names to (min, max) pairs defining histogram ranges
    """
    import numpy as np

    # Helper functions (reuse from save_histogram_summary)
    def get_layer_num(name):
        parts = name.split(".")
        if "layers" in parts:
            try:
                idx = parts.index("layers")
                if idx + 1 < len(parts) and parts[idx + 1].isdigit():
                    return int(parts[idx + 1])
            except ValueError:
                pass
        return -1

    def calculate_layer_stats(counts, min_val, max_val):
        """Calculate comprehensive statistics for a layer"""
        total_count = np.sum(counts)
        if total_count == 0:
            return None

        bin_edges = np.linspace(min_val, max_val, len(counts) + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Basic stats
        mean = np.sum(bin_centers * counts) / total_count

        # Variance and std dev
        variance = np.sum(counts * (bin_centers - mean) ** 2) / total_count
        std_dev = np.sqrt(variance)

        # Percentiles
        cumsum = np.cumsum(counts)
        cumsum_norm = cumsum / total_count

        percentiles = {}
        for p in [1, 5, 25, 50, 75, 95, 99]:
            idx = np.searchsorted(cumsum_norm, p / 100.0)
            percentiles[p] = bin_centers[min(idx, len(bin_centers) - 1)]

        # Range and outlier info
        range_val = max_val - min_val
        outlier_ratio = (
            cumsum_norm[-1] - cumsum_norm[np.searchsorted(cumsum_norm, 0.99)]
        ) * 100

        return {
            "total_count": total_count,
            "mean": mean,
            "std_dev": std_dev,
            "range": range_val,
            "outlier_ratio": outlier_ratio,
            "percentiles": percentiles,
        }

    # Analyze all layers
    layer_names = sorted(histograms.keys())
    layer_stats = {}

    for name in layer_names:
        counts = histograms[name]
        min_val, max_val = layer_ranges[name]
        stats = calculate_layer_stats(counts, min_val, max_val)
        if stats:
            layer_stats[name] = stats

    if not layer_stats:
        logger.info("No histogram data to analyze")
        return

    # Group layers
    mlp_layers = [name for name in layer_names if "mlp" in name and name in layer_stats]
    attn_layers = [
        name for name in layer_names if "self_attn" in name and name in layer_stats
    ]
    other_layers = [
        name
        for name in layer_names
        if "mlp" not in name and "self_attn" not in name and name in layer_stats
    ]

    # Sort layers by range
    all_layers_by_range = sorted(
        layer_stats.items(), key=lambda x: x[1]["range"], reverse=True
    )

    # Log summary
    logger.info("=" * 60)
    logger.info("ACTIVATION DISTRIBUTION ANALYSIS SUMMARY")
    logger.info("=" * 60)

    # Basic counts
    logger.info(f"Total layers analyzed: {len(layer_stats)}")
    logger.info(
        f"MLP: {len(mlp_layers)} | Attention: {len(attn_layers)} | Other: {len(other_layers)}"
    )

    # Global statistics
    all_ranges = [layer_ranges[name][1] - layer_ranges[name][0] for name in layer_stats]
    all_stds = [layer_stats[name]["std_dev"] for name in layer_stats]
    all_outliers = [layer_stats[name]["outlier_ratio"] for name in layer_stats]

    global_min = min(layer_ranges[name][0] for name in layer_stats)
    global_max = max(layer_ranges[name][1] for name in layer_stats)

    logger.info(f"Global activation range: [{global_min:.4f}, {global_max:.4f}]")
    logger.info(
        f"Average layer range: {np.mean(all_ranges):.4f} ± {np.std(all_ranges):.4f}"
    )
    logger.info(f"Average std deviation: {np.mean(all_stds):.4f}")
    logger.info(f"Average outlier ratio: {np.mean(all_outliers):.2f}%")

    # Analysis
    high_range_count = sum(
        1
        for _, stats in layer_stats.items()
        if stats["range"] > np.mean(all_ranges) + np.std(all_ranges)
    )
    high_outlier_count = sum(
        1 for _, stats in layer_stats.items() if stats["outlier_ratio"] > 1.0
    )

    logger.info("-" * 60)
    logger.info("QUANTIZATION ANALYSIS:")
    logger.info(f"  Large range layers:   {high_range_count:3d} layers")
    logger.info(f"  High outlier layers:  {high_outlier_count:3d} layers")
    logger.info(
        f"  Well-behaved layers:  {len(layer_stats) - high_range_count - high_outlier_count:3d} layers"
    )

    # Top layers by range
    logger.info("-" * 60)
    logger.info("TOP 5 LAYERS WITH LARGEST RANGES:")
    for i, (name, stats) in enumerate(all_layers_by_range[:5]):
        short_name = ".".join(name.split(".")[-2:]) if "." in name else name
        logger.info(
            f"  {i+1}. {short_name:<30} | Range: {stats['range']:.4f} | Outliers: {stats['outlier_ratio']:.2f}%"
        )

    # Layer type analysis
    def log_group_stats(group, group_name):
        if not group:
            return
        group_stats = [layer_stats[name] for name in group]
        ranges = [layer_stats[name]["range"] for name in group]
        outliers = [layer_stats[name]["outlier_ratio"] for name in group]
        well_behaved = sum(1 for r in ranges if r < np.mean(all_ranges))

        logger.info(f"{group_name} layers ({len(group)} total):")
        logger.info(
            f"  Avg range: {np.mean(ranges):.4f} | Avg outliers: {np.mean(outliers):.2f}% | Well-behaved: {well_behaved}/{len(group)}"
        )

    logger.info("-" * 60)
    logger.info("LAYER TYPE ANALYSIS:")
    log_group_stats(mlp_layers, "MLP")
    log_group_stats(attn_layers, "Attention")
    log_group_stats(other_layers, "Other")

    logger.info("=" * 60)


class OutlierDistributionAccumulator:
    """
    GPU-based outlier distribution accumulator focusing on tail behavior.

    Unlike regular histograms that focus on the full distribution,
    this specifically counts activations in outlier ranges to analyze
    tail behavior for quantization decisions.
    """

    def __init__(self, layer_ranges: dict, outlier_thresholds=[90, 95, 99, 99.9, 99.99], device="cuda"):
        """
        Args:
            layer_ranges: dict[name] = (min_val, max_val)
            outlier_thresholds: Percentile thresholds to track (e.g., P99, P99.9)
            device: Device for GPU acceleration
        """
        self.device = device
        self.handles = []
        self.outlier_thresholds = outlier_thresholds
        self.layer_ranges = {}

        # Pre-allocate outlier count tensors on GPU
        self.outlier_counts = {}
        self.total_counts = {}

        for name, (min_val, max_val) in layer_ranges.items():
            self.layer_ranges[name] = (min_val, max_val)
            # Initialize counters for each percentile threshold
            self.outlier_counts[name] = {
                f'p{th}': torch.zeros(1, dtype=torch.long, device=device)
                for th in outlier_thresholds
            }
            self.total_counts[name] = torch.zeros(1, dtype=torch.long, device=device)

    def hook_fn(self, name):
        mn, mx = self.layer_ranges[name]

        def hook(module, input, output):
            t = output[0] if isinstance(output, tuple) else output
            t = t.detach().float().to(self.device)

            # Count total elements
            total_elements = t.numel()
            self.total_counts[name] += total_elements

            # Get absolute values for outlier detection
            t_abs = torch.abs(t)

            # Calculate percentile thresholds dynamically from current batch
            # Use kthvalue for exact percentiles on GPU
            flat_abs = t_abs.flatten()
            n = flat_abs.numel()

            for threshold in self.outlier_thresholds:
                # Calculate k for percentile (0-indexed)
                k = int((threshold / 100.0) * n)
                k = min(max(k, 0), n - 1)

                if k < n - 1:  # Ensure we don't go out of bounds
                    percentile_val = torch.kthvalue(flat_abs, k + 1).values  # kthvalue is 1-indexed
                    outlier_mask = t_abs >= percentile_val
                    outlier_count = outlier_mask.sum()
                    self.outlier_counts[name][f'p{threshold}'] += outlier_count

        return hook

    def register_hooks(self, model):
        """Register hooks on model layers"""
        for name, m in model.named_modules():
            if (
                "module" in name
                or "rotary_emb" in name
                or "q_proj.1" in name
                or "k_proj.1" in name
                or "0.linear_layer" in name
                or "proj_module" in name
                or "linear_layer" in name
            ):
                continue
            if name in self.outlier_counts:
                h = m.register_forward_hook(self.hook_fn(name))
                self.handles.append(h)

    def remove_hooks(self):
        """Remove all registered hooks"""
        for h in self.handles:
            h.remove()

    def get_outlier_distributions(self):
        """Get outlier distributions as CPU numpy arrays"""
        results = {}
        for name in self.outlier_counts:
            results[name] = {
                'total_count': self.total_counts[name].cpu().item(),
                'outlier_counts': {
                    threshold: count.cpu().item()
                    for threshold, count in self.outlier_counts[name].items()
                },
                'outlier_percentages': {}
            }

            # Calculate percentages
            total = results[name]['total_count']
            for threshold, count in results[name]['outlier_counts'].items():
                results[name]['outlier_percentages'][threshold] = (count / total * 100) if total > 0 else 0.0

        return results


def save_outlier_distribution_summary(
    outlier_distributions, layer_ranges, output_file="outlier_distribution_summary.txt"
):
    """
    Save outlier distribution analysis focusing on tail behavior.

    Args:
        outlier_distributions: Dict from OutlierDistributionAccumulator.get_outlier_distributions()
        layer_ranges: Dict mapping layer names to (min, max) pairs
        output_file: Path to output text file
    """
    import os
    import numpy as np

    # Create directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Helper functions
    def get_layer_num(name):
        parts = name.split(".")
        if "layers" in parts:
            try:
                idx = parts.index("layers")
                if idx + 1 < len(parts) and parts[idx + 1].isdigit():
                    return int(parts[idx + 1])
            except ValueError:
                pass
        return -1

    # Group layers
    layer_names = sorted(outlier_distributions.keys())
    mlp_layers = [name for name in layer_names if "mlp" in name]
    attn_layers = [name for name in layer_names if "self_attn" in name]
    other_layers = [name for name in layer_names if "mlp" not in name and "self_attn" not in name]

    with open(output_file, "w") as f:
        # Header
        f.write("🎯 OUTLIER DISTRIBUTION ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Executive Summary
        f.write("📊 OUTLIER FOCUS SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total layers analyzed: {len(outlier_distributions)}\n")
        f.write(f"MLP layers: {len(mlp_layers)} | Attention layers: {len(attn_layers)} | Other: {len(other_layers)}\n")
        f.write("Focus: Tail behavior analysis for quantization outlier detection\n\n")

        # Get available thresholds from first layer
        if outlier_distributions:
            sample_layer = next(iter(outlier_distributions.values()))
            thresholds = sorted(sample_layer['outlier_percentages'].keys())

            # Global outlier statistics
            f.write("🚨 GLOBAL OUTLIER STATISTICS\n")
            f.write("-" * 40 + "\n")

            for threshold in thresholds:
                all_percentages = [
                    dist['outlier_percentages'][threshold]
                    for dist in outlier_distributions.values()
                ]
                mean_pct = np.mean(all_percentages)
                std_pct = np.std(all_percentages)
                max_pct = np.max(all_percentages)

                f.write(f"{threshold.upper()}: μ={mean_pct:.3f}%, σ={std_pct:.3f}%, max={max_pct:.3f}%\n")

            f.write("\n")

            # Detailed layer analysis
            f.write("📋 DETAILED OUTLIER ANALYSIS BY LAYER\n")
            f.write("=" * 80 + "\n\n")

            # Table header
            header = f"{'Layer':<35}"
            for threshold in thresholds:
                header += f" {threshold.upper():<8}"
            header += f" {'Range':<12} {'Total':<10}\n"
            f.write(header)
            f.write("-" * len(header) + "\n")

            def write_layer_group(layer_group, group_name):
                if not layer_group:
                    return

                f.write(f"\n--- {group_name} Layers ---\n")
                layer_group.sort(key=get_layer_num)

                for name in layer_group:
                    if name not in outlier_distributions:
                        continue

                    dist = outlier_distributions[name]
                    min_val, max_val = layer_ranges[name]

                    # Extract short name for display
                    short_name = name.split(".")[-2:] if "." in name else [name]
                    display_name = ".".join(short_name)

                    # Build row
                    row = f"{display_name:<35}"
                    for threshold in thresholds:
                        pct = dist['outlier_percentages'][threshold]
                        row += f" {pct:<8.3f}"

                    range_val = max_val - min_val
                    total_count = dist['total_count']
                    row += f" {range_val:<12.4f} {total_count:<10,}\n"
                    f.write(row)

            # Write tables for each group
            write_layer_group(mlp_layers, "MLP")
            write_layer_group(attn_layers, "Attention")
            write_layer_group(other_layers, "Other")

            # Top outlier layers analysis
            f.write("\n\n🔥 TOP LAYERS WITH HIGHEST OUTLIER RATES\n")
            f.write("=" * 80 + "\n")

            # Find layers with highest P99.9 outlier rates
            p999_rates = [
                (name, dist['outlier_percentages'].get('p99.9', 0.0))
                for name, dist in outlier_distributions.items()
            ]
            p999_rates.sort(key=lambda x: x[1], reverse=True)

            f.write("Top 5 layers by P99.9 outlier rate:\n")
            for i, (name, rate) in enumerate(p999_rates[:5]):
                short_name = ".".join(name.split(".")[-2:]) if "." in name else name
                f.write(f"  {i+1}. {short_name:<30} | P99.9: {rate:.3f}%\n")

            # Quantization recommendations
            f.write("\n\n💡 QUANTIZATION RECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")

            high_outlier_layers = [
                name for name, rate in p999_rates
                if rate > np.mean([r[1] for r in p999_rates]) + np.std([r[1] for r in p999_rates])
            ]

            f.write(f"🎯 Focus quantization attention on {len(high_outlier_layers)} high-outlier layers\n")
            f.write("🔍 These layers show significant tail behavior requiring careful quantization\n")
            f.write(f"📈 {len(p999_rates) - len(high_outlier_layers)} layers show normal outlier distribution\n")

            # Layer type analysis
            f.write("\n📈 OUTLIER ANALYSIS BY LAYER TYPE\n")
            f.write("=" * 80 + "\n")

            def analyze_group_outliers(group, group_name):
                if not group:
                    return

                group_p999_rates = [
                    outlier_distributions[name]['outlier_percentages'].get('p99.9', 0.0)
                    for name in group if name in outlier_distributions
                ]

                if group_p999_rates:
                    f.write(f"\n{group_name} Layers ({len(group)} total):\n")
                    f.write(f"  Average P99.9 outlier rate: {np.mean(group_p999_rates):.3f}%\n")
                    f.write(f"  Std deviation: {np.std(group_p999_rates):.3f}%\n")
                    f.write(f"  Max outlier rate: {np.max(group_p999_rates):.3f}%\n")
                    f.write(f"  Layers above average: {sum(1 for r in group_p999_rates if r > np.mean(group_p999_rates))}/{len(group)}\n")

            analyze_group_outliers(mlp_layers, "MLP")
            analyze_group_outliers(attn_layers, "Attention")
            analyze_group_outliers(other_layers, "Other")

        # Footer
        f.write("\n" + "=" * 80 + "\n")
        f.write("📝 Report generated by SpinQuant outlier distribution analysis\n")
        f.write("   Focus: Tail behavior for informed quantization decisions\n")
        f.write("=" * 80 + "\n")


def log_outlier_distribution_summary(logger, outlier_distributions, layer_ranges):
    """
    Log a condensed version of outlier distribution analysis.

    Args:
        logger: Logger instance
        outlier_distributions: Dict from OutlierDistributionAccumulator.get_outlier_distributions()
        layer_ranges: Dict mapping layer names to (min, max) pairs
    """
    import numpy as np

    if not outlier_distributions:
        logger.info("No outlier distribution data to analyze")
        return

    # Get available thresholds
    sample_layer = next(iter(outlier_distributions.values()))
    thresholds = sorted(sample_layer['outlier_percentages'].keys())

    # Group layers
    layer_names = sorted(outlier_distributions.keys())
    mlp_layers = [name for name in layer_names if "mlp" in name]
    attn_layers = [name for name in layer_names if "self_attn" in name]
    other_layers = [name for name in layer_names if "mlp" not in name and "self_attn" not in name]

    logger.info("=" * 60)
    logger.info("OUTLIER DISTRIBUTION ANALYSIS")
    logger.info("=" * 60)

    # Basic counts
    logger.info(f"Total layers analyzed: {len(outlier_distributions)}")
    logger.info(f"MLP: {len(mlp_layers)} | Attention: {len(attn_layers)} | Other: {len(other_layers)}")

    # Global outlier statistics
    logger.info("\nGlobal Outlier Rates:")
    for threshold in thresholds:
        all_percentages = [
            dist['outlier_percentages'][threshold]
            for dist in outlier_distributions.values()
        ]
        mean_pct = np.mean(all_percentages)
        max_pct = np.max(all_percentages)
        logger.info(f"  {threshold.upper()}: μ={mean_pct:.3f}%, max={max_pct:.3f}%")

    # Top outlier layers
    p999_rates = [
        (name, dist['outlier_percentages'].get('p99.9', 0.0))
        for name, dist in outlier_distributions.items()
    ]
    p999_rates.sort(key=lambda x: x[1], reverse=True)

    logger.info("\nTop 3 Highest P99.9 Outlier Layers:")
    for i, (name, rate) in enumerate(p999_rates[:3]):
        short_name = ".".join(name.split(".")[-2:]) if "." in name else name
        logger.info(f"  {i+1}. {short_name:<25} | P99.9: {rate:.3f}%")

    # Quantization focus
    high_outlier_count = sum(1 for _, rate in p999_rates if rate > 0.1)  # > 0.1% outliers
    logger.info(f"\nQuantization Focus: {high_outlier_count} layers with >0.1% P99.9 outliers")

    logger.info("=" * 60)


def save_histograms_bincount(
    histograms, layer_ranges, output_dir="histograms_bincount"
):
    """
    Save raw histogram bin counts for each layer in separate files for outlier analysis.

    Creates one file per layer in a dedicated directory, focusing on outlier distributions.
    Each file contains raw bin counts for custom tail behavior analysis.

    Args:
        histograms: Dict mapping layer names to numpy arrays of histogram counts
        layer_ranges: Dict mapping layer names to (min, max) pairs defining histogram ranges
        output_dir: Directory to save individual layer bin count files
    """
    import os
    import numpy as np

    # Create dedicated directory
    os.makedirs(output_dir, exist_ok=True)

    def sanitize_filename(layer_name):
        """Convert layer name to safe filename"""
        return layer_name.replace(".", "_").replace("/", "_")

    # Process each layer individually
    layer_names = sorted(histograms.keys())
    summary_stats = []

    for name in layer_names:
        if name not in histograms:
            continue

        counts = histograms[name]
        min_val, max_val = layer_ranges[name]

        # Create individual file for this layer
        safe_name = sanitize_filename(name)
        layer_file = os.path.join(output_dir, f"{safe_name}.txt")

        with open(layer_file, "w") as f:
            # Header for individual layer file
            f.write(f"📊 HISTOGRAM BIN COUNTS: {name}\n")
            f.write("=" * 80 + "\n\n")

            f.write("Purpose: Raw bin counts for outlier analysis\n")
            f.write("Use: Analyze tail behavior for quantization decisions\n\n")

            # Layer metadata
            f.write("LAYER INFORMATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Layer: {name}\n")
            f.write(f"Range: [{min_val:.6f}, {max_val:.6f}]\n")
            f.write(f"Total bins: {len(counts)}\n")

            # Calculate bin properties
            bin_edges = np.linspace(min_val, max_val, len(counts) + 1)
            bin_width = (max_val - min_val) / len(counts)
            total_count = np.sum(counts)

            f.write(f"Bin width: {bin_width:.6f}\n")
            f.write(f"Total activations: {total_count:,}\n\n")

            # Raw bin counts - complete data
            f.write("RAW BIN COUNTS\n")
            f.write("-" * 40 + "\n")
            f.write("bin_idx count bin_start bin_end\n")
            f.write("-" * 40 + "\n")

            for i, count in enumerate(counts):
                bin_start = bin_edges[i]
                bin_end = bin_edges[i + 1]
                f.write(f"{i:3d} {count:8,} {bin_start:10.6f} {bin_end:10.6f}\n")

            # Outlier analysis
            f.write(f"\nOUTLIER ANALYSIS\n")
            f.write("-" * 40 + "\n")

            if total_count > 0:
                # Non-zero bins
                nonzero_bins = [(i, count) for i, count in enumerate(counts) if count > 0]
                f.write(f"Non-zero bins: {len(nonzero_bins)}/{len(counts)}\n")

                # Percentile bins
                top_10pct_bins = len(counts) // 10
                top_1pct_bins = max(1, len(counts) // 100)

                top_10pct_count = np.sum(counts[-top_10pct_bins:]) if top_10pct_bins > 0 else 0
                top_1pct_count = np.sum(counts[-top_1pct_bins:]) if top_1pct_bins > 0 else 0

                f.write(f"Top 10% bins ({top_10pct_bins} bins): {top_10pct_count:,} activations ({top_10pct_count/total_count*100:.3f}%)\n")
                f.write(f"Top 1% bins ({top_1pct_bins} bins): {top_1pct_count:,} activations ({top_1pct_count/total_count*100:.3f}%)\n")

                # Peak bin
                max_bin_idx = np.argmax(counts)
                max_bin_count = counts[max_bin_idx]
                max_bin_start = bin_edges[max_bin_idx]
                max_bin_end = bin_edges[max_bin_idx + 1]
                f.write(f"Peak bin: {max_bin_idx} with {max_bin_count:,} activations [{max_bin_start:.6f}, {max_bin_end:.6f})\n")

                # Store for summary
                outlier_pct = (top_1pct_count / total_count * 100) if total_count > 0 else 0
                summary_stats.append((name, total_count, outlier_pct, max_bin_idx, max_bin_count))

                # Tail bins (last 10 non-empty bins)
                f.write(f"\nTAIL BINS (last 10 non-zero bins):\n")
                f.write("-" * 40 + "\n")
                tail_bins = [x for x in nonzero_bins if x[1] > 0][-10:]
                for i, count in tail_bins:
                    bin_start = bin_edges[i]
                    bin_end = bin_edges[i + 1]
                    pct = (count / total_count * 100) if total_count > 0 else 0
                    f.write(f"  {i:3d}: {count:8,} activations ({pct:.4f}%) [{bin_start:.6f}, {bin_end:.6f})\n")

            f.write(f"\n" + "=" * 80 + "\n")
            f.write(f"📝 Individual layer analysis complete\n")
            f.write(f"💡 Focus on high bin indices for quantization outliers\n")
            f.write("=" * 80 + "\n")

    # Create summary file
    summary_file = os.path.join(output_dir, "_SUMMARY.txt")
    with open(summary_file, "w") as f:
        f.write("📊 HISTOGRAM BIN COUNTS - SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total layers processed: {len(summary_stats)}\n")
        f.write(f"Individual files created in: {output_dir}/\n\n")

        if summary_stats:
            # Overall statistics
            all_total_counts = [s[1] for s in summary_stats]
            all_outlier_percentages = [s[2] for s in summary_stats]

            f.write("GLOBAL STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total activations across all layers: {sum(all_total_counts):,}\n")
            f.write(f"Average activations per layer: {np.mean(all_total_counts):,.0f}\n")
            f.write(f"Average outlier percentage (top 1% bins): {np.mean(all_outlier_percentages):.3f}%\n")
            f.write(f"Max outlier percentage: {np.max(all_outlier_percentages):.3f}%\n\n")

            # Top outlier layers
            summary_stats.sort(key=lambda x: x[2], reverse=True)  # Sort by outlier percentage

            f.write("TOP OUTLIER LAYERS (by top 1% bin percentage)\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Rank':<4} {'Layer':<45} {'Outlier%':<10} {'Peak Bin':<8} {'Peak Count':<12}\n")
            f.write("-" * 80 + "\n")

            for i, (name, total_count, outlier_pct, peak_bin, peak_count) in enumerate(summary_stats[:10]):
                short_name = name.split(".")[-2:] if "." in name else [name]
                display_name = ".".join(short_name)
                f.write(f"{i+1:<4} {display_name:<45} {outlier_pct:<10.3f} {peak_bin:<8} {peak_count:<12,}\n")

            f.write(f"\n" + "=" * 80 + "\n")
            f.write(f"📁 {len(layer_names)} individual layer files saved\n")
            f.write(f"🎯 Focus on layers with highest outlier percentages\n")
            f.write(f"📄 Each file contains complete bin counts for custom analysis\n")
            f.write("=" * 80 + "\n")
