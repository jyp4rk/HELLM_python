#!/usr/bin/env python3
"""
Memory-Efficient Streaming Outlier Loss for SpinQuant 97-Point Tracking

Fixed version that preserves gradients for backpropagation.
"""

import torch
import torch.nn as nn
from typing import Dict, List

class StreamingOutlierLoss(nn.Module):
    """
    Memory-efficient streaming outlier loss computation with gradient preservation.
    """

    def __init__(self,
                 loss_type: str = "cvar",
                 alpha: float = 0.99,
                 initial_threshold: float = 4.0,
                 target_threshold: float = 4.0,
                 ema_decay: float = 0.95):
        super().__init__()

        self.loss_type = loss_type
        self.ema_decay = ema_decay

        if loss_type == "cvar":
            self.alpha = alpha
            self.cvar_coefficient = 1.0 / (1.0 - alpha)
            self.register_buffer('threshold', torch.tensor(initial_threshold))
        elif loss_type == "hinged":
            self.target_threshold = target_threshold
            self.register_buffer('current_threshold', torch.tensor(initial_threshold))
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        # Statistics buffers (no gradients needed)
        self.register_buffer('global_max', torch.tensor(0.0))
        self.register_buffer('running_percentile', torch.tensor(initial_threshold))

    def compute_layer_loss(self, activations: torch.Tensor, layer_name: str = "") -> torch.Tensor:
        """
        Compute loss for a single layer with gradient preservation.
        """
        device = activations.device

        # Convert to float32 if needed (preserve gradients)
        if activations.dtype != torch.float32:
            activations_float = activations.float()
        else:
            activations_float = activations

        # Flatten efficiently
        if activations_float.dim() > 2:
            flat_activations = activations_float.view(-1, activations_float.shape[-1])
        else:
            flat_activations = activations_float

        # Compute magnitudes (preserve gradients)
        magnitudes = flat_activations.abs()

        # Update statistics (no gradients)
        with torch.no_grad():
            batch_max = magnitudes.max()
            self.global_max = torch.max(self.global_max, batch_max)

            # Update running percentile
            if magnitudes.numel() > 100:
                batch_seq_size = magnitudes.shape[0]
                if batch_seq_size > 10000:
                    n_samples = min(10000, batch_seq_size)
                    indices = torch.randperm(batch_seq_size, device=device)[:n_samples]
                    percentile = torch.quantile(magnitudes[indices], 0.99, dim=1).median()
                else:
                    percentile = torch.quantile(magnitudes, 0.99, dim=1).median()

                self.running_percentile = (
                    self.ema_decay * self.running_percentile +
                    (1 - self.ema_decay) * percentile
                )

        # Compute loss (preserve gradients)
        if self.loss_type == "cvar":
            layer_loss = self._compute_cvar_loss(magnitudes)
        elif self.loss_type == "hinged":
            layer_loss = self._compute_hinged_loss(magnitudes)
        else:
            layer_loss = torch.tensor(0.0, device=device, requires_grad=True)

        return layer_loss

    def _compute_cvar_loss(self, magnitudes: torch.Tensor) -> torch.Tensor:
        """CVaR loss computation."""
        threshold = self.threshold
        exceedances = torch.relu(magnitudes - threshold)
        per_token_cvar = exceedances.mean(dim=-1)
        per_token_cvar = per_token_cvar * self.cvar_coefficient + threshold
        return per_token_cvar.mean()

    def _compute_hinged_loss(self, magnitudes: torch.Tensor) -> torch.Tensor:
        """Hinged loss computation."""
        threshold = self.current_threshold
        exceedances = torch.relu(magnitudes - threshold)
        per_token_loss = exceedances.mean(dim=-1)
        return per_token_loss.mean()

    def get_stats(self) -> Dict:
        """Get statistics for monitoring."""
        return {
            'global_max': self.global_max.item(),
            'batch_max': self.global_max.item(),
            'running_percentile': self.running_percentile.item(),
            'loss_type': self.loss_type,
            'current_threshold': (
                self.threshold.item() if self.loss_type == "cvar"
                else self.current_threshold.item()
            )
        }

    def reset_stats(self):
        """Reset statistics."""
        self.global_max.zero_()
        self.running_percentile.fill_(
            self.threshold if self.loss_type == "cvar" else self.current_threshold
        )


class StreamingOutlierTracker:
    """
    Gradient-preserving streaming outlier tracker.
    """

    def __init__(self, loss_type: str = "cvar", **config):
        self.streaming_loss = StreamingOutlierLoss(loss_type=loss_type, **config)
        self.enabled = True
        self.layer_losses = []  # Store individual layer losses with gradients

    def track_layer(self, activations: torch.Tensor, layer_name: str = "") -> None:
        """Track outliers for a single layer."""
        if self.enabled:
            layer_loss = self.streaming_loss.compute_layer_loss(activations, layer_name)
            self.layer_losses.append(layer_loss)

    def get_loss(self) -> torch.Tensor:
        """Get accumulated loss with gradients and reset."""
        if not self.layer_losses:
            device = next(self.streaming_loss.parameters()).device
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Sum all layer losses while preserving gradients
        total_loss = torch.stack(self.layer_losses).sum()
        # avg_loss = total_loss / len(self.layer_losses)

        # Clear the list for next batch
        self.layer_losses.clear()

        return total_loss

    def get_stats(self) -> Dict:
        """Get tracking statistics."""
        stats = self.streaming_loss.get_stats()
        stats['layers_processed'] = len(self.layer_losses) if self.layer_losses else 0
        return stats

    def reset(self):
        """Reset tracker state."""
        self.streaming_loss.reset_stats()
        self.layer_losses.clear()

    def enable(self):
        """Enable tracking."""
        self.enabled = True

    def disable(self):
        """Disable tracking."""
        self.enabled = False


def create_streaming_tracker(loss_type: str = "cvar", **config) -> StreamingOutlierTracker:
    """
    Factory function to create streaming outlier tracker.
    """
    return StreamingOutlierTracker(loss_type=loss_type, **config)
