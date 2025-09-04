#!/usr/bin/env python3
"""
Precomputed Prefix Token Cache System for PrefixQuant

This module provides a caching mechanism to avoid expensive recomputation of
get_prefixed_tokens() by using precomputed token sequences.
"""

import json
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path


# Precomputed prefix tokens based on experimental data from Llama-2-7b model
PRECOMPUTED_PREFIX_TOKENS = {
    "hidden_state": {
        4: [29896, 13, 29871, 29906, 29900, 29929, 29889, 525, 732, 29947, 2221, 278, 29941, 29946, 29883, 262, 29955, 353, 29945, 1919, 376, 29953, 29878, 29899, 29886, 326, 322, 29884, 277, 29876, 1332, 1038, 10800, 354, 324, 29885, 272, 345, 310, 29890, 284, 29877, 275, 450, 29875, 276, 1274, 869, 332, 263, 7268, 342, 261, 304, 297, 29874, 455, 381, 29893, 352, 2650, 459, 29892, 329, 17818, 2146, 3673, 1],
        5: [13, 29896, 29871, 29906, 29900, 29929, 29889, 525, 29947, 2221, 29941, 262, 29946, 29955, 29883, 732, 29945, 29878, 29953, 326, 353, 29886, 278, 277, 29884, 29876, 29899, 1332, 354, 324, 29885, 10800, 272, 1038, 29890, 345, 1],
        6: [13, 29896, 29871, 29906, 29889, 29900, 29929, 525, 29947, 2221, 29955, 29941, 29945, 29946, 262, 29953, 29883, 353, 29878, 1],
        7: [13, 29871, 29896, 29889, 29906, 29900, 29929, 29947, 29955, 29941, 1],
        8: [13, 29871, 29889, 29896, 29906, 1],
        9: [13, 29889, 29871, 1],
        10: [13, 29889, 29871, 1],
        64: [13, 29889, 29871, 1],
        # Note: threshold 11 and above may have empty token lists
    },
    "down_proj": {
        4: [13, 29871, 29896, 29906, 29900, 353, 278, 525, 310, 304, 1919, 263, 29941, 297, 313, 322, 29929, 29879, 450, 408, 29945, 29947, 29946, 869, 29953, 471, 3002, 472, 29955, 363, 373, 1723, 1],
        5: [13, 29896, 29906, 29871, 525, 353, 278, 29900, 304, 1919, 310, 29941, 1],
        6: [13, 29896, 29906, 29871, 353, 1],
        7: [13, 29896, 29906, 525, 1],
        64: [13, 29896, 29906, 29871, 1],
        # Note: threshold 8 and above not provided in experimental data
    },
    "q_k_up_gate": {
        4: [13, 29896, 29906, 29889, 29900, 29899, 525, 29883, 2221, 1],
        5: [13, 29889, 29896, 29906, 525, 353, 1],
        6: [13, 29889, 353, 29896, 29900, 1],
        7: [13, 29889, 353, 29896, 29900, 1],
        8: [13, 29889, 353, 29896, 29900, 1],
        9: [13, 29889, 353, 29896, 29900, 1],
        10: [13, 29889, 353, 1],
        11: [13, 29889, 353, 1],
        12: [13, 29889, 1],
        # Original wikitext2 perplexity: 5.472103595733643
    },
    "all": {
        4: [13, 29896, 29906, 29871, 869, 278, 29899, 1919, 353, 29900, 732, 29992, 376, 525, 29929, 322, 1723, 29889, 310, 262, 29879, 383, 2221, 323, 304, 360, 29883, 29947, 29878, 319, 315, 29941, 379, 297, 284, 317, 263, 405, 265, 450, 29884, 261, 29955, 390, 501, 277, 29953, 29886, 267, 29945, 1],
        5: [13, 29896, 29906, 29871, 29899, 278, 869, 353, 29889, 525, 1],
        6: [13, 29896, 29906, 29889, 29871, 525, 278, 353, 1],
        7: [13, 29896, 29906, 29889, 525, 29871, 29900, 1],
        8: [13, 29896, 29906, 29889, 525, 29900, 29899, 1],
        9: [13, 29896, 29889, 29906, 525, 29899, 1],
    }
}

# Expected perplexity results for validation
EXPECTED_PERPLEXITY = {
    "hidden_state": {
        4: 5.48888635635376,
        5: 5.492566108703613,
        6: 5.482699394226074,
        7: 5.472611904144287,
        8: 5.460096836090088,
        9: 5.474933624267578,
        10: 5.474933624267578,
        64: 5.474933624267578,
    },
    "down_proj": {
        4: 5.480926036834717,
        5: 5.483226776123047,
        6: 5.472769737243652,
        7: 5.473462104797363,
        64: 5.473462104797363,
    },
    "q_k_up_gate": {
        4: 5.481555461883545,
        5: 5.475161552429199,
        6: 5.471742153167725,
        7: 5.471742153167725,
        8: 5.471742153167725,
        9: 5.471742153167725,
        10: 5.462153911590576,
        11: 5.462153911590576,
        12: 5.470261573791504,
        # Original wikitext2 perplexity: 5.472103595733643
    },
    "all": {
        4: 5.483314514160156,
        5: 5.472009181976318,
        6: 5.458256721496582,
        7: 5.479524612426758,
        8: 5.48851203918457,
        9: 5.485409259796143,
        # All layers combined activation analysis
        # Best performance: threshold 6 (5.458256721496582)
    }
}

class PrefixTokenCache:
    """
    Cache manager for precomputed prefix tokens.

    Provides fast lookup of precomputed prefix tokens to avoid expensive
    computation via get_prefixed_tokens().
    """

    def __init__(self, cache_file: Optional[str] = None):
        """
        Initialize the prefix token cache.

        Args:
            cache_file: Optional path to JSON file for persistent caching
        """
        self.cache_file = cache_file
        self.logger = logging.getLogger(__name__)
        self._cache = PRECOMPUTED_PREFIX_TOKENS.copy()

        # Load from file if provided
        if cache_file and Path(cache_file).exists():
            self._load_from_file()

    def get_cached_tokens(self,
                         activation_type: str = 'hidden_state',
                         outlier_threshold: int = 8,
                         model_name: str = 'llama-2-7b') -> Optional[List[int]]:
        """
        Retrieve cached prefix tokens for given parameters.

        Args:
            activation_type: Type of activation ('hidden_state', 'down_proj', or 'q_k_up_gate')
            outlier_threshold: Outlier detection threshold (4-10)
            model_name: Model identifier (for future multi-model support)

        Returns:
            List of token IDs if found in cache, None otherwise
        """
        cache_key = self._build_cache_key(activation_type, outlier_threshold, model_name)

        if activation_type in self._cache:
            if outlier_threshold in self._cache[activation_type]:
                tokens = self._cache[activation_type][outlier_threshold]
                self.logger.info(f"Cache HIT: Found {len(tokens)} cached tokens for {cache_key}")
                return tokens

        self.logger.info(f"Cache MISS: No cached tokens for {cache_key}")
        return None

    def add_tokens(self,
                   tokens: List[int],
                   activation_type: str = 'hidden_state',
                   outlier_threshold: int = 8,
                   model_name: str = 'llama-2-7b',
                   perplexity: Optional[float] = None) -> None:
        """
        Add new tokens to cache.

        Args:
            tokens: List of token IDs to cache
            activation_type: Type of activation
            outlier_threshold: Outlier detection threshold
            model_name: Model identifier
            perplexity: Optional perplexity score for validation
        """
        if activation_type not in self._cache:
            self._cache[activation_type] = {}

        self._cache[activation_type][outlier_threshold] = tokens

        cache_key = self._build_cache_key(activation_type, outlier_threshold, model_name)
        self.logger.info(f"Cache ADD: Added {len(tokens)} tokens for {cache_key}")

        if perplexity:
            self.logger.info(f"  Perplexity: {perplexity:.6f}")

        # Save to file if configured
        if self.cache_file:
            self._save_to_file()

    def is_cached(self,
                  activation_type: str = 'hidden_state',
                  outlier_threshold: int = 8,
                  model_name: str = 'llama-2-7b') -> bool:
        """
        Check if tokens are available in cache.

        Args:
            activation_type: Type of activation
            outlier_threshold: Outlier detection threshold
            model_name: Model identifier

        Returns:
            True if tokens are cached, False otherwise
        """
        return (activation_type in self._cache and
                outlier_threshold in self._cache[activation_type])

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        stats = {
            'activation_types': list(self._cache.keys()),
            'total_entries': 0,
            'entries_by_type': {}
        }

        for activation_type, thresholds in self._cache.items():
            count = len(thresholds)
            stats['entries_by_type'][activation_type] = count
            stats['total_entries'] += count

        return stats

    def _build_cache_key(self, activation_type: str, outlier_threshold: int, model_name: str) -> str:
        """Build cache key for logging."""
        return f"{model_name}:{activation_type}:threshold={outlier_threshold}"

    def _load_from_file(self) -> None:
        """Load cache from JSON file."""
        try:
            with open(self.cache_file, 'r') as f:
                file_cache = json.load(f)

            # Merge with default cache
            for activation_type, thresholds in file_cache.items():
                if activation_type not in self._cache:
                    self._cache[activation_type] = {}
                self._cache[activation_type].update(thresholds)

            self.logger.info(f"Loaded cache from {self.cache_file}")

        except Exception as e:
            self.logger.warning(f"Failed to load cache from {self.cache_file}: {e}")

    def _save_to_file(self) -> None:
        """Save cache to JSON file."""
        try:
            # Ensure directory exists
            Path(self.cache_file).parent.mkdir(parents=True, exist_ok=True)

            with open(self.cache_file, 'w') as f:
                json.dump(self._cache, f, indent=2)

            self.logger.info(f"Saved cache to {self.cache_file}")

        except Exception as e:
            self.logger.error(f"Failed to save cache to {self.cache_file}: {e}")


def get_cached_prefixed_tokens(dataloader, model, tokenizer, model_name: str,
                              outlier_threshold: int = 8,
                              activation_type: str = 'hidden_state',
                              cache_file: Optional[str] = None,
                              force_recompute: bool = False) -> List[int]:
    """
    Get prefix tokens with caching support.

    This function first checks the cache for precomputed tokens. If not found,
    it falls back to the original get_prefixed_tokens() computation.

    Args:
        dataloader: Data loader for calibration
        model: The transformer model
        tokenizer: Model tokenizer
        model_name: Model identifier
        outlier_threshold: Outlier detection threshold
        activation_type: Type of activation ('hidden_state', 'down_proj', or 'q_k_up_gate')
        cache_file: Optional path for persistent caching
        force_recompute: If True, skip cache and recompute tokens

    Returns:
        List of prefix token IDs
    """
    cache = PrefixTokenCache(cache_file)
    logger = logging.getLogger(__name__)

    # Skip cache if forced recomputation
    if force_recompute:
        logger.info("Forced recomputation: skipping cache lookup")
    else:
        # Try cache first
        cached_tokens = cache.get_cached_tokens(
            activation_type=activation_type,
            outlier_threshold=outlier_threshold,
            model_name=model_name
        )

        if cached_tokens is not None:
            logger.info(f"Using cached prefix tokens: {len(cached_tokens)} tokens")
            return cached_tokens

    # Cache miss or forced recompute - use original function
    logger.info("Computing prefix tokens using get_prefixed_tokens()...")

    from utils.stat_utils import get_prefixed_tokens
    import time

    tick = time.time()
    computed_tokens = get_prefixed_tokens(
        dataloader, model, tokenizer, model_name,
        outlier_threshold=outlier_threshold,
        activation_type=activation_type
    )
    computation_time = time.time() - tick

    logger.info(f"Computed {len(computed_tokens)} tokens in {computation_time:.1f}s")

    # Add to cache for future use
    if not force_recompute:
        cache.add_tokens(
            computed_tokens,
            activation_type=activation_type,
            outlier_threshold=outlier_threshold,
            model_name=model_name
        )

    return computed_tokens
