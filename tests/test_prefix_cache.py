#!/usr/bin/env python3
"""
Test script for prefix token cache functionality
"""

import sys
import logging
from src.utils.prefix_cache import PrefixTokenCache

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_cache_functionality():
    """Test prefix token cache with both activation types"""
    
    print("=== Testing Prefix Token Cache ===\n")
    
    # Initialize cache
    cache = PrefixTokenCache()
    
    # Test cache statistics
    stats = cache.get_cache_stats()
    print(f"Cache Statistics:")
    print(f"  Activation types: {stats['activation_types']}")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Entries by type: {stats['entries_by_type']}")
    print()
    
    # Test hidden_state cache hits
    print("=== Testing hidden_state activation type ===")
    for threshold in [4, 5, 6, 7, 8, 9, 10]:
        tokens = cache.get_cached_tokens(
            activation_type='hidden_state',
            outlier_threshold=threshold,
            model_name='llama-2-7b'
        )
        if tokens:
            print(f"✅ Threshold {threshold}: Found {len(tokens)} cached tokens")
        else:
            print(f"❌ Threshold {threshold}: No cached tokens found")
    
    print()
    
    # Test down_proj cache hits
    print("=== Testing down_proj activation type ===")
    for threshold in [4, 5, 6, 7, 8]:
        tokens = cache.get_cached_tokens(
            activation_type='down_proj',
            outlier_threshold=threshold,
            model_name='llama-2-7b'
        )
        if tokens:
            print(f"✅ Threshold {threshold}: Found {len(tokens)} cached tokens")
        else:
            print(f"❌ Threshold {threshold}: No cached tokens found")
    
    print()
    
    # Test specific token sequences
    print("=== Testing specific token sequences ===")
    
    # Test down_proj threshold 6
    tokens_6 = cache.get_cached_tokens('down_proj', 6, 'llama-2-7b')
    expected_6 = [13, 29896, 29906, 29871, 353, 1]
    if tokens_6 == expected_6:
        print(f"✅ down_proj threshold 6: Correct tokens {tokens_6}")
    else:
        print(f"❌ down_proj threshold 6: Expected {expected_6}, got {tokens_6}")
    
    # Test down_proj threshold 7
    tokens_7 = cache.get_cached_tokens('down_proj', 7, 'llama-2-7b')
    expected_7 = [13, 29896, 29906, 525, 1]
    if tokens_7 == expected_7:
        print(f"✅ down_proj threshold 7: Correct tokens {tokens_7}")
    else:
        print(f"❌ down_proj threshold 7: Expected {expected_7}, got {tokens_7}")
    
    # Test hidden_state threshold 8
    tokens_hs_8 = cache.get_cached_tokens('hidden_state', 8, 'llama-2-7b')
    expected_hs_8 = [13, 29871, 29889, 29896, 29906, 1]
    if tokens_hs_8 == expected_hs_8:
        print(f"✅ hidden_state threshold 8: Correct tokens {tokens_hs_8}")
    else:
        print(f"❌ hidden_state threshold 8: Expected {expected_hs_8}, got {tokens_hs_8}")
    
    print()
    
    # Test cache misses
    print("=== Testing cache misses ===")
    
    # Test non-existent threshold
    miss_tokens = cache.get_cached_tokens('down_proj', 15, 'llama-2-7b')
    if miss_tokens is None:
        print("✅ Cache miss for down_proj threshold 15: Correctly returned None")
    else:
        print("❌ Cache miss test failed: Should have returned None")
    
    # Test non-existent activation type
    miss_tokens = cache.get_cached_tokens('invalid_type', 6, 'llama-2-7b')
    if miss_tokens is None:
        print("✅ Cache miss for invalid activation type: Correctly returned None")
    else:
        print("❌ Cache miss test failed: Should have returned None")
    
    print("\n=== Cache Test Completed ===")

if __name__ == "__main__":
    test_cache_functionality()