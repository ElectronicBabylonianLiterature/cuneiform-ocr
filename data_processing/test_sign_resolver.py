#!/usr/bin/env python3
"""Test SignToABZResolver functionality"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_processing.sign_resolver import SignToABZResolver

MONGODB_URI = "YOUR_MONGODB_URI"  # Replace with your actual MongoDB URI


def test_sign_resolver():
    print("=" * 70)
    print("Testing SignToABZResolver")
    print("=" * 70)
    
    with SignToABZResolver(MONGODB_URI) as resolver:
        # Test some common signs
        test_signs = [
            "u", "qa", "tum", "ma", "lu", "ka", 
            "mit", "ha", "ru", "iš", "te", "niš",
            "git", "AŠ", "DIŠ", "MIN"
        ]
        
        print("\nTesting individual sign conversion:")
        print("-" * 70)
        for sign in test_signs:
            abz = resolver.get_abz_number(sign)
            print(f"{sign:15s} -> {abz}")
        
        # Test complete string conversion
        test_strings = [
            "u qa tum",
            "AŠ ka",
            "DIŠ AŠ mit ha ru",
            "git ma lu"
        ]
        
        print("\n\nTesting string conversion:")
        print("-" * 70)
        for signs_str in test_strings:
            abz_str = resolver.convert_signs_to_abz(signs_str)
            print(f"Input: {signs_str}")
            print(f"Output: {abz_str}")
            print()
        
        # Display cache statistics
        stats = resolver.get_cache_stats()
        print("\nCache statistics:")
        print("-" * 70)
        print(f"Total cached signs: {stats['cached_signs']}")
        print(f"Found signs:     {stats['found']}")
        print(f"Not found signs:   {stats['not_found']}")
    
    print("\n✓ Testing complete")


if __name__ == "__main__":
    test_sign_resolver()
