#!/usr/bin/env python3
"""Test broken_sign_filter module - Fetch data from EBL API and compare filtering results"""

import sys
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_processing.broken_sign_filter import filter_broken_from_text, extract_all_signs
from data_processing.sign_resolver import SignToABZResolver

# MongoDB connection URI
MONGODB_URI = "YOUR_MONGODB_URI"  # Replace with your actual MongoDB URI


def test_with_api_fragment():
    """Fetch fragment data from API and compare filtering results"""
    
    fragment_id = "ND.5437"
    url = f"https://ebl.badw.de/api/fragments/{fragment_id}"
    
    print("=" * 70)
    print(f"Testing Fragment: {fragment_id}")
    print("=" * 70)
    
    # Initialize ABZ converter
    print("\nInitializing ABZ converter...")
    abz_resolver = SignToABZResolver(MONGODB_URI)
    
    try:
        print(f"\nFetching data: {url}")
        response = requests.get(url, timeout=15)
        
        if response.status_code != 200:
            print(f"✗ API request failed with status code: {response.status_code}")
            return
        
        data = response.json()
        
        # Get signs field (preprocessed, broken signs filtered)
        signs_field = data.get('signs', '')
        
        # Get text field (complete structured data)
        text_data = data.get('text')
        
        if not signs_field:
            print("⚠ No signs field found")
            return
        
        if not text_data:
            print("⚠ No text field found")
            return
        
        # Filtered results (extracted from text.lines and broken signs filtered)
        lines_filtered = filter_broken_from_text(text_data)
        
        # Unfiltered results (includes broken signs)
        lines_all = extract_all_signs(text_data)
        
        # Convert to ABZ format
        lines_filtered_abz = []
        for line in lines_filtered:
            signs_str = ' '.join(line)
            abz_str = abz_resolver.convert_signs_to_abz(signs_str)
            lines_filtered_abz.append(abz_str.split() if abz_str else [])
        
        # Output comparison
        print("\n" + "=" * 70)
        print("Before filtering (signs field - API preprocessed):")
        print("=" * 70)
        print(signs_field)
        
        print("\n" + "=" * 70)
        print("After filtering (extracted from text.lines with BROKEN_AWAY filtered):")
        print("=" * 70)
        for i, line in enumerate(lines_filtered, 1):
            print(f"Line {i}: {' '.join(line)}")
        
        print("\n" + "=" * 70)
        print("After filtering and converted to ABZ format:")
        print("=" * 70)
        for i, line in enumerate(lines_filtered_abz, 1):
            print(f"Line {i}: {' '.join(line)}")
        
        print("\n" + "=" * 70)
        print("Unfiltered (extracted from text.lines, including broken signs):")
        print("=" * 70)
        for i, line in enumerate(lines_all, 1):
            print(f"Line {i}: {' '.join(line)}")
        
        # Statistics
        signs_field_count = len(signs_field.split())
        filtered_count = sum(len(line) for line in lines_filtered)
        filtered_abz_count = sum(len(line) for line in lines_filtered_abz)
        all_count = sum(len(line) for line in lines_all)
        
        print("\n" + "=" * 70)
        print("Statistics:")
        print("=" * 70)
        print(f"Signs field count:           {signs_field_count}")
        print(f"Filtered signs count:        {filtered_count}")
        print(f"Filtered ABZ signs count:    {filtered_abz_count}")
        print(f"Unfiltered signs count:      {all_count}")
        print(f"Filtered out signs count:    {all_count - filtered_count}")
        print(f"Not converted to ABZ count:  {filtered_count - filtered_abz_count}")
        
        # ABZ converter cache statistics
        cache_stats = abz_resolver.get_cache_stats()
        print("\n" + "=" * 70)
        print("ABZ Converter Cache Statistics:")
        print("=" * 70)
        print(f"Total cached signs:    {cache_stats['cached_signs']}")
        print(f"Found signs:           {cache_stats['found']}")
        print(f"Not found signs:       {cache_stats['not_found']}")
        
        print("\n✓ Testing complete")
        
    except requests.RequestException as e:
        print(f"✗ Network request failed: {e}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        abz_resolver.close()


if __name__ == "__main__":
    test_with_api_fragment()
