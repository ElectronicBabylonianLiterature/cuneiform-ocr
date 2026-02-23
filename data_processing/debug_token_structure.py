#!/usr/bin/env python3
"""Debug: View the actual structure of tokens"""

import requests
import json

fragment_id = "ND.5437"
url = f"https://ebl.badw.de/api/fragments/{fragment_id}"

print(f"Fetching line 2 token structure for {fragment_id}...\n")

response = requests.get(url, timeout=15)
data = response.json()

# Fetch the first TextLine (should contain GAR-MEŠ-šu₂)
if data.get('text') and data['text'].get('lines'):
    for line_idx, line in enumerate(data['text']['lines']):
        if line.get('type') == 'TextLine':
            print("=" * 70)
            print(f"Line {line_idx}: type={line.get('type')}, prefix={line.get('prefix', 'None')}")
            print("=" * 70)
            
            if 'content' in line:
                for i, token in enumerate(line['content']):
                    print(f"\nToken {i}:")
                    print(json.dumps(token, indent=2, ensure_ascii=False))
                    print("-" * 70)
            
            # Display only the first 2 TextLines
            if line_idx >= 5:
                break
