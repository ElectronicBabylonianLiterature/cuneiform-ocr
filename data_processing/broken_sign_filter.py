#!/usr/bin/env python3
"""Filter broken cuneiform signs from EBL API data"""

from typing import List, Dict, Any
from .sign_resolver import SignResolver


class BrokenSignFilter:
    
    # Unicode subscript digit mapping
    SUBSCRIPT_MAP = {
        '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
        '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉'
    }
    
    @staticmethod
    def is_broken_away(token: Dict[str, Any]) -> bool:
        """Check if token is completely broken (BROKEN_AWAY in enclosureType)"""
        return 'BROKEN_AWAY' in token.get('enclosureType', [])
    
    @staticmethod
    def format_sign_name(name: str, sub_index: int) -> str:
        """Format sign name with subscript"""
        if sub_index is None or sub_index == 1:
            return name
        
        # Convert digits to Unicode subscripts
        sub_str = str(sub_index)
        subscript = ''.join(BrokenSignFilter.SUBSCRIPT_MAP.get(c, c) for c in sub_str)
        return f"{name}{subscript}"
    
    @staticmethod
    def extract_signs_from_token(token: Dict[str, Any], filter_broken: bool = True) -> List[str]:
        """Recursively extract sign names from token"""
        signs = []
        
        if filter_broken and BrokenSignFilter.is_broken_away(token):
            return signs
        
        token_type = token.get('type', '')
        
        # Directly named sign types
        if token_type in ['Reading', 'Logogram', 'Number']:
            if 'name' in token and token['name']:
                name = token['name']
                sub_index = token.get('subIndex', 1)
                signs.append(BrokenSignFilter.format_sign_name(name, sub_index))
        
        # Compound types with parts
        elif token_type in ['Word', 'AkkadianWord', 'GreekWord', 'LoneDeterminative']:
            if 'parts' in token:
                for part in token['parts']:
                    signs.extend(BrokenSignFilter.extract_signs_from_token(part, filter_broken))
        
        # CompoundGrapheme
        elif token_type == 'CompoundGrapheme':
            if filter_broken and 'BROKEN_AWAY' in token.get('enclosureType', []):
                return signs
            if 'compound_parts' in token:
                signs.extend(token['compound_parts'])
            elif 'cleanValue' in token:
                signs.append(token['cleanValue'])
        
        # Grapheme
        elif token_type == 'Grapheme':
            if 'name' in token:
                name = token['name']
                sub_index = token.get('subIndex', 1)
                signs.append(BrokenSignFilter.format_sign_name(name, sub_index))
        
        # Variant
        elif token_type in ['Variant', 'Variant2']:
            if 'tokens' in token:
                for variant_token in token['tokens']:
                    signs.extend(BrokenSignFilter.extract_signs_from_token(variant_token, filter_broken))
        
        # Divider
        elif token_type == 'Divider':
            if 'divider' in token:
                signs.append(token['divider'])
        
        # Other types with parts
        elif 'parts' in token:
            for part in token['parts']:
                signs.extend(BrokenSignFilter.extract_signs_from_token(part, filter_broken))
        
        return signs
    
    @staticmethod
    def parse_text_lines(text_data: Dict[str, Any], filter_broken: bool = True) -> List[List[str]]:
        """Parse text.lines and extract sign names (TextLine type only)"""
        if not text_data or 'lines' not in text_data:
            return []
        
        result_lines = []
        
        for line in text_data['lines']:
            if line.get('type') != 'TextLine' or 'content' not in line:
                continue
            
            line_signs = []
            for token in line['content']:
                sign_names = BrokenSignFilter.extract_signs_from_token(token, filter_broken)
                line_signs.extend(sign_names)
            
            if line_signs:
                result_lines.append(line_signs)
        
        return result_lines


def filter_broken_from_text(text_data: Dict[str, Any]) -> List[List[str]]:
    """Filter broken signs"""
    return BrokenSignFilter.parse_text_lines(text_data, filter_broken=True)


def extract_all_signs(text_data: Dict[str, Any]) -> List[List[str]]:
    """Extract all signs including broken ones"""
    return BrokenSignFilter.parse_text_lines(text_data, filter_broken=False)
