
import os
from typing import List, Optional, Tuple
from pathlib import Path
import cv2
import requests
from .sign import SignResolver

from .bounding_box import BoundingBox, Detection, GroundTruths


class LocalDataSource:    
    def __init__(self, annotations_dir: str):
        self.annotations_dir = Path(annotations_dir)
        self.imgs_path = self.annotations_dir / "imgs"
        self.annotations_path = self.annotations_dir / "annotations"
        
        if not self.imgs_path.exists():
            raise ValueError(f"Images directory not found: {self.imgs_path}")
        if not self.annotations_path.exists():
            raise ValueError(f"Annotations directory not found: {self.annotations_path}")
    
    def get_available_fragments(self) -> List[str]:
        fragments = []
        for img_file in os.listdir(self.imgs_path):
            if img_file.endswith(('.jpg', '.jpeg', '.png')):
                fragment_id = os.path.splitext(img_file)[0]
                # Check if annotation exists
                gt_file = self.annotations_path / f"gt_{fragment_id}.txt"
                if gt_file.exists():
                    fragments.append(fragment_id)
        return fragments
    
    def load_image(self, fragment_id: str) -> Optional[cv2.Mat]:
        possible_names = [
            f"{fragment_id}.jpg",
            f"{fragment_id}.jpeg",
            f"{fragment_id}.png",
        ]
        for name in possible_names:
            filepath = self.imgs_path / name
            if filepath.exists():
                return cv2.imread(str(filepath))
        return None
    
    def load_annotation(self, fragment_id: str) -> Optional[GroundTruths]:
        """
        Load ground truth annotations for a fragment.
        
        Args:
            fragment_id: Fragment identifier
            
        Returns:
            List of BoundingBox objects or None if not found
        """
        gt_file = self.annotations_path / f"gt_{fragment_id}.txt"
        if not gt_file.exists():
            return None
        
        boxes = []
        with open(gt_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 5:
                    x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                    sign_name = parts[4]
                    
                    # Convert to Sign object
                    sign = SignResolver.resolve(sign_name, expected_type='SIGN')
                    
                    # Create BoundingBox with x1, y1, x2, y2 format
                    bbox = BoundingBox(
                        x1=float(x),
                        y1=float(y),
                        x2=float(x + w),
                        y2=float(y + h),
                        score=1.0,  # Ground truth has full confidence
                        sign=sign
                    )
                    boxes.append(bbox)
        
        return boxes if boxes else None


class EBLAPISource:    
    BASE_URL = "https://ebl.badw.de/api"
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
    
    def get_fragment_data(self, fragment_id: str) -> Optional[dict]:

        url = f"{self.BASE_URL}/fragments/{fragment_id}"
        try:
            response = requests.get(url, timeout=self.timeout)
            if response.status_code == 200:
                return response.json()
        except requests.RequestException as e:
            print(f"API request failed for fragment {fragment_id}: {e}")
        return None
    
    def get_signs(self, fragment_id: str) -> Optional[str]:
        """
        Get the pre-processed 'signs' field from fragment API.
        This field already has broken signs filtered out.
        
        Args:
            fragment_id: Fragment identifier
            
        Returns:
            Signs text string or None
        """
        data = self.get_fragment_data(fragment_id)
        if data:
            return data.get('signs', None)
        return None
    
    def get_text_data(self, fragment_id: str) -> Optional[dict]:
        """
        Get the full 'text' field from fragment API.
        This field contains structured line and token data with broken sign markers.
        
        Args:
            fragment_id: Fragment identifier
            
        Returns:
            Text data dictionary or None
        """
        data = self.get_fragment_data(fragment_id)
        if data:
            return data.get('text', None)
        return None
    
    def get_signs_filtered(self, fragment_id: str, filter_broken: bool = True) -> Optional[List[List[str]]]:
        """
        Get signs from text.lines field with broken sign filtering.
        This provides more control over filtering compared to the pre-processed 'signs' field.
        
        Args:
            fragment_id: Fragment identifier
            filter_broken: If True, filter out completely broken away signs
            
        Returns:
            List of lines with sign names, or None if fragment not found
        """
        text_data = self.get_text_data(fragment_id)
        if text_data:
            return SignTextParser.parse_text_lines(text_data, filter_broken)
        return None


class SignTextParser:    
    @staticmethod
    def parse_api_signs(signs_text: str) -> List[List[str]]:
        if not signs_text:
            return []
        
        lines = []
        for line_text in signs_text.strip().split('\n'):
            line_signs = []
            for token in line_text.split():
                # Handle alternatives like ABZ579/ABZ129/ABZ312
                if '/' in token:
                    token = token.split('/')[0]  # take first alternative
                
                # Convert ABZ to sign name
                sign = SignResolver.resolve(token, expected_type='ABZ')
                line_signs.append(sign.name)
            
            if line_signs:
                lines.append(line_signs)
        return lines
    
    @staticmethod
    def parse_api_signs_with_abz(signs_text: str) -> List[List[Tuple[str, str]]]:
        if not signs_text:
            return []
        
        lines = []
        for line_text in signs_text.strip().split('\n'):
            line_signs = []
            for token in line_text.split():
                # Handle alternatives like ABZ579/ABZ129/ABZ312
                if '/' in token:
                    token = token.split('/')[0]  # take first alternative
                
                # Convert ABZ to sign name
                sign = SignResolver.resolve(token, expected_type='ABZ')
                line_signs.append((sign.abz, sign.name))
            
            if line_signs:
                lines.append(line_signs)
        return lines
    
    @staticmethod
    def _is_broken_away(token: dict) -> bool:
        """
        Check if a token is completely broken away.
        Based on ebl-frontend logic: effectiveEnclosure includes BROKEN_AWAY.
        
        Args:
            token: Token dictionary from API
            
        Returns:
            True if token is completely broken away
        """
        enclosure_type = token.get('enclosureType', [])
        return 'BROKEN_AWAY' in enclosure_type
    
    @staticmethod
    def _is_partially_broken(token: dict) -> bool:
        """
        Check if a token has partial (not complete) BROKEN_AWAY enclosure.
        This checks if token has parts with different enclosure states.
        
        Args:
            token: Token dictionary from API
            
        Returns:
            True if partially broken
        """
        # For named signs with parts, check if parts have mixed enclosure
        if 'nameParts' in token:
            enclosure_types = [part.get('enclosureType', []) for part in token['nameParts']]
            has_broken = any('BROKEN_AWAY' in et for et in enclosure_types)
            all_broken = all('BROKEN_AWAY' in et for et in enclosure_types)
            return has_broken and not all_broken
        return False
    
    @staticmethod
    def _extract_signs_from_token(token: dict, filter_broken: bool = True) -> List[str]:
        """
        Extract sign names from a token, recursively handling nested parts.
        
        Args:
            token: Token dictionary from API
            filter_broken: If True, skip tokens that are broken away
            
        Returns:
            List of sign names from this token
        """
        signs = []
        
        # Skip if completely broken away
        if filter_broken and SignTextParser._is_broken_away(token):
            return signs
        
        token_type = token.get('type', '')
        
        # Handle named signs (Reading, Logogram, Number)
        if token_type in ['Reading', 'Logogram', 'Number']:
            # Extract the sign name
            if 'name' in token and token['name']:
                sign_name = token['name']
                # The API returns sign names like 'u', 'qa', 'tum', etc.
                # We need to resolve these to proper sign names
                signs.append(sign_name)
        
        # Handle Word and AkkadianWord - process parts recursively
        elif token_type in ['Word', 'AkkadianWord', 'GreekWord', 'LoneDeterminative']:
            if 'parts' in token:
                for part in token['parts']:
                    signs.extend(SignTextParser._extract_signs_from_token(part, filter_broken))
        
        # Handle CompoundGrapheme
        elif token_type == 'CompoundGrapheme':
            if filter_broken and 'BROKEN_AWAY' in token.get('enclosureType', []):
                return signs
            # Extract from compound
            if 'compound_parts' in token:
                signs.extend(token['compound_parts'])
            elif 'cleanValue' in token:
                # Some compound graphemes might store the value directly
                signs.append(token['cleanValue'])
        
        # Handle Grapheme
        elif token_type == 'Grapheme':
            if 'name' in token:
                signs.append(token['name'])
        
        # Handle Variant - process each variant option
        elif token_type in ['Variant', 'Variant2']:
            if 'tokens' in token:
                for variant_token in token['tokens']:
                    signs.extend(SignTextParser._extract_signs_from_token(variant_token, filter_broken))
        
        # Handle Divider
        elif token_type == 'Divider':
            if 'divider' in token:
                signs.append(token['divider'])
        
        # Recursively handle tokens with parts
        elif 'parts' in token:
            for part in token['parts']:
                signs.extend(SignTextParser._extract_signs_from_token(part, filter_broken))
        
        return signs
    
    @staticmethod
    def parse_text_lines(text_data: dict, filter_broken: bool = True) -> List[List[str]]:
        """
        Parse text.lines field from fragment API, extracting sign names and filtering broken signs.
        This method processes the structured token data from the API.
        
        Args:
            text_data: The 'text' field from fragment API response
            filter_broken: If True, filter out completely broken away signs
            
        Returns:
            List of lines, each containing list of sign names
        """
        if not text_data or 'lines' not in text_data:
            return []
        
        result_lines = []
        
        for line in text_data['lines']:
            line_type = line.get('type', '')
            
            # Only process TextLine types (skip SurfaceAtLine, NoteLine, etc.)
            if line_type != 'TextLine':
                continue
            
            if 'content' not in line:
                continue
            
            line_signs = []
            for token in line['content']:
                # Extract sign names from token
                sign_names = SignTextParser._extract_signs_from_token(token, filter_broken)
                
                # Resolve sign names to proper format
                for sign_name in sign_names:
                    try:
                        # Try to resolve as SIGN name (uppercase format like 'U', 'QA', etc.)
                        sign = SignResolver.resolve(sign_name.upper(), expected_type='SIGN')
                        line_signs.append(sign.name)
                    except:
                        # If resolution fails, keep the original name
                        line_signs.append(sign_name)
            
            if line_signs:
                result_lines.append(line_signs)
        
        return result_lines
    
    @staticmethod
    def parse_text_lines_with_abz(text_data: dict, filter_broken: bool = True) -> List[List[Tuple[str, str]]]:
        """
        Parse text.lines field from fragment API, extracting both ABZ codes and sign names.
        
        Args:
            text_data: The 'text' field from fragment API response
            filter_broken: If True, filter out completely broken away signs
            
        Returns:
            List of lines, each containing list of (abz, sign_name) tuples
        """
        if not text_data or 'lines' not in text_data:
            return []
        
        result_lines = []
        
        for line in text_data['lines']:
            line_type = line.get('type', '')
            
            # Only process TextLine types
            if line_type != 'TextLine':
                continue
            
            if 'content' not in line:
                continue
            
            line_signs = []
            for token in line['content']:
                # Extract sign names from token
                sign_names = SignTextParser._extract_signs_from_token(token, filter_broken)
                
                # Resolve sign names to proper format with ABZ codes
                for sign_name in sign_names:
                    try:
                        # Try to resolve as SIGN name
                        sign = SignResolver.resolve(sign_name.upper(), expected_type='SIGN')
                        line_signs.append((sign.abz, sign.name))
                    except:
                        # If resolution fails, include with empty ABZ
                        line_signs.append(('', sign_name))
            
            if line_signs:
                result_lines.append(line_signs)
        
        return result_lines


# ============ Convenience Functions ============

def create_local_source(annotations_dir: str) -> LocalDataSource:
    return LocalDataSource(annotations_dir)


def create_api_source(timeout: int = 10) -> EBLAPISource:
    return EBLAPISource(timeout)