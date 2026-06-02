
import json
import os
import re
from typing import List, Optional, Tuple
from pathlib import Path
import cv2
import numpy as np
import requests
from .sign import SignResolver

from .bounding_box import BoundingBox, Detection, GroundTruths


class SignAPIResolver:
    """Resolves cuneiform reading values to sign names via the EBL signs API, with file-based caching."""
    
    SIGNS_API_URL = "https://www.ebl.lmu.de/api/signs"
    
    def __init__(self, cache_file: str = None):
        if cache_file is None:
            cache_file = os.path.join(os.path.dirname(__file__), '.sign_api_cache.json')
        self._cache_file = cache_file
        self._cache = {}  # {"value|subIndex": sign_name_or_null}
        self._load_cache()
    
    def _load_cache(self):
        if os.path.exists(self._cache_file):
            try:
                with open(self._cache_file, 'r', encoding='utf-8') as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._cache = {}
    
    def _save_cache(self):
        with open(self._cache_file, 'w', encoding='utf-8') as f:
            json.dump(self._cache, f, indent=2, ensure_ascii=False)
    
    def _cache_key(self, value: str, sub_index: int) -> str:
        return f"{value}|{sub_index}"
    
    def resolve(self, value: str, sub_index: int) -> Optional[str]:
        """Resolve a reading value + sub_index to a sign name.
        
        For Reading tokens: pass name (lowercase) and subIndex.
        For Logogram tokens: pass name lowercased and subIndex.
        
        Returns:
            Sign name string like "IB", "GA", "TUR" or None if not found.
        """
        key = self._cache_key(value.lower(), sub_index)
        if key in self._cache:
            return self._cache[key]
        
        result = self._query_api(value.lower(), sub_index)
        self._cache[key] = result
        self._save_cache()
        return result
    
    def _query_api(self, value: str, sub_index: int) -> Optional[str]:
        try:
            resp = requests.get(
                self.SIGNS_API_URL,
                params={'value': value, 'subIndex': sub_index},
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                if data and isinstance(data, list) and len(data) > 0:
                    return data[0].get('name', None)
        except requests.RequestException:
            pass
        return None


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


class LocalTestDataSource:
    """
    Data source backed by a COCO-format dataset (val2017 split).

    Directory layout expected::

        coco_dir/
            val2017/              # subtablet images
            annotations/
                instances_val2017.json

    Each image file is treated as an independent fragment.  The ``fragment_id``
    is the filename without the file-type suffix (e.g. ``NBC.1712-0``).
    Subtablet images belonging to the same physical tablet are *not* merged.

    Use :class:`SubtabletEBLAPISource` as the API source when the EBL API
    must be queried with the bare fragment ID (without the ``-N`` suffix).
    """

    ANNOTATION_FILE = "annotations/instances_val2017.json"
    IMAGES_DIR = "val2017"

    def __init__(self, coco_dir: str):
        self.coco_dir = Path(coco_dir)
        self._data: Optional[dict] = None
        self._image_by_stem: dict = {}         # stem (no ext) -> image info dict
        self._annotations_by_image: dict = {}  # image_id -> [annotation, ...]
        self._category_names: dict = {}        # category_id -> sign_name

    def _ensure_loaded(self):
        if self._data is not None:
            return

        ann_path = self.coco_dir / self.ANNOTATION_FILE
        with open(ann_path, 'r', encoding='utf-8') as f:
            self._data = json.load(f)

        self._category_names = {c['id']: c['name'] for c in self._data['categories']}

        for img in self._data['images']:
            stem = img['file_name'].rsplit('.', 1)[0]
            self._image_by_stem[stem] = img

        for ann in self._data['annotations']:
            self._annotations_by_image.setdefault(ann['image_id'], [])
            self._annotations_by_image[ann['image_id']].append(ann)

    def get_available_fragments(self) -> List[str]:
        self._ensure_loaded()
        return sorted(self._image_by_stem.keys())

    def load_image(self, fragment_id: str) -> Optional[cv2.Mat]:
        self._ensure_loaded()
        img_info = self._image_by_stem.get(fragment_id)
        if img_info is None:
            return None
        path = self.coco_dir / self.IMAGES_DIR / img_info['file_name']
        return cv2.imread(str(path))

    def load_annotation(self, fragment_id: str) -> Optional[GroundTruths]:
        self._ensure_loaded()
        img_info = self._image_by_stem.get(fragment_id)
        if img_info is None:
            return None

        boxes: GroundTruths = []
        for ann in self._annotations_by_image.get(img_info['id'], []):
            if ann.get('iscrowd', 0):
                continue
            x, y, w, h = ann['bbox']
            sign_name = self._category_names[ann['category_id']]
            sign = SignResolver.resolve(sign_name, expected_type='SIGN')
            boxes.append(BoundingBox(
                x1=float(x),
                y1=float(y),
                x2=float(x + w),
                y2=float(y + h),
                score=1.0,
                sign=sign,
            ))
        return boxes if boxes else None


class EBLAPISource:
    BASE_URL = "https://ebl.badw.de/api"

    def __init__(self, timeout: int = 60, retries: int = 3):
        self.timeout = timeout
        self.retries = retries
        self._fragment_cache = {}  # in-memory cache: fragment_id -> data

    def get_fragment_data(self, fragment_id: str) -> Optional[dict]:
        if fragment_id in self._fragment_cache:
            return self._fragment_cache[fragment_id]

        url = f"{self.BASE_URL}/fragments/{fragment_id}"
        last_err = None
        for attempt in range(self.retries + 1):
            try:
                response = requests.get(url, timeout=self.timeout)
                if response.status_code == 200:
                    data = response.json()
                    self._fragment_cache[fragment_id] = data
                    return data
                last_err = f"HTTP {response.status_code}"
            except requests.RequestException as e:
                last_err = str(e)
        print(f"API request failed for fragment {fragment_id} after {self.retries+1} attempts: {last_err}")
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
    
    def get_signs_filtered(self, fragment_id: str, filter_broken: bool = True,
                           sign_resolver: 'SignAPIResolver' = None) -> Optional[List[List[str]]]:
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
            return SignTextParser.parse_text_lines(text_data, filter_broken, sign_resolver=sign_resolver)
        return None


class SubtabletEBLAPISource(EBLAPISource):
    """
    EBL API source that strips trailing subtablet suffixes (``-0``, ``-1``, …)
    from fragment IDs before querying the API.

    Use this together with :class:`LocalTestDataSource` so that the text
    transcription for a full tablet fragment is retrieved correctly even when
    the ``fragment_id`` refers to one subtablet image (e.g. ``NBC.1712-0``).
    """

    def get_fragment_data(self, fragment_id: str) -> Optional[dict]:
        api_id = re.sub(r'-\d+$', '', fragment_id)
        return super().get_fragment_data(api_id)


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
    def _extract_signs_from_token(token: dict, filter_broken: bool = True) -> List[Tuple[str, int]]:
        """
        Extract (reading_name, sub_index) tuples from a token, recursively handling nested parts.
        
        Container types (Word, Determinative, etc.) are always recursed into regardless
        of their enclosureType, because they may contain a mix of broken and non-broken parts.
        Only leaf sign-producing types (Reading, Logogram, etc.) are checked for broken status.
        
        Args:
            token: Token dictionary from API
            filter_broken: If True, skip tokens that are broken away
            
        Returns:
            List of (name, sub_index) tuples. sub_index=0 for CompoundGrapheme/UnclearSign.
        """
        signs = []
        token_type = token.get('type', '')
        
        # Container types - always recurse into parts (don't check broken at container level,
        # since a Word can span across broken brackets e.g. "DUMU-er-ṣe-tim]-ke₄")
        if token_type in ['Word', 'AkkadianWord', 'GreekWord', 'LoneDeterminative', 'Determinative']:
            for part in token.get('parts', []):
                signs.extend(SignTextParser._extract_signs_from_token(part, filter_broken))
            return signs
        
        # For sign-producing types, check broken status
        if filter_broken and SignTextParser._is_broken_away(token):
            return signs
        
        # Handle named signs (Reading, Logogram, Number)
        if token_type in ['Reading', 'Logogram', 'Number']:
            name = token.get('name', '')
            sub_index = token.get('subIndex', 1)
            if sub_index is None:
                sub_index = 1
            if name:
                signs.append((name, sub_index))
        
        # Handle CompoundGrapheme
        elif token_type == 'CompoundGrapheme':
            if filter_broken and 'BROKEN_AWAY' in token.get('enclosureType', []):
                return signs
            clean = token.get('cleanValue', '')
            if clean:
                signs.append((clean, 0))  # sub_index=0 signals direct sign name
        
        # Handle Grapheme
        elif token_type == 'Grapheme':
            name = token.get('name', '')
            if name:
                signs.append((name, 0))
        
        # Handle UnclearSign
        elif token_type == 'UnclearSign':
            signs.append(('X', 0))
        
        # Handle Variant - process first variant
        elif token_type in ['Variant', 'Variant2']:
            for variant_token in token.get('tokens', []):
                signs.extend(SignTextParser._extract_signs_from_token(variant_token, filter_broken))
                break  # take first variant only
        
        # Handle Divider
        elif token_type == 'Divider':
            divider = token.get('divider', '')
            if divider:
                signs.append((divider, 0))
        
        # Recursively handle other tokens with parts
        elif 'parts' in token:
            for part in token['parts']:
                signs.extend(SignTextParser._extract_signs_from_token(part, filter_broken))
        
        return signs
    
    @staticmethod
    def _resolve_sign_token(name: str, sub_index: int, sign_resolver: 'SignAPIResolver' = None) -> Optional[str]:
        """Resolve a (name, sub_index) tuple to a sign name."""
        if name == 'X':
            return 'UnclearSign'
        
        # sub_index=0 means direct sign name (CompoundGrapheme, Grapheme, Divider)
        if sub_index == 0:
            return name
        
        # Use signs API resolver if available
        if sign_resolver is not None:
            # For Logograms (uppercase names like DUMU), also query lowercase
            resolved = sign_resolver.resolve(name, sub_index)
            if resolved:
                return resolved
        
        # Fallback: try local SignResolver with uppercase name
        try:
            sign = SignResolver.resolve(name.upper(), expected_type='SIGN')
            if sign.name != 'UnclearSign':
                return sign.name
        except Exception:
            pass
        
        return name.upper()

    @staticmethod
    def parse_text_lines(text_data: dict, filter_broken: bool = True,
                         sign_resolver: 'SignAPIResolver' = None) -> List[List[str]]:
        """
        Parse text.lines field from fragment API, extracting sign names and filtering broken signs.
        Uses SignAPIResolver to properly resolve reading values to sign names.
        
        Args:
            text_data: The 'text' field from fragment API response
            filter_broken: If True, filter out completely broken away signs
            sign_resolver: SignAPIResolver instance for reading→sign name resolution.
                           If None, falls back to local SignResolver (less accurate).
            
        Returns:
            List of lines, each containing list of sign names
        """
        if not text_data or 'lines' not in text_data:
            return []
        
        result_lines = []
        
        for line in text_data['lines']:
            if line.get('type', '') != 'TextLine':
                continue
            if 'content' not in line:
                continue
            
            line_signs = []
            for token in line['content']:
                sign_tuples = SignTextParser._extract_signs_from_token(token, filter_broken)
                for name, sub_index in sign_tuples:
                    resolved = SignTextParser._resolve_sign_token(name, sub_index, sign_resolver)
                    if resolved:
                        line_signs.append(resolved)
            
            if line_signs:
                result_lines.append(line_signs)
        
        return result_lines
    
    @staticmethod
    def parse_text_lines_with_abz(text_data: dict, filter_broken: bool = True,
                                  sign_resolver: 'SignAPIResolver' = None) -> List[List[Tuple[str, str]]]:
        """
        Parse text.lines field from fragment API, extracting both ABZ codes and sign names.
        
        Args:
            text_data: The 'text' field from fragment API response
            filter_broken: If True, filter out completely broken away signs
            sign_resolver: SignAPIResolver instance for reading→sign name resolution.
            
        Returns:
            List of lines, each containing list of (abz, sign_name) tuples
        """
        if not text_data or 'lines' not in text_data:
            return []
        
        result_lines = []
        
        for line in text_data['lines']:
            if line.get('type', '') != 'TextLine':
                continue
            if 'content' not in line:
                continue
            
            line_signs = []
            for token in line['content']:
                sign_tuples = SignTextParser._extract_signs_from_token(token, filter_broken)
                for name, sub_index in sign_tuples:
                    resolved = SignTextParser._resolve_sign_token(name, sub_index, sign_resolver)
                    if resolved:
                        # Try to get ABZ code from local resolver
                        try:
                            sign = SignResolver.resolve(resolved, expected_type='SIGN')
                            line_signs.append((sign.abz, sign.name))
                        except Exception:
                            line_signs.append(('', resolved))
            
            if line_signs:
                result_lines.append(line_signs)
        
        return result_lines


# ============ Convenience Functions ============

def create_local_source(annotations_dir: str) -> LocalDataSource:
    return LocalDataSource(annotations_dir)


def create_api_source(timeout: int = 10) -> EBLAPISource:
    return EBLAPISource(timeout)