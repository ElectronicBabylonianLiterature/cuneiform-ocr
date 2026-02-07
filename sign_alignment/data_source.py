
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

        data = self.get_fragment_data(fragment_id)
        if data:
            return data.get('signs', None)
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


# ============ Convenience Functions ============

def create_local_source(annotations_dir: str) -> LocalDataSource:
    return LocalDataSource(annotations_dir)


def create_api_source(timeout: int = 10) -> EBLAPISource:
    return EBLAPISource(timeout)