"""
Cuneiform Signs Alignment
Aligns detected signs with unlocated text signs from ebl API.
"""

import json
import os
import requests
import copy
import numpy as np
import cv2
from typing import List, Dict
from dataclasses import dataclass, field
from PIL import Image, ImageDraw, ImageFont
from pymongo import MongoClient
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from data_processing.divide_photos import divide_tablet_photo

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Allow large image processing
Image.MAX_IMAGE_PIXELS = None

# ============ Configuration ============
ANNOTATIONS_DIR = os.path.expanduser("~/erc-work-data/data-of-cuneiform-ocr-data/filtered_annotations")
CONFIG_FILE = "configs/detr.py"
CHECKPOINT_FILE = os.path.expanduser("~/erc-work-data/retrained_models/detr-173/epoch_1000.pth")
SCORE_THRESHOLD = 0.5
Y_THRESHOLD = 35  # for grouping signs into lines
OUTPUT_DIR = "alignment_results"
SAMPLE_LIMIT = 5  # number of samples to process






    

# ============ Data Loading ============
def load_ground_truth(fragment_id):
    """Load ground truth annotations from file"""
    # Try different naming patterns
    possible_names = [
        f"gt_{fragment_id}.txt",
        f"gt_{fragment_id.replace('.', ',')}.txt",
    ]
    
    annotations_path = os.path.join(ANNOTATIONS_DIR, "annotations")
    for name in possible_names:
        filepath = os.path.join(annotations_path, name)
        if os.path.exists(filepath):
            boxes = []
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 5:
                        x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                        sign_name = parts[4]
                        boxes.append({
                            'bbox': [x, y, x+w, y+h],  # x1, y1, x2, y2
                            'sign_name': sign_name
                        })
            return boxes
    return None

def transform_gt_to_cropped_region(gt_boxes: List[Dict], crop_info: Dict) -> List[Dict]:
    """
    Transform ground truth boxes from full image coordinates to cropped region coordinates.
    
    Args:
        gt_boxes: List of ground truth boxes in full image coordinates
        crop_info: Dictionary with crop region info, must contain 'x', 'y', 'w', 'h' keys
                   where (x, y) is the top-left corner of the crop in the full image
    
    Returns:
        List of ground truth boxes in cropped region coordinates (only boxes that intersect the crop)
    """
    if not gt_boxes:
        return []
    
    crop_x = crop_info['x']
    crop_y = crop_info['y']
    crop_w = crop_info['w']
    crop_h = crop_info['h']
    
    transformed_boxes = []
    
    for box in gt_boxes:
        bbox = box['bbox']  # [x1, y1, x2, y2] in full image coordinates
        
        # Check if box intersects with crop region
        box_x1, box_y1, box_x2, box_y2 = bbox
        
        # Calculate intersection
        inter_x1 = max(box_x1, crop_x)
        inter_y1 = max(box_y1, crop_y)
        inter_x2 = min(box_x2, crop_x + crop_w)
        inter_y2 = min(box_y2, crop_y + crop_h)
        
        # Check if there's actual intersection
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            # Calculate center of original box
            box_cx = (box_x1 + box_x2) / 2
            box_cy = (box_y1 + box_y2) / 2
            
            # Check if center falls within crop region
            if crop_x <= box_cx < crop_x + crop_w and crop_y <= box_cy < crop_y + crop_h:
                # Transform to cropped coordinates
                new_x1 = box_x1 - crop_x
                new_y1 = box_y1 - crop_y
                new_x2 = box_x2 - crop_x
                new_y2 = box_y2 - crop_y
                
                # Clip to cropped region bounds
                new_x1 = max(0, new_x1)
                new_y1 = max(0, new_y1)
                new_x2 = min(crop_w, new_x2)
                new_y2 = min(crop_h, new_y2)
                
                transformed_boxes.append({
                    'bbox': [new_x1, new_y1, new_x2, new_y2],
                    'sign_name': box['sign_name']
                })
    
    return transformed_boxes

def load_image(fragment_id):
    """Load image for a fragment"""
    imgs_path = os.path.join(ANNOTATIONS_DIR, "imgs")
    possible_names = [
        f"{fragment_id}.jpg",
        f"{fragment_id}.jpeg",
        f"{fragment_id}.png",
    ]
    for name in possible_names:
        filepath = os.path.join(imgs_path, name)
        if os.path.exists(filepath):
            return cv2.imread(filepath)
    return None

def get_signs_from_api(fragment_id):
    """Get signs text from ebl API"""
    url = f"https://ebl.badw.de/api/fragments/{fragment_id}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get('signs', None)
    return None

def parse_api_signs(signs_text):
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
            converter = SignNameConverter(token, expected_type='ABZ')
            line_signs.append(converter.get_sign_name())
        if line_signs:
            lines.append(line_signs)
    return lines



# ============ Detection ============




# ============ Sub-Tablet Data Structure ============
@dataclass
class SignBox:
    """
    Represents a single sign bounding box with dual representation:
    - bbox format [x1, y1, x2, y2] for visualization (same as detection/ground truth format)
    - center format (cx, cy, w, h) for heatmap generation
    """
    abz_name: str
    sign_name: str
    score: float = 1.0
    # Primary storage: center format
    cx: float = 0.0
    cy: float = 0.0
    width: float = 0.0
    height: float = 0.0
    # Additional metadata
    row_idx: int = -1  # row index in text lines (-1 if from detection)
    col_idx: int = -1  # column index in text lines (-1 if from detection)
    
    @classmethod
    def from_bbox(cls, bbox: List[float], abz_name: str, sign_name: str = None, 
                  score: float = 1.0, row_idx: int = -1, col_idx: int = -1) -> 'SignBox':
        """Create SignBox from [x1, y1, x2, y2] bbox format"""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        if sign_name is None:
            converter = SignNameConverter(abz_name)
            sign_name = converter.get_sign_name()
        return cls(abz_name=abz_name, sign_name=sign_name, score=score,
                   cx=cx, cy=cy, width=width, height=height,
                   row_idx=row_idx, col_idx=col_idx)
    
    @classmethod
    def from_center(cls, cx: float, cy: float, width: float, height: float,
                    abz_name: str, sign_name: str = None, score: float = 1.0,
                    row_idx: int = -1, col_idx: int = -1) -> 'SignBox':
        """Create SignBox from center format (cx, cy, w, h)"""
        if sign_name is None:
            converter = SignNameConverter(abz_name)
            sign_name = converter.get_sign_name()
        return cls(abz_name=abz_name, sign_name=sign_name, score=score,
                   cx=cx, cy=cy, width=width, height=height,
                   row_idx=row_idx, col_idx=col_idx)
    
    @classmethod
    def from_detection(cls, detection: Dict) -> 'SignBox':
        """Create SignBox from existing detection dict format"""
        bbox = detection['bbox']
        return cls.from_bbox(
            bbox=bbox,
            abz_name=detection.get('abz_name', 'X'),
            sign_name=detection.get('sign_name'),
            score=detection.get('score', 1.0)
        )
    
    @property
    def bbox(self) -> List[float]:
        """Return bbox as [x1, y1, x2, y2] for visualization"""
        x1 = self.cx - self.width / 2
        y1 = self.cy - self.height / 2
        x2 = self.cx + self.width / 2
        y2 = self.cy + self.height / 2
        return [x1, y1, x2, y2]
    
    @property
    def center(self) -> tuple:
        """Return center as (cx, cy)"""
        return (self.cx, self.cy)
    
    @property
    def dimensions(self) -> tuple:
        """Return dimensions as (width, height)"""
        return (self.width, self.height)
    
    def to_detection_dict(self) -> Dict:
        """Convert to detection dict format for compatibility with existing visualizers"""
        return {
            'bbox': self.bbox,
            'abz_name': self.abz_name,
            'sign_name': self.sign_name,
            'score': self.score,
            'center': self.center
        }
    
    def translate(self, dx: float, dy: float) -> 'SignBox':
        """Return a new SignBox translated by (dx, dy)"""
        return SignBox(
            abz_name=self.abz_name,
            sign_name=self.sign_name,
            score=self.score,
            cx=self.cx + dx,
            cy=self.cy + dy,
            width=self.width,
            height=self.height,
            row_idx=self.row_idx,
            col_idx=self.col_idx
        )
    
    def scale(self, scale_x: float, scale_y: float = None) -> 'SignBox':
        """Return a new SignBox with scaled dimensions (keeps center)"""
        if scale_y is None:
            scale_y = scale_x
        return SignBox(
            abz_name=self.abz_name,
            sign_name=self.sign_name,
            score=self.score,
            cx=self.cx,
            cy=self.cy,
            width=self.width * scale_x,
            height=self.height * scale_y,
            row_idx=self.row_idx,
            col_idx=self.col_idx
        )
    
    def copy(self) -> 'SignBox':
        """Return a deep copy of this SignBox"""
        return SignBox(
            abz_name=self.abz_name,
            sign_name=self.sign_name,
            score=self.score,
            cx=self.cx,
            cy=self.cy,
            width=self.width,
            height=self.height,
            row_idx=self.row_idx,
            col_idx=self.col_idx
        )


@dataclass
class SubTablet:
    """
    Represents a sub-tablet region with image, bounding boxes, and heatmap.
    Can represent:
    - Detection results from model
    - Text-based bboxes (from API text alignment)
    - Full hypothetical tablet (text heatmap domain)
    - Intermediate optimization results
    """
    # Image data (can be None for text-only representations like full text heatmap)
    img: np.ndarray = None
    
    # Sign boxes in unified format
    sign_boxes: List[SignBox] = field(default_factory=list)
    
    # Heatmap (can be None if not computed yet)
    heatmap: np.ndarray = None
    
    # Metadata
    name: str = ""  # identifier (e.g., "detection", "text_aligned", "full_text")
    scale_factor: int = 10  # scale factor used for heatmap
    avg_width: float = 80.0  # average sign width
    avg_height: float = 80.0  # average sign height
    margin: float = 0.0  # margin used for text heatmap
    
    # Origin offset (for coordinate transformation between different SubTablets)
    origin_x: float = 0.0  # x offset of this sub-tablet in full text coordinate
    origin_y: float = 0.0  # y offset of this sub-tablet in full text coordinate
    
    def __len__(self):
        return len(self.sign_boxes)
    
    @property
    def shape(self) -> tuple:
        """Return image shape or estimated shape from sign boxes"""
        if self.img is not None:
            return self.img.shape[:2]
        elif self.sign_boxes:
            # Estimate from sign box bounds
            max_x = max(sb.cx + sb.width / 2 for sb in self.sign_boxes)
            max_y = max(sb.cy + sb.height / 2 for sb in self.sign_boxes)
            return (int(max_y + self.margin), int(max_x + self.margin))
        else:
            return (0, 0)
    
    @classmethod
    def from_detections(cls, img: np.ndarray, detections: List[Dict], 
                        name: str = "detection", 
                        avg_width: float = None, avg_height: float = None) -> 'SubTablet':
        """Create SubTablet from image and detection list"""
        sign_boxes = [SignBox.from_detection(det) for det in detections]
        
        # Compute avg dimensions if not provided
        if avg_width is None or avg_height is None:
            computed_width, computed_height = compute_avg_dimensions(detections)
            avg_width = avg_width or computed_width
            avg_height = avg_height or computed_height
        
        return cls(
            img=img,
            sign_boxes=sign_boxes,
            name=name,
            avg_width=avg_width,
            avg_height=avg_height
        )
    
    @classmethod
    def from_text_lines(cls, text_lines: List[List[str]], 
                        avg_width: float, avg_height: float,
                        margin: float = None,
                        name: str = "full_text") -> 'SubTablet':
        if margin is None:
            margin = max(avg_width, avg_height)
        
        sign_boxes = []
        for row_idx, line in enumerate(text_lines):
            for col_idx, sign_name in enumerate(line):
                # Calculate center position with margin
                cx = margin + col_idx * avg_width + avg_width / 2
                cy = margin + row_idx * avg_height + avg_height / 2
                
                # Convert sign_name to abz_name
                converter = SignNameConverter(sign_name, expected_type='SIGN')
                abz_name = converter.get_abz()
                
                sign_box = SignBox.from_center(
                    cx=cx, cy=cy,
                    width=avg_width, height=avg_height,
                    abz_name=abz_name, sign_name=sign_name,
                    row_idx=row_idx, col_idx=col_idx
                )
                sign_boxes.append(sign_box)
        
        return cls(
            img=None,
            sign_boxes=sign_boxes,
            name=name,
            avg_width=avg_width,
            avg_height=avg_height,
            margin=margin
        )
    
    def to_detection_list(self) -> List[Dict]:
        """Convert sign_boxes to detection dict list for visualization compatibility"""
        return [sb.to_detection_dict() for sb in self.sign_boxes]
    
    def get_sign_boxes_in_bounds(self, width: float, height: float, 
                                  offset_x: float = 0, offset_y: float = 0) -> List[SignBox]:
        """Get sign boxes whose centers fall within specified bounds"""
        result = []
        for sb in self.sign_boxes:
            rel_cx = sb.cx - offset_x
            rel_cy = sb.cy - offset_y
            if 0 <= rel_cx < width and 0 <= rel_cy < height:
                # Create translated copy
                translated = SignBox(
                    abz_name=sb.abz_name,
                    sign_name=sb.sign_name,
                    score=sb.score,
                    cx=rel_cx,
                    cy=rel_cy,
                    width=sb.width,
                    height=sb.height,
                    row_idx=sb.row_idx,
                    col_idx=sb.col_idx
                )
                result.append(translated)
        return result
    
    def create_heatmap(self, CLASSES_ABZ: List[str], scale_factor: int = None,
                       img_shape: tuple = None) -> np.ndarray:
        """Generate heatmap from sign_boxes"""
        if scale_factor is None:
            scale_factor = self.scale_factor
        else:
            self.scale_factor = scale_factor
        
        # Determine heatmap dimensions
        if img_shape is not None:
            img_height, img_width = img_shape[:2]
        elif self.img is not None:
            img_height, img_width = self.img.shape[:2]
        else:
            # Estimate from sign boxes with margin
            height, width = self.shape
            img_height, img_width = height, width
        
        num_classes = len(CLASSES_ABZ)
        heatmap_height = img_height // scale_factor
        heatmap_width = img_width // scale_factor
        
        # Initialize heatmap
        heatmap = np.zeros((heatmap_height, heatmap_width, num_classes), dtype=np.float32)
        
        # Generate heatmap for each sign box
        for sb in self.sign_boxes:
            # Find class index
            if sb.abz_name in CLASSES_ABZ:
                class_id = CLASSES_ABZ.index(sb.abz_name)
            else:
                continue  # Skip unknown classes
            
            # Scale center to heatmap coordinates
            center_x = sb.cx / scale_factor
            center_y = sb.cy / scale_factor
            
            # Use anisotropic Gaussian based on this sign box's dimensions
            sigma_x = sb.width * 1.5 / scale_factor / 3
            sigma_y = sb.height * 1.5 / scale_factor / 3
            
            # Generate 2D Gaussian
            gaussian = create_2d_gaussian(center_x, center_y, 
                                          heatmap_width, heatmap_height, 
                                          sigma_x, sigma_y)
            heatmap[:, :, class_id] = np.maximum(heatmap[:, :, class_id], gaussian)
        
        self.heatmap = heatmap
        return heatmap
    
    def extract_sub_region(self, offset_x: float, offset_y: float,
                           width: float, height: float,
                           img: np.ndarray = None,
                           name: str = "sub_region") -> 'SubTablet':
        """Extract a sub-region from this SubTablet with coordinate transformation"""
        # Get sign boxes in the specified region
        sub_sign_boxes = self.get_sign_boxes_in_bounds(width, height, offset_x, offset_y)
        
        return SubTablet(
            img=img,
            sign_boxes=sub_sign_boxes,
            name=name,
            avg_width=self.avg_width,
            avg_height=self.avg_height,
            margin=self.margin,
            origin_x=offset_x,
            origin_y=offset_y
        )
    
    def copy(self) -> 'SubTablet':
        """Return a deep copy of this SubTablet"""
        return SubTablet(
            img=self.img.copy() if self.img is not None else None,
            sign_boxes=[sb.copy() for sb in self.sign_boxes],
            heatmap=self.heatmap.copy() if self.heatmap is not None else None,
            name=self.name,
            scale_factor=self.scale_factor,
            avg_width=self.avg_width,
            avg_height=self.avg_height,
            margin=self.margin,
            origin_x=self.origin_x,
            origin_y=self.origin_y
        )
    
    def get_rows(self) -> List[List[SignBox]]:
        """Group sign boxes by row_idx and return as list of rows"""
        if not self.sign_boxes:
            return []
        
        # Group by row_idx
        rows_dict = {}
        for sb in self.sign_boxes:
            if sb.row_idx not in rows_dict:
                rows_dict[sb.row_idx] = []
            rows_dict[sb.row_idx].append(sb)
        
        # Sort each row by col_idx
        for row_idx in rows_dict:
            rows_dict[row_idx].sort(key=lambda sb: sb.col_idx)
        
        # Return sorted rows
        return [rows_dict[k] for k in sorted(rows_dict.keys())]


# ============ Elastic Chain Optimizer ============
import torch
import torch.nn.functional as F

class ElasticChainOptimizer:
    """
    Elastic Chain Model for refining text-aligned bounding boxes.
    
    Uses gradient-based optimization to minimize an energy function that combines:
    - L_data: Heatmap matching score (signs should be at high-response positions)
    - L_seq: Sequential constraint (signs in a row should be tightly distributed)
    - L_smooth: Height consistency (adjacent signs should have similar heights)
    - L_anchor: Line baseline constraint (signs shouldn't deviate too far from row baseline)
    """
    
    def __init__(self, 
                 sub_tablet_text: 'SubTablet',
                 detection_heatmap: np.ndarray,
                 CLASSES_ABZ: List[str],
                 scale_factor: int = 10,
                 lambda_data: float = 1.0,
                 lambda_seq: float = 0.1,
                 lambda_smooth: float = 0.05,
                 lambda_anchor: float = 0.02,
                 prior_aspect_ratio: float = 1.15,  # typical width/height ratio
                 device: str = None):
        """
        Initialize the optimizer.
        
        Args:
            sub_tablet_text: SubTablet with text-aligned sign boxes to optimize
            detection_heatmap: Heatmap from detection results (H, W, num_classes)
            CLASSES_ABZ: List of ABZ class names
            scale_factor: Scale factor used in heatmap
            lambda_data: Weight for data term (heatmap matching)
            lambda_seq: Weight for sequential constraint
            lambda_smooth: Weight for height smoothness
            lambda_anchor: Weight for baseline anchor
            prior_aspect_ratio: Prior width/height ratio for signs
            device: Torch device ('cuda' or 'cpu')
        """
        self.sub_tablet_text = sub_tablet_text
        self.CLASSES_ABZ = CLASSES_ABZ
        self.scale_factor = scale_factor
        self.lambda_data = lambda_data
        self.lambda_seq = lambda_seq
        self.lambda_smooth = lambda_smooth
        self.lambda_anchor = lambda_anchor
        self.prior_aspect_ratio = prior_aspect_ratio
        
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Convert heatmap to torch tensor
        self.heatmap = torch.from_numpy(detection_heatmap).float().to(self.device)
        
        # Get rows of sign boxes
        self.rows = sub_tablet_text.get_rows()
        self.num_rows = len(self.rows)
        
        # Build flat list and index mapping
        self.sign_boxes_flat = []
        self.row_indices = []  # which row each sign belongs to
        self.col_indices = []  # which column within row
        self.class_ids = []  # class id for each sign
        
        for row_idx, row in enumerate(self.rows):
            for col_idx, sb in enumerate(row):
                self.sign_boxes_flat.append(sb)
                self.row_indices.append(row_idx)
                self.col_indices.append(col_idx)
                # Find class id
                if sb.abz_name in CLASSES_ABZ:
                    self.class_ids.append(CLASSES_ABZ.index(sb.abz_name))
                else:
                    self.class_ids.append(-1)  # unknown class
        
        self.num_signs = len(self.sign_boxes_flat)
        
        # Initialize optimization parameters
        # params[i] = [cx, cy, w, h] for sign i
        init_params = []
        for sb in self.sign_boxes_flat:
            init_params.append([sb.cx, sb.cy, sb.width, sb.height])
        
        self.params = torch.tensor(init_params, dtype=torch.float32, 
                                   device=self.device, requires_grad=True)
        
        # Store initial params for reference
        self.initial_params = self.params.clone().detach()
        
        # Compute initial row baselines (average y for each row)
        self.row_baselines = []
        for row in self.rows:
            if row:
                avg_y = np.mean([sb.cy for sb in row])
                self.row_baselines.append(avg_y)
            else:
                self.row_baselines.append(0)
        self.row_baselines = torch.tensor(self.row_baselines, dtype=torch.float32, 
                                          device=self.device)
        
        # History for tracking optimization
        self.loss_history = []
        self.loss_components_history = []
    
    def compute_data_loss(self) -> torch.Tensor:
        """
        Compute data term: negative sum of heatmap scores at sign positions.
        L_data = -sum_i ScoreMap[class_i](x_i, y_i)
        
        Uses grid_sample for differentiable bilinear interpolation.
        """
        loss = torch.tensor(0.0, device=self.device)
        valid_count = 0
        
        heatmap_h, heatmap_w, num_classes = self.heatmap.shape
        
        for i in range(self.num_signs):
            class_id = self.class_ids[i]
            if class_id < 0 or class_id >= num_classes:
                continue
            
            # Get scaled position (differentiable)
            cx = self.params[i, 0] / self.scale_factor
            cy = self.params[i, 1] / self.scale_factor
            
            # Normalize to [-1, 1] for grid_sample
            # grid_sample expects normalized coordinates where:
            # -1 corresponds to left/top edge, +1 corresponds to right/bottom edge
            norm_x = (cx / (heatmap_w - 1)) * 2 - 1
            norm_y = (cy / (heatmap_h - 1)) * 2 - 1
            
            # Clamp to valid range for sampling
            norm_x = torch.clamp(norm_x, -1, 1)
            norm_y = torch.clamp(norm_y, -1, 1)
            
            # Create grid for sampling: shape (1, 1, 1, 2)
            grid = torch.stack([norm_x, norm_y]).view(1, 1, 1, 2)
            
            # Extract single channel heatmap: (1, 1, H, W)
            single_heatmap = self.heatmap[:, :, class_id].unsqueeze(0).unsqueeze(0)
            
            # Sample using grid_sample (differentiable bilinear interpolation)
            score = F.grid_sample(single_heatmap, grid, mode='bilinear', 
                                  padding_mode='border', align_corners=True)
            
            loss = loss - score.squeeze()
            valid_count += 1
        
        return loss / max(1, valid_count)
    
    def compute_seq_loss(self) -> torch.Tensor:
        """
        Compute sequential constraint: signs in a row should be adjacent.
        L_seq = sum_{i,j} ((x_{i,j+1} - x_{i,j}) - (w_{i,j} + w_{i,j+1})/2)^2
        """
        loss = torch.tensor(0.0, device=self.device)
        count = 0
        
        idx = 0
        for row in self.rows:
            row_len = len(row)
            for j in range(row_len - 1):
                # Get params for adjacent signs
                cx_j = self.params[idx + j, 0]
                cx_j1 = self.params[idx + j + 1, 0]
                w_j = self.params[idx + j, 2]
                w_j1 = self.params[idx + j + 1, 2]
                
                # Expected gap: (w_j + w_j1) / 2
                expected_gap = (w_j + w_j1) / 2
                actual_gap = cx_j1 - cx_j
                
                loss = loss + (actual_gap - expected_gap) ** 2
                count += 1
            
            idx += row_len
        
        return loss / max(1, count)
    
    def compute_smooth_loss(self) -> torch.Tensor:
        """
        Compute smoothness constraint:
        - Height consistency: (h_{i,j} - h_{i,j+1})^2
        - Aspect ratio prior: (w_{i,j} - prior_ratio * h_{i,j})^2
        """
        loss_height = torch.tensor(0.0, device=self.device)
        loss_aspect = torch.tensor(0.0, device=self.device)
        count_height = 0
        count_aspect = 0
        
        idx = 0
        for row in self.rows:
            row_len = len(row)
            for j in range(row_len):
                w_j = self.params[idx + j, 2]
                h_j = self.params[idx + j, 3]
                
                # Aspect ratio constraint
                loss_aspect = loss_aspect + (w_j - self.prior_aspect_ratio * h_j) ** 2
                count_aspect += 1
                
                # Height consistency with next sign
                if j < row_len - 1:
                    h_j1 = self.params[idx + j + 1, 3]
                    loss_height = loss_height + (h_j - h_j1) ** 2
                    count_height += 1
            
            idx += row_len
        
        loss = loss_height / max(1, count_height) + loss_aspect / max(1, count_aspect)
        return loss
    
    def compute_anchor_loss(self) -> torch.Tensor:
        """
        Compute anchor constraint: signs should stay near their row baseline.
        L_anchor = sum_{i,j} (y_{i,j} - baseline_i)^2
        """
        loss = torch.tensor(0.0, device=self.device)
        
        idx = 0
        for row_idx, row in enumerate(self.rows):
            baseline = self.row_baselines[row_idx]
            row_len = len(row)
            
            for j in range(row_len):
                cy = self.params[idx + j, 1]
                loss = loss + (cy - baseline) ** 2
            
            idx += row_len
        
        return loss / max(1, self.num_signs)
    
    def compute_total_loss(self) -> tuple:
        """Compute total loss and return all components"""
        L_data = self.compute_data_loss()
        L_seq = self.compute_seq_loss()
        L_smooth = self.compute_smooth_loss()
        L_anchor = self.compute_anchor_loss()
        
        L_total = (self.lambda_data * L_data + 
                   self.lambda_seq * L_seq + 
                   self.lambda_smooth * L_smooth + 
                   self.lambda_anchor * L_anchor)
        
        return L_total, L_data, L_seq, L_smooth, L_anchor
    
    def optimize(self, num_iterations: int = 100, lr: float = 1.0, 
                 verbose: bool = True, log_every: int = 10) -> 'SubTablet':
        """
        Run optimization and return optimized SubTablet.
        
        Args:
            num_iterations: Number of optimization iterations
            lr: Learning rate
            verbose: Print progress
            log_every: Print every N iterations
            
        Returns:
            Optimized SubTablet
        """
        optimizer = torch.optim.Adam([self.params], lr=lr)
        
        if verbose:
            print(f"Starting optimization with {self.num_signs} signs, {self.num_rows} rows")
            print(f"Lambdas: data={self.lambda_data}, seq={self.lambda_seq}, "
                  f"smooth={self.lambda_smooth}, anchor={self.lambda_anchor}")
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            L_total, L_data, L_seq, L_smooth, L_anchor = self.compute_total_loss()
            
            L_total.backward()
            optimizer.step()
            
            # Ensure positive widths and heights
            with torch.no_grad():
                self.params[:, 2] = torch.clamp(self.params[:, 2], min=10)  # min width
                self.params[:, 3] = torch.clamp(self.params[:, 3], min=10)  # min height
            
            # Record history
            self.loss_history.append(L_total.item())
            self.loss_components_history.append({
                'total': L_total.item(),
                'data': L_data.item(),
                'seq': L_seq.item(),
                'smooth': L_smooth.item(),
                'anchor': L_anchor.item()
            })
            
            if verbose and (iteration % log_every == 0 or iteration == num_iterations - 1):
                print(f"Iter {iteration:4d}: L_total={L_total.item():.4f}, "
                      f"L_data={L_data.item():.4f}, L_seq={L_seq.item():.4f}, "
                      f"L_smooth={L_smooth.item():.4f}, L_anchor={L_anchor.item():.4f}")
        
        return self.get_optimized_subtablet()
    
    def get_optimized_subtablet(self) -> 'SubTablet':
        """Create a new SubTablet with optimized parameters"""
        optimized_params = self.params.detach().cpu().numpy()
        
        # Create new sign boxes
        new_sign_boxes = []
        for i, sb in enumerate(self.sign_boxes_flat):
            new_sb = SignBox(
                abz_name=sb.abz_name,
                sign_name=sb.sign_name,
                score=sb.score,
                cx=float(optimized_params[i, 0]),
                cy=float(optimized_params[i, 1]),
                width=float(optimized_params[i, 2]),
                height=float(optimized_params[i, 3]),
                row_idx=sb.row_idx,
                col_idx=sb.col_idx
            )
            new_sign_boxes.append(new_sb)
        
        return SubTablet(
            img=self.sub_tablet_text.img,
            sign_boxes=new_sign_boxes,
            heatmap=None,
            name="optimized",
            scale_factor=self.sub_tablet_text.scale_factor,
            avg_width=self.sub_tablet_text.avg_width,
            avg_height=self.sub_tablet_text.avg_height,
            margin=self.sub_tablet_text.margin,
            origin_x=self.sub_tablet_text.origin_x,
            origin_y=self.sub_tablet_text.origin_y
        )
    
    def get_param_changes(self) -> np.ndarray:
        """Get the change in parameters from initial to current"""
        current = self.params.detach().cpu().numpy()
        initial = self.initial_params.cpu().numpy()
        return current - initial
    
    def plot_loss_history(self, figsize: tuple = (12, 4)):
        """Plot loss history"""
        if not self.loss_components_history:
            print("No optimization history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Total loss
        axes[0].plot(self.loss_history)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Total Loss')
        axes[0].set_title('Total Loss over Iterations')
        axes[0].grid(True)
        
        # Component losses
        components = ['data', 'seq', 'smooth', 'anchor']
        for comp in components:
            values = [h[comp] for h in self.loss_components_history]
            axes[1].plot(values, label=f'L_{comp}')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Loss Components over Iterations')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()



    

def group_detections_into_lines(detections):
    if not detections:
        return []
    
    # Sort by y-coordinate (center)
    sorted_dets = sorted(detections, key=lambda d: (d['bbox'][1] + d['bbox'][3]) / 2)
    
    lines = []
    current_line = [sorted_dets[0]]
    
    for det in sorted_dets[1:]:
        prev_y = (current_line[-1]['bbox'][1] + current_line[-1]['bbox'][3]) / 2
        curr_y = (det['bbox'][1] + det['bbox'][3]) / 2
        
        if curr_y - prev_y < Y_THRESHOLD:
            current_line.append(det)
        else:
            # Sort current line by x-coordinate
            current_line = sorted(current_line, key=lambda d: d['bbox'][0])
            lines.append(current_line)
            current_line = [det]
    
    if current_line:
        current_line = sorted(current_line, key=lambda d: d['bbox'][0])
        lines.append(current_line)
    
    return lines

def compute_avg_dimensions(detections):
    """Compute average width and height of detected signs"""
    if not detections:
        return 80, 80  # default
    
    widths = [d['bbox'][2] - d['bbox'][0] for d in detections]
    heights = [d['bbox'][3] - d['bbox'][1] for d in detections]
    
    return np.mean(widths), np.mean(heights)

# ============ Heatmap-based Alignment ============
def create_2d_gaussian(center_x, center_y, width, height, sigma_x, sigma_y):
    """Create a 2D Gaussian centered at (center_x, center_y)"""
    x = np.arange(0, width)
    y = np.arange(0, height)
    xx, yy = np.meshgrid(x, y)
    
    gaussian = np.exp(-((xx - center_x)**2 / (2 * sigma_x**2) + 
                        (yy - center_y)**2 / (2 * sigma_y**2)))
    return gaussian

def create_detection_heatmap(detections, img_shape, CLASSES_ABZ, scale_factor=10, avg_width=None, avg_height=None):
    img_height, img_width = img_shape[:2]
    num_CLASSES_ABZ = len(CLASSES_ABZ)
    
    # Use a smaller scale for heatmap to reduce memory
    heatmap_height = img_height // scale_factor
    heatmap_width = img_width // scale_factor
    
    # Initialize heatmap with reduced size
    heatmap = np.zeros((heatmap_height, heatmap_width, num_CLASSES_ABZ), dtype=np.float32)
    
    # Compute average dimensions for reference (scaled)
    if avg_width is None or avg_height is None:
        avg_width, avg_height = compute_avg_dimensions(detections)
    
    influence_radius = (avg_width + avg_height) / 2 * 1.5 / scale_factor  # for reference
    
    # Generate heatmap for each detection
    for det in detections:
        # Get bounding box and class
        x1, y1, x2, y2 = det['bbox']
        class_id = CLASSES_ABZ.index(det['abz_name']) 
        
        # Compute bbox dimensions
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        # Compute center and scale to heatmap coordinates
        center_x = (x1 + x2) / 2 / scale_factor
        center_y = (y1 + y2) / 2 / scale_factor
        
        # Use anisotropic Gaussian based on this bbox's dimensions
        sigma_x = bbox_width * 1.5 / scale_factor / 3  # ~99.7% of Gaussian within 3 sigma
        sigma_y = bbox_height * 1.5 / scale_factor / 3
        
        # Generate 2D Gaussian and add to corresponding class channel
        gaussian = create_2d_gaussian(center_x, center_y, heatmap_width, heatmap_height, sigma_x, sigma_y)
        heatmap[:, :, class_id] = np.maximum(heatmap[:, :, class_id], gaussian)
    
    return heatmap, influence_radius, sigma_x

def create_text_heatmap(text_lines, CLASSES_ABZ, avg_width, avg_height, scale_factor=10):
    num_CLASSES_ABZ = len(CLASSES_ABZ)
    
    # Determine grid dimensions for text_lines
    max_row_length = max(len(line) for line in text_lines) if text_lines else 1
    num_rows = len(text_lines)
    
    # Calculate heatmap dimensions with margins
    margin_width = avg_width
    margin_height = avg_height
    margin = max(margin_width, margin_height)  # Use the larger margin for both directions
    heatmap_width_text = int(max_row_length * avg_width + 2 * margin)
    heatmap_height_text = int(num_rows * avg_height + 2 * margin)
    
    # Use the same scale factor
    heatmap_width_text_scaled = heatmap_width_text // scale_factor
    heatmap_height_text_scaled = heatmap_height_text // scale_factor
    
    # Initialize heatmap
    heatmap_text = np.zeros((heatmap_height_text_scaled, heatmap_width_text_scaled, num_CLASSES_ABZ), dtype=np.float32)
    
    # Gaussian parameters - use anisotropic Gaussian
    sigma_x_text = avg_width * 1.5 / scale_factor / 3
    sigma_y_text = avg_height * 1.5 / scale_factor / 3
    influence_radius_text = (avg_width + avg_height) / 2 * 1.5 / scale_factor  # for reference
    
    # Build sign_name to class_id lookup (do this once instead of in nested loops)
    sign_name_to_class_id = {}
    for i, abz_name in enumerate(CLASSES_ABZ):
        converter = SignNameConverter(abz_name)
        sign_name_to_class_id[converter.get_sign_name()] = i
    
    # Generate heatmap for each sign in text_lines
    for row_idx, line in enumerate(text_lines):
        for col_idx, sign_name in enumerate(line):
            # Find class_id for this sign_name using the lookup
            class_id = sign_name_to_class_id.get(sign_name)
            
            if class_id is None:
                continue
            
            # Calculate center position in original coordinates
            center_y_orig = margin + row_idx * avg_height + avg_height / 2
            center_x_orig = margin + col_idx * avg_width + avg_width / 2
            
            # Scale to heatmap coordinates
            center_x_scaled = center_x_orig / scale_factor
            center_y_scaled = center_y_orig / scale_factor
            
            # Generate 2D Gaussian and add to corresponding class channel
            gaussian_text = create_2d_gaussian(center_x_scaled, center_y_scaled, 
                                              heatmap_width_text_scaled, heatmap_height_text_scaled, 
                                              sigma_x_text, sigma_y_text)
            heatmap_text[:, :, class_id] = np.maximum(heatmap_text[:, :, class_id], gaussian_text)
    
    return heatmap_text, margin, influence_radius_text, sigma_x_text

def match_heatmaps_ncc(detection_heatmap, text_heatmap, scale_factor=10):
    num_CLASSES_ABZ = detection_heatmap.shape[2]
    
    # Check if detection heatmap is larger than text heatmap
    # If so, we cannot perform template matching - return center position with low score
    if (detection_heatmap.shape[0] > text_heatmap.shape[0] or 
        detection_heatmap.shape[1] > text_heatmap.shape[1]):
        # Return center of text heatmap as the match position with low confidence
        center_y = text_heatmap.shape[0] // 2
        center_x = text_heatmap.shape[1] // 2
        top_left_scaled = (center_x - detection_heatmap.shape[1] // 2, 
                          center_y - detection_heatmap.shape[0] // 2)
        # Clamp to valid range
        top_left_scaled = (max(0, top_left_scaled[0]), max(0, top_left_scaled[1]))
        top_left_original = (top_left_scaled[0] * scale_factor, top_left_scaled[1] * scale_factor)
        return top_left_scaled, 0.1, top_left_original  # Low confidence score
    
    # Initialize combined correlation result
    combined_result = None
    valid_channels = 0
    
    # Calculate NCC for each class channel
    for class_idx in range(num_CLASSES_ABZ):
        # Extract single channel
        template_channel = detection_heatmap[:, :, class_idx]
        target_channel = text_heatmap[:, :, class_idx]
        
        # Skip if both template and target are empty (all zeros)
        if template_channel.max() == 0 and target_channel.max() == 0:
            continue
        
        # Normalize the template and target for this channel
        template_norm = cv2.normalize(template_channel, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
        target_norm = cv2.normalize(target_channel, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
        
        # Calculate normalized cross-correlation for this channel
        result_channel = cv2.matchTemplate(target_norm, template_norm, cv2.TM_CCORR_NORMED)
        
        # Accumulate results with equal weights
        if combined_result is None:
            combined_result = result_channel
        else:
            combined_result += result_channel
        valid_channels += 1
    
    # Check if we have any valid channels
    if combined_result is None or valid_channels == 0:
        # No valid matching, return center position with low score
        center_y = text_heatmap.shape[0] // 2
        center_x = text_heatmap.shape[1] // 2
        top_left_scaled = (center_x - detection_heatmap.shape[1] // 2, 
                          center_y - detection_heatmap.shape[0] // 2)
        top_left_scaled = (max(0, top_left_scaled[0]), max(0, top_left_scaled[1]))
        top_left_original = (top_left_scaled[0] * scale_factor, top_left_scaled[1] * scale_factor)
        return top_left_scaled, 0.1, top_left_original
    
    # Average the combined result (equal weights for all valid channels)
    combined_result = combined_result / valid_channels
    
    # Find the location of the best match in the combined result
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(combined_result)
    top_left_scaled = max_loc  # (x, y) in scaled coordinates
    
    # Convert to original (unscaled) coordinates
    top_left_original = (top_left_scaled[0] * scale_factor, top_left_scaled[1] * scale_factor)
    
    return top_left_scaled, max_val, top_left_original

def create_text_based_detections(text_lines, CLASSES_ABZ, match_position_x, match_position_y, 
                                   margin, avg_width, avg_height, image_bounds):
    img_width, img_height = image_bounds
    detection_with_texts = []
    
    # Build sign_name to abz_name lookup (do this once instead of in nested loops)
    sign_name_converter = SignNameConverter()
    
    # Iterate through text_lines to find signs that fall within the image region
    for row_idx, line in enumerate(text_lines):
        for col_idx, sign_name in enumerate(line):
            # Calculate center position in original text heatmap coordinates
            center_y_orig_text = margin + row_idx * avg_height + avg_height / 2
            center_x_orig_text = margin + col_idx * avg_width + avg_width / 2
            
            # Convert to coordinates relative to the matched position
            center_x_rel = center_x_orig_text - match_position_x
            center_y_rel = center_y_orig_text - match_position_y
            
            # Check if this center falls within the image bounds
            if (0 <= center_x_rel < img_width and 
                0 <= center_y_rel < img_height):
                
                # Create bbox using avg_width and avg_height
                x1 = center_x_rel - avg_width / 2
                y1 = center_y_rel - avg_height / 2
                x2 = center_x_rel + avg_width / 2
                y2 = center_y_rel + avg_height / 2
                
                # Find the ABZ name for this sign using the lookup
                sign_name_converter.set_signs(sign_name)
                abz_name = sign_name_converter.get_abz()
                
                if abz_name is None:
                    continue
                
                # Create detection dict
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'abz_name': abz_name,
                    'sign_name': sign_name,
                    'score': 1.0,  # Set high confidence for text-based detections
                    'center': (center_x_rel, center_y_rel)
                }
                detection_with_texts.append(detection)
    
    return detection_with_texts

# ============ Alignment Algorithm ============
def align_signs(detected_lines, text_lines, avg_width, avg_height):

    aligned_signs = []
    
    # Flatten detected signs with line info
    detected_flat = []
    for line_idx, line in enumerate(detected_lines):
        for det in line:
            detected_flat.append({**det, 'line_idx': line_idx, 'matched': False})
    
    # Process each text line
    for text_line_idx, text_line in enumerate(text_lines):
        line_aligned = []
        
        # Find closest detected line
        best_det_line_idx = None
        min_dist = float('inf')
        
        for det_line_idx, det_line in enumerate(detected_lines):
            if det_line:
                det_line_y = np.mean([(d['bbox'][1] + d['bbox'][3]) / 2 for d in det_line])
                # Estimate text line y based on line index
                estimated_y = text_line_idx * avg_height * 1.2  # rough estimate
                dist = abs(det_line_y - estimated_y) if text_line_idx < 3 else det_line_idx
                if det_line_idx == text_line_idx:
                    dist = 0  # prefer matching line indices
                if dist < min_dist:
                    min_dist = dist
                    best_det_line_idx = det_line_idx
        
        # Match signs in this line
        matched_positions = []  # (text_idx, det, position_x)
        
        for text_idx, text_sign in enumerate(text_line):
            # Find matching unmatched detection in this or nearby lines
            best_match = None
            best_score = float('inf')
            
            for det in detected_flat:
                if det['matched']:
                    continue
                # Prefer same line index
                line_penalty = abs(det['line_idx'] - (best_det_line_idx if best_det_line_idx is not None else text_line_idx)) * 1000
                
                if det['sign_name'] == text_sign:
                    x_center = (det['bbox'][0] + det['bbox'][2]) / 2
                    # Score based on expected position (proportional in line)
                    expected_x = text_idx * avg_width * 1.1
                    position_score = abs(x_center - expected_x) + line_penalty
                    
                    if position_score < best_score:
                        best_score = position_score
                        best_match = det
            
            if best_match is not None:
                best_match['matched'] = True
                matched_positions.append((text_idx, best_match, (best_match['bbox'][0] + best_match['bbox'][2]) / 2))
                line_aligned.append({
                    'text_idx': text_idx,
                    'sign_name': text_sign,
                    'bbox': best_match['bbox'],
                    'located': True
                })
            else:
                line_aligned.append({
                    'text_idx': text_idx,
                    'sign_name': text_sign,
                    'bbox': None,
                    'located': False
                })
        
        # Interpolate unlocated signs if at least 2 signs are located
        located_signs = [s for s in line_aligned if s['located']]
        
        if len(located_signs) >= 2:
            # Fit linear regression for x and y coordinates
            xs = [s['text_idx'] for s in located_signs]
            x_centers = [(s['bbox'][0] + s['bbox'][2]) / 2 for s in located_signs]
            y_centers = [(s['bbox'][1] + s['bbox'][3]) / 2 for s in located_signs]
            
            # Linear regression: x_center = a * text_idx + b
            coeffs_x = np.polyfit(xs, x_centers, 1)
            coeffs_y = np.polyfit(xs, y_centers, 1)
            
            for sign in line_aligned:
                if not sign['located']:
                    pred_x = np.polyval(coeffs_x, sign['text_idx'])
                    pred_y = np.polyval(coeffs_y, sign['text_idx'])
                    sign['bbox'] = [
                        pred_x - avg_width / 2,
                        pred_y - avg_height / 2,
                        pred_x + avg_width / 2,
                        pred_y + avg_height / 2
                    ]
                    sign['interpolated'] = True
        
        aligned_signs.extend(line_aligned)
    
    return aligned_signs

# ============ Visualization ============
class BboxVisualizer:
    def __init__(self, boxes_color=(0, 255, 0)):
        self.boxes_color = boxes_color
        self.label_key = 'sign_name'
        self.color_func = None
        self.visualized_result = None

    def draw_boxes(self, img, boxes):
        img_vis = img.copy()
        
        # First, draw all rectangles using OpenCV
        for box in boxes:
            bbox = box['bbox']
            if bbox is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            
            # Determine color: use color_func if provided, otherwise use default color
            box_color = self.color_func(box) if self.color_func else self.boxes_color
            box_color = tuple(reversed(box_color))  # Convert RGB to BGR for OpenCV
            
            # Draw rectangle using OpenCV (BGR color)
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), box_color, 2)
        
        # Convert to PIL for text drawing (supports Unicode)
        img_pil = Image.fromarray(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # Try to load a Unicode-supporting font, fallback to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
        except:
            font = ImageFont.load_default()
        
        # Draw labels using PIL
        for box in boxes:
            bbox = box['bbox']
            if bbox is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            
            # Draw label using PIL (RGB color, supports Unicode)
            label = box.get(self.label_key, '')[:10]
            if label:
                # Get text size
                bbox_text = draw.textbbox((0, 0), label, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
                
                # Draw black background
                draw.rectangle([x1, y1-text_height-10, x1+text_width+4, y1-2], fill=(0, 0, 0))
                # Draw white text
                draw.text((x1+2, y1-text_height-8), label, font=font, fill=(255, 255, 255))
        
        # Convert back to OpenCV format
        img_vis = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        self.visualized_result = img_vis
        return img_vis
    
    def display_result(self, vis_opt = "save", path=None):
        # vis_opt: "save" or "show"
        if vis_opt == "show" and self.visualized_result is not None:
            cv2.imshow('Visualization', self.visualized_result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif vis_opt == "draw":
            # draw using plt
            import matplotlib.pyplot as plt
            if self.visualized_result is not None:
                plt.imshow(cv2.cvtColor(self.visualized_result, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()
        elif vis_opt == "save" and self.visualized_result is not None:
            if path is None:
                print("visualization path not provided, saving to 'visualization_result.jpg'")
                path = 'visualization_result.jpg'
            cv2.imwrite(path, self.visualized_result)
        else:
            print("No visualization result to display or save.")

class TextVisualizer:
    def __init__(self, text_lines):
        self.text_lines = text_lines
    def write_text_file(self, filepath, fragment_id):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Fragment: {fragment_id}\n")
            f.write("=" * 50 + "\n")
            f.write("Text lines (converted from ABZ to sign names):\n")
            for i, line in enumerate(self.text_lines):
                f.write(f"Line {i+1}: {' '.join(line)}\n")

class HeatmapVisualizer:
    def __init__(self, bboxes_color=(255, 0, 0)):
        self.visualized_result = None
        self.fig = None
        self.bboxes_color = bboxes_color
        
    
    def draw_heatmap(self, img, heatmap, channels=(0, 1, 2), detection = None, texts = None):
        # Close previous figure to prevent memory leakage
        if self.fig is not None:
            plt.close(self.fig)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        if texts is not None:
            background_img = np.ones((heatmap.shape[0],  heatmap.shape[1], 3), dtype=np.uint8) * 255
            img = background_img
            # Display white background first
            axes[0, 0].imshow(background_img)
            axes[0, 0].set_title('Text Lines')
            axes[0, 0].axis('off')
            # Display text on the first subplot using normalized axes coordinates
            # Start from top (0.95) and go down
            text_y_start = 0.95
            text_y_step = 0.85 / max(len(texts), 1)  # Distribute evenly in the available space
            for i, line in enumerate(texts):
                line_text = ' '.join(line)
                axes[0, 0].text(0.05, text_y_start - i * text_y_step, line_text, 
                               fontsize=10, color='black', family='monospace',
                               transform=axes[0, 0].transAxes, verticalalignment='top')
        elif detection is not None:
            bbox_viz = BboxVisualizer(boxes_color=self.bboxes_color)
            bbox_viz.draw_boxes(img, detection)
            axes[0, 0].imshow(cv2.cvtColor(bbox_viz.visualized_result, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title('Detection with Bounding Box')
            axes[0, 0].axis('off')
        else:
            # Show original image
            axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
        
        # visualize heatmap channels
        for i in range(min(len(channels), 3)):  # Max 3 channels to fit in 2x2 grid
            row = (i + 1) // 2
            col = (i + 1) % 2
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, (heatmap.shape[1], heatmap.shape[0]))
            heatmap_channel = heatmap[:, :, channels[i]]
            
            # Create heatmap overlay
            axes[row, col].imshow(img_rgb, alpha=0.5)
            im = axes[row, col].imshow(heatmap_channel, cmap='hot', alpha=0.6, vmin=0, vmax=1)
            if channels[i] < len(CLASSES_ABZ):
                converter = SignNameConverter(CLASSES_ABZ[channels[i]])
                title = f'Class {channels[i]}: {converter.get_abz()} → {converter.get_sign_name()}'
            else:
                title = f'Class {channels[i]}: Unknown'
            axes[row, col].set_title(title)
            axes[row, col].axis('off')
            plt.colorbar(im, ax=axes[row, col], fraction=0.046)
        
        plt.tight_layout()
        self.fig = fig
        self.visualized_result = fig
        return fig
    
    def draw_heatmap_pca(self, img, heatmap, n_components=3, detection=None, texts=None, alpha=0.6):
        """
        Visualize heatmap using PCA to extract and display the first 3 principal components as false RGB.
        
        Args:
            img: Original image (can be None for text-only visualization)
            heatmap: Heatmap array with shape (H, W, num_classes)
            n_components: Number of PCA components to compute (default: 3 for RGB)
            detection: Optional detection boxes to overlay
            texts: Optional text lines to display
            alpha: Transparency for heatmap overlay (0-1)
        """
        # Close previous figure to prevent memory leakage
        if self.fig is not None:
            plt.close(self.fig)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Prepare background image
        if texts is not None:
            background_img = np.ones((heatmap.shape[0], heatmap.shape[1], 3), dtype=np.uint8) * 255
            img = background_img
        
        # Perform PCA on heatmap
        H, W, C = heatmap.shape
        
        # Reshape heatmap to (H*W, num_classes) for PCA
        heatmap_flat = heatmap.reshape(-1, C)
        
        # Apply PCA to extract first n_components
        pca = PCA(n_components=min(n_components, C))
        pca_result = pca.fit_transform(heatmap_flat)
        
        # Reshape back to spatial dimensions (H, W, n_components)
        pca_spatial = pca_result.reshape(H, W, -1)
        
        # Create false RGB from first 3 components
        false_rgb = np.zeros((H, W, 3), dtype=np.float32)
        for i in range(min(3, pca_spatial.shape[2])):
            false_rgb[:, :, i] = pca_spatial[:, :, i]
        
        # Normalize to [0, 1] for visualization
        false_rgb_normalized = np.zeros_like(false_rgb)
        for i in range(3):
            channel_min = false_rgb[:, :, i].min()
            channel_max = false_rgb[:, :, i].max()
            if channel_max > channel_min:
                false_rgb_normalized[:, :, i] = (false_rgb[:, :, i] - channel_min) / (channel_max - channel_min)
            else:
                false_rgb_normalized[:, :, i] = false_rgb[:, :, i]
        
        # Store explained variance for display
        explained_var = pca.explained_variance_ratio_[:3]
        
        # Subplot 1: Original image or text
        if texts is not None:
            axes[0].imshow(background_img)
            axes[0].set_title('Text Lines')
            axes[0].axis('off')
            text_y_start = 0.95
            text_y_step = 0.85 / max(len(texts), 1)
            for i, line in enumerate(texts):
                line_text = ' '.join(line)
                axes[0].text(0.05, text_y_start - i * text_y_step, line_text,
                           fontsize=10, color='black', family='monospace',
                           transform=axes[0].transAxes, verticalalignment='top')
        elif detection is not None:
            bbox_viz = BboxVisualizer(boxes_color=self.bboxes_color)
            bbox_viz.draw_boxes(img, detection)
            axes[0].imshow(cv2.cvtColor(bbox_viz.visualized_result, cv2.COLOR_BGR2RGB))
            axes[0].set_title('Detection with Bounding Box')
            axes[0].axis('off')
        elif img is not None:
            axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[0].set_title('Original Image')
            axes[0].axis('off')
        else:
            axes[0].axis('off')
        
        # Subplot 2: False RGB heatmap only (PCA components)
        axes[1].imshow(false_rgb_normalized)
        title_text = 'PCA False RGB Heatmap\n'
        title_text += f'R: PC1 ({explained_var[0]:.1%}) '
        title_text += f'G: PC2 ({explained_var[1]:.1%}) '
        title_text += f'B: PC3 ({explained_var[2]:.1%})'
        axes[1].set_title(title_text)
        axes[1].axis('off')
        
        # Subplot 3: Overlay on image
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb_resized = cv2.resize(img_rgb, (heatmap.shape[1], heatmap.shape[0]))
            # Normalize image to [0, 1]
            img_normalized = img_rgb_resized.astype(np.float32) / 255.0
            # Blend
            blended = img_normalized * (1 - alpha) + false_rgb_normalized * alpha
            blended = np.clip(blended, 0, 1)
            axes[2].imshow(blended)
            axes[2].set_title(f'Overlay (alpha={alpha})')
            axes[2].axis('off')
        else:
            axes[2].imshow(false_rgb_normalized)
            axes[2].set_title('False RGB Heatmap')
            axes[2].axis('off')
        
        plt.tight_layout()
        self.fig = fig
        self.visualized_result = fig
        return fig
    

    def display_result(self, vis_opt = "save", path=None):
        import matplotlib.pyplot as plt
        
        if vis_opt == "show" and self.visualized_result is not None:
            plt.show()
        elif vis_opt == "draw" and self.visualized_result is not None:
            plt.show()
        elif vis_opt == "save" and self.visualized_result is not None:
            if path is None:
                print("heatmap visualization path not provided, saving to 'heatmap_visualization_result.jpg'")
                path = 'heatmap_visualization_result.jpg'
            self.fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Heatmap visualization saved to {path}")
            plt.close(self.fig)
        else:
            print("No heatmap visualization result to display or save.")


def visualize_results(fragment_id, img, gt_boxes, detections, text_lines, aligned_signs, output_dir):
    """Create and save visualization results"""
    os.makedirs(output_dir, exist_ok=True)
    
    
    # 1. Ground truth boxes
    if gt_boxes:
        bbox_visualizer = BboxVisualizer(boxes_color=(0, 255, 0))
        vis_gt = bbox_visualizer.draw_boxes(img, gt_boxes)  # green
        cv2.imwrite(os.path.join(output_dir, f"{fragment_id}_1_ground_truth.jpg"), vis_gt)
    
    # 2. Detection results
    bbox_visualizer = BboxVisualizer(boxes_color=(255, 0, 0))
    vis_det = bbox_visualizer.draw_boxes(img, detections)  # blue
    cv2.imwrite(os.path.join(output_dir, f"{fragment_id}_2_detections.jpg"), vis_det)
    
    # 3. Print text conversion (ABZ -> sign names)
    text_visualizer = TextVisualizer(text_lines)
    text_filepath = os.path.join(output_dir, f"{fragment_id}_3_text_signs.txt")
    text_visualizer.write_text_file(text_filepath, fragment_id)
    
    # 4. Aligned bounding boxes
    # Color: located=blue, interpolated=red
    def get_aligned_color(sign):
        return (0, 0, 255) if sign.get('interpolated') else (255, 0, 0)
    
    bbox_visualizer = BboxVisualizer(color_func=get_aligned_color)
    vis_aligned = bbox_visualizer.draw_boxes(img, aligned_signs, label_key='sign_name')
    cv2.imwrite(os.path.join(output_dir, f"{fragment_id}_4_aligned.jpg"), vis_aligned)
    
    print(f"Saved visualizations to {output_dir}/{fragment_id}_*.jpg/txt")

# ============ Main ============
def process_fragment(model, fragment_id):
    """Process a single fragment"""
    print(f"\n{'='*60}")
    print(f"Processing fragment: {fragment_id}")
    
    # Load image
    img = load_image(fragment_id)
    if img is None:
        print(f"  Image not found for {fragment_id}")
        return None
    
    # Load ground truth
    gt_boxes = load_ground_truth(fragment_id)
    print(f"  Ground truth boxes: {len(gt_boxes) if gt_boxes else 0}")
    
    # Get signs from API
    signs_text = get_signs_from_api(fragment_id)
    if signs_text is None:
        print(f"  Could not get signs from API for {fragment_id}")
        return None
    
    print(f"  Raw API signs (first 200 chars): {signs_text[:200]}...")
    
    # Parse text signs (ABZ -> sign names)
    text_lines = parse_api_signs(signs_text)
    total_text_signs = sum(len(line) for line in text_lines)
    print(f"  Text lines: {len(text_lines)}, total signs: {total_text_signs}")
    
    # Detect signs
    detector = TabletImageDetector(model, CLASSES_ABZ, SCORE_THRESHOLD, visualize_crop=False, logging_crop=False)
    detections = detector.detect(img)
    print(f"  Detected signs (score > {SCORE_THRESHOLD}): {len(detections)}")
    
    # Compute average dimensions
    avg_width, avg_height = compute_avg_dimensions(detections)
    print(f"  Avg sign dimensions: {avg_width:.1f} x {avg_height:.1f}")
    
    # Group detections into lines
    detected_lines = group_detections_into_lines(detections)
    print(f"  Detected lines: {len(detected_lines)}")
    
    # Align signs
    aligned_signs = align_signs(detected_lines, text_lines, avg_width, avg_height)
    located_count = sum(1 for s in aligned_signs if s.get('located'))
    interpolated_count = sum(1 for s in aligned_signs if s.get('interpolated'))
    print(f"  Aligned: {located_count} located, {interpolated_count} interpolated")
    
    # Visualize
    visualize_results(fragment_id, img, gt_boxes, detections, text_lines, aligned_signs, OUTPUT_DIR)
    
    return {
        'fragment_id': fragment_id,
        'gt_count': len(gt_boxes) if gt_boxes else 0,
        'text_signs': total_text_signs,
        'detected': len(detections),
        'located': located_count,
        'interpolated': interpolated_count
    }

def get_available_fragments():
    """Get list of available fragments (have both image and annotation)"""
    imgs_path = os.path.join(ANNOTATIONS_DIR, "imgs")
    annotations_path = os.path.join(ANNOTATIONS_DIR, "annotations")
    
    fragments = []
    for img_file in os.listdir(imgs_path):
        if img_file.endswith(('.jpg', '.jpeg', '.png')):
            fragment_id = os.path.splitext(img_file)[0]
            # Check if annotation exists
            gt_file = os.path.join(annotations_path, f"gt_{fragment_id}.txt")
            if os.path.exists(gt_file):
                fragments.append(fragment_id)
    
    return fragments

if __name__ == "__main__":
    print("Cuneiform Signs Alignment")
    print("=" * 60)
    
    print("procedure not exists")