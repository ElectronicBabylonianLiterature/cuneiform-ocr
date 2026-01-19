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

# ABZ class names from model
CLASSES = ['ABZ58', 'ABZ441', 'ABZ207', 'ABZ55', 'ABZ139', 'ABZ597', 'ABZ343', 'ABZ142', 'ABZ73', 'ABZ59', 'ABZ586', 'ABZ579', 'ABZ457', 'ABZ427', 'ABZ86', 'ABZ212', 'ABZ5', 'ABZ537', 'ABZ376', 'ABZ335', 'ABZ170', 'ABZ342', 'ABZ324', 'ABZ480', 'ABZ61', 'ABZ206', 'ABZ545', 'ABZ99', 'ABZ72', 'ABZ112', 'ABZ142a', 'ABZ396', 'ABZ103', 'ABZ13', 'ABZ70', 'ABZ69', 'ABZ437', 'ABZ381', 'X', 'ABZ279', 'ABZ52', 'ABZ128', 'ABZ97', 'ABZ151', 'ABZ465', 'ABZ461', 'ABZ595', 'ABZ468', 'ABZ1', 'ABZ449', 'ABZ318', 'ABZ384', 'ABZ214', 'ABZ111', 'ABZ367', 'ABZ84', 'ABZ319', 'ABZ62', 'ABZ314', 'ABZ556', 'ABZ7', 'ABZ230', 'ABZ74', 'ABZ144', 'ABZ331', 'ABZ330', 'ABZ598a', 'ABZ575', 'ABZ322', 'NoABZ0', 'ABZ6', 'ABZ354', 'ABZ172', 'ABZ399', 'ABZ328', 'ABZ471', 'ABZ332', 'ABZ593', 'ABZ233', 'ABZ148', 'ABZ538', 'ABZ12', 'ABZ57', 'ABZ481', 'ABZ313', 'ABZ167', 'ABZ15', 'ABZ68', 'ABZ353', 'ABZ398', 'ABZ532', 'ABZ371', 'ABZ231', 'ABZ80', 'ABZ314', 'ABZ295', 'ABZ115', 'ABZ411', 'ABZ308', 'ABZ191', 'ABZ296', 'ABZ412', 'ABZ565', 'ABZ401', 'ABZ589', 'ABZ211', 'ABZ472', 'ABZ570', 'ABZ79', 'ABZ75', 'ABZ298', 'ABZ420', 'ABZ535', 'ABZ134', 'ABZ536', 'ABZ101', 'ABZ533', 'ABZ536', 'ABZ126', 'ABZ94', 'ABZ9', 'ABZ232', 'ABZ393', 'ABZ60', 'ABZ104', 'ABZ131', 'ABZ306', 'ABZ38', 'ABZ470', 'ABZ557', 'ABZ333', 'NoABZ0', 'ABZ147', 'ABZ145', 'ABZ56', 'ABZ564', 'ABZ383', 'ABZ360', 'ABZ114', 'ABZ138', 'ABZ331e+152i', 'ABZ297', 'ABZ334', 'ABZ366', 'ABZ50', 'ABZ455', 'ABZ598b', 'ABZ339', 'ABZ205', 'ABZ78', 'ABZ87', 'ABZ554', 'ABZ85', 'ABZ536', 'ABZ312', 'ABZ69', 'ABZ433', 'ABZ124', 'ABZ164', 'ABZ129a', 'NoABZ0', 'ABZ76', 'ABZ326', 'ABZ143', 'ABZ440', 'ABZ559', 'ABZ307', 'ABZ374', 'ABZ74', 'ABZ451', 'ABZ574', 'NoABZ0', 'ABZ529']

# ============ MongoDB Connection for ABZ -> Sign Name ============
client = MongoClient("mongodb://wentaoche:thP1go8%40n%40me7L_&vkOI@badwcai-ebl01.srv.mwn.de:27017,badwcai-ebl02.srv.mwn.de:27018,badwcai-ebl03.srv.mwn.de:27019/ebl?ssl=true&retryWrites=true&loadBalanced=false&readPreference=primary&connectTimeoutMS=10000&authSource=ebl&authMechanism=SCRAM-SHA-1&tlsAllowInvalidCertificates=true")
db = client['ebl']
signs_collection = db['signs']

# Build ABZ -> Sign Name mapping (reverse lookup)
def build_abz_to_sign_mapping():
    """Build a mapping from ABZ numbers to sign names"""
    abz_to_sign = {}
    for doc in signs_collection.find({}):
        sign_name = doc['_id']
        for entry in doc.get('lists', []):
            if entry['name'] == 'ABZ':
                abz_key = 'ABZ' + entry['number']
                if abz_key not in abz_to_sign:
                    abz_to_sign[abz_key] = sign_name
    abz_to_sign['X'] = 'UnclearSign'
    abz_to_sign['NoABZ0'] = 'UnclearSign'
    return abz_to_sign

ABZ_TO_SIGN = build_abz_to_sign_mapping()

def abz_to_sign_name(abz):
    """Convert ABZ number to sign name"""
    return ABZ_TO_SIGN.get(abz, 'UnclearSign')

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
    """
    Parse signs text from API into structured format.
    Returns list of lines, each line is list of sign names.
    """
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
            sign_name = abz_to_sign_name(token)
            line_signs.append(sign_name)
        if line_signs:
            lines.append(line_signs)
    return lines



# ============ Detection ============
class BaseDetector(ABC):
    def __init__(self, model, classes: List[str], score_threshold: float = 0.5):
        self.model = model
        self.classes = classes
        self.score_threshold = score_threshold
    
    @abstractmethod
    def detect(self, img) -> List[Dict]:
        pass
    
    def _filter_detections(self, labels, bboxes, scores) -> List[Dict]:
        mask = scores > self.score_threshold
        labels = labels[mask]
        bboxes = bboxes[mask]
        scores = scores[mask]
        
        detections = []
        for i in range(len(labels)):
            bbox = bboxes[i]
            abz_name = self.classes[labels[i]]
            sign_name = abz_to_sign_name(abz_name)
            detections.append({
                'bbox': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                'abz_name': abz_name,
                'sign_name': sign_name,
                'score': float(scores[i])
            })
        
        return detections

class SingleImageDetector(BaseDetector):
    def detect(self, img) -> List[Dict]:
        result = inference_detector(self.model, img)
        OCR_result = result.pred_instances.cpu()
        
        labels = OCR_result['labels'].numpy()
        bboxes = OCR_result['bboxes'].numpy()
        scores = OCR_result['scores'].numpy()
        
        return self._filter_detections(labels, bboxes, scores)


@dataclass
class SingleImage:
    img: np.ndarray
    detections: List[Dict] = field(default_factory=list)
    
    def __len__(self):
        return len(self.detections)


class TabletImageDetector(BaseDetector):
    def __init__(self, model, classes: List[str], score_threshold: float = 0.5, 
                 visualize_crop: bool = False, logging_crop: bool = False, keep_crops: bool = False):
        super().__init__(model, classes, score_threshold)
        self.visualize_crop = visualize_crop
        self.logging_crop = logging_crop
        self.keep_crops = keep_crops
        self.cropped_images = []  
    
    def detect(self, img) -> List[Dict]:
        if self.keep_crops:
            self.cropped_images = [] # reset cropped for each detection

        cropped_images, crop_coordinates = divide_tablet_photo(
            img, 
            visualize=self.visualize_crop, 
            logging=self.logging_crop, 
            return_coordinates=True
        )
        
        # Create single image detector for processing cropped pieces
        single_detector = SingleImageDetector(self.model, self.classes, self.score_threshold)
        
        all_detections = []
        
        # Process each cropped piece
        for idx, img_piece in enumerate(cropped_images):
            # Use SingleImageDetector to detect signs in the cropped piece
            piece_detections = single_detector.detect(img_piece)

            if self.keep_crops:
                # Deep copy to avoid modifying stored detections when transforming coordinates
                self.cropped_images.append(SingleImage(img=img_piece, detections=copy.deepcopy(piece_detections)))
            
            # Transform to original image coordinates
            piece_offset_x = crop_coordinates[idx]['x']
            piece_offset_y = crop_coordinates[idx]['y']
            
            for det in piece_detections:
                bbox = det['bbox']
                det['bbox'] = [
                    bbox[0] + piece_offset_x,
                    bbox[1] + piece_offset_y,
                    bbox[2] + piece_offset_x,
                    bbox[3] + piece_offset_y
                ]
                all_detections.append(det)
        
        return all_detections
    
    def get_cropped_images(self) -> List[SingleImage]:
        return self.cropped_images
    

def group_detections_into_lines(detections):
    """Group detections into lines based on y-coordinate"""
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

def create_detection_heatmap(detections, img_shape, classes, scale_factor=10, avg_width=None, avg_height=None):
    """
    Create heatmap from detections.
    
    Args:
        detections: List of detection dicts with 'bbox', 'abz_name'
        img_shape: Tuple of (height, width) of the image
        classes: List of class names (ABZ)
        scale_factor: Scale factor to reduce heatmap size
        avg_width: Average sign width (computed from detections if None)
        avg_height: Average sign height (computed from detections if None)
    
    Returns:
        heatmap: numpy array of shape (height/scale, width/scale, num_classes)
        influence_radius: The influence radius used
        sigma: The sigma value used for Gaussian
    """
    img_height, img_width = img_shape[:2]
    num_classes = len(classes)
    
    # Use a smaller scale for heatmap to reduce memory
    heatmap_height = img_height // scale_factor
    heatmap_width = img_width // scale_factor
    
    # Initialize heatmap with reduced size
    heatmap = np.zeros((heatmap_height, heatmap_width, num_classes), dtype=np.float32)
    
    # Compute influence radius based on average dimensions (scaled)
    if avg_width is None or avg_height is None:
        avg_width, avg_height = compute_avg_dimensions(detections)
    
    influence_radius = (avg_width + avg_height) / 2 * 1.5 / scale_factor
    sigma_x = influence_radius / 3  # ~99.7% of Gaussian within 3 sigma
    sigma_y = influence_radius / 3
    
    # Generate heatmap for each detection
    for det in detections:
        # Get bounding box and class
        x1, y1, x2, y2 = det['bbox']
        class_id = classes.index(det['abz_name']) 
        
        # Compute center and scale to heatmap coordinates
        center_x = (x1 + x2) / 2 / scale_factor
        center_y = (y1 + y2) / 2 / scale_factor
        
        # Generate 2D Gaussian and add to corresponding class channel
        gaussian = create_2d_gaussian(center_x, center_y, heatmap_width, heatmap_height, sigma_x, sigma_y)
        heatmap[:, :, class_id] = np.maximum(heatmap[:, :, class_id], gaussian)
    
    return heatmap, influence_radius, sigma_x

def create_text_heatmap(text_lines, classes, avg_width, avg_height, scale_factor=10):
    """
    Create heatmap from text lines with synthetic sign positions.
    
    Args:
        text_lines: List of lines, each line is list of sign names
        classes: List of class names (ABZ)
        avg_width: Average sign width
        avg_height: Average sign height
        scale_factor: Scale factor to reduce heatmap size
    
    Returns:
        heatmap: numpy array of shape (height/scale, width/scale, num_classes)
        margin: The margin used
        influence_radius: The influence radius used
        sigma: The sigma value used for Gaussian
    """
    num_classes = len(classes)
    
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
    heatmap_text = np.zeros((heatmap_height_text_scaled, heatmap_width_text_scaled, num_classes), dtype=np.float32)
    
    # Gaussian parameters
    influence_radius_text = (avg_width + avg_height) / 2 * 1.5 / scale_factor
    sigma_x_text = influence_radius_text / 3
    sigma_y_text = influence_radius_text / 3
    
    # Generate heatmap for each sign in text_lines
    for row_idx, line in enumerate(text_lines):
        for col_idx, sign_name in enumerate(line):
            # Find class_id for this sign_name
            class_id = None
            for i, abz_name in enumerate(classes):
                if abz_to_sign_name(abz_name) == sign_name:
                    class_id = i
                    break
            
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
    """
    Use normalized cross-correlation to find the best match position of detection heatmap in text heatmap.
    
    Args:
        detection_heatmap: numpy array of shape (h1, w1, num_classes)
        text_heatmap: numpy array of shape (h2, w2, num_classes)
        scale_factor: Scale factor used when creating heatmaps
    
    Returns:
        top_left_scaled: Tuple (x, y) of match position in scaled coordinates
        max_val: Correlation value of the best match
        top_left_original: Tuple (x, y) of match position in original coordinates
    """
    num_classes = detection_heatmap.shape[2]
    
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
    for class_idx in range(num_classes):
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

def create_text_based_detections(text_lines, classes, match_position_x, match_position_y, 
                                   margin, avg_width, avg_height, image_bounds):
    """
    Create text-based detections by assigning sign centers from text image into detection region.
    
    Args:
        text_lines: List of lines, each line is list of sign names
        classes: List of class names (ABZ)
        match_position_x: X coordinate of matched position in text heatmap (original coords)
        match_position_y: Y coordinate of matched position in text heatmap (original coords)
        margin: Margin used in text heatmap
        avg_width: Average sign width
        avg_height: Average sign height
        image_bounds: Tuple (width, height) of target image bounds
    
    Returns:
        detection_with_texts: List of detection dicts with 'bbox', 'abz_name', 'sign_name', 'score', 'center'
    """
    img_width, img_height = image_bounds
    detection_with_texts = []
    
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
                
                # Find the ABZ name for this sign
                abz_name = None
                for abz in classes:
                    if abz_to_sign_name(abz) == sign_name:
                        abz_name = abz
                        break
                
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
    """
    Align detected signs with text signs.
    Returns aligned_signs: list of dicts with bbox and sign_name for all text signs.
    """
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
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 90)
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
    def __init__(self):
        self.visualized_result = None
        self.fig = None
    
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
            bbox_viz = BboxVisualizer(boxes_color=(255, 0, 0))
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
            axes[row, col].set_title(f'Class {channels[i]}: {CLASSES[channels[i]] if channels[i] < len(CLASSES) else "Unknown"}, abz_name: {abz_to_sign_name(CLASSES[channels[i]]) if channels[i] < len(CLASSES) else "Unknown"}, sign_name: {abz_to_sign_name(CLASSES[channels[i]]) if channels[i] < len(CLASSES) else "Unknown"}')
            axes[row, col].axis('off')
            plt.colorbar(im, ax=axes[row, col], fraction=0.046)
        
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
    detector = TabletImageDetector(model, CLASSES, SCORE_THRESHOLD, visualize_crop=False, logging_crop=False)
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
    
    # Initialize model
    print("Loading detection model...")
    register_all_modules()
    model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device='cuda:0')
    print("Model loaded.")
    
    # Get available fragments
    fragments = get_available_fragments()
    print(f"Found {len(fragments)} fragments with both image and annotation")
    
    # Process samples
    results = []
    for i, fragment_id in enumerate(fragments[:SAMPLE_LIMIT]):
        result = process_fragment(model, fragment_id)
        if result:
            results.append(result)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r['fragment_id']}: GT={r['gt_count']}, Text={r['text_signs']}, "
              f"Det={r['detected']}, Located={r['located']}, Interp={r['interpolated']}")
    
    # Save results
    with open(os.path.join(OUTPUT_DIR, "alignment_summary.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {OUTPUT_DIR}/")
