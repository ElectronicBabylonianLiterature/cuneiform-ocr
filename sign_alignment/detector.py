
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict
import copy

from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules

import torch
import numpy as np

from data_processing.divide_photos import divide_tablet_photo

from .sign import SignResolver

@dataclass
class ModelConfig():
    config_file: str
    checkpoint_file: str
    device: str = 'cuda:0'
    
@dataclass
class SingleImage:
    img: np.ndarray
    detections: List[Dict] = field(default_factory=list)
    
    def __len__(self):
        return len(self.detections)

class BaseDetector(ABC):
    def __init__(self, model_config: ModelConfig, score_threshold: float = 0.5):
        print("Initializing detector, loading model...")
        register_all_modules()
        model_config.device = self._select_device(model_config.device)
        print(f"Using device: {model_config.device}")
        self.model = init_detector(model_config.config_file, model_config.checkpoint_file, device=model_config.device)

        self.score_threshold = score_threshold
    
    @abstractmethod
    def detect(self, img) -> List[Dict]:
        pass
    
    def _select_device(self, device: str):
        # detect environment, if cuda is available use it, otherwise use cpu
        if device == 'auto':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        return device

    def _filter_detections(self, labels, bboxes, scores) -> List[Dict]:
        mask = scores > self.score_threshold
        labels = labels[mask]
        bboxes = bboxes[mask]
        scores = scores[mask]
        
        detections = []
        for i in range(len(labels)):
            bbox = bboxes[i]
            sign = SignResolver.from_idx(labels[i])
            detections.append({
                'bbox': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                'abz_name': sign.abz,
                'sign_name': sign.name,
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
    
class TabletImageDetector(BaseDetector):
    def __init__(self, model_config: ModelConfig, score_threshold: float = 0.5, 
                 visualize_crop: bool = False, logging_crop: bool = False, keep_crops: bool = False):
        super().__init__(model_config, score_threshold)
        self.visualize_crop = visualize_crop
        self.logging_crop = logging_crop
        self.keep_crops = keep_crops
        self.cropped_images = []
        self.crop_coordinates = []  # Store crop coordinates for GT transformation
    
    def detect(self, img) -> List[Dict]:
        if self.keep_crops:
            self.cropped_images = [] # reset cropped for each detection

        cropped_images, crop_coordinates = divide_tablet_photo(
            img, 
            visualize=self.visualize_crop, 
            logging=self.logging_crop, 
            return_coordinates=True
        )
        
        # Store crop coordinates
        self.crop_coordinates = crop_coordinates
        
        # Create single image detector for processing cropped pieces
        single_detector = SingleImageDetector(self.model, self.score_threshold)
        
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