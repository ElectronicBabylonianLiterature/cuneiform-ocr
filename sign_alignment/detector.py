
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
import copy

from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules

import torch
import numpy as np

from data_processing.divide_photos import divide_tablet_photo

from .sign import SignResolver
from .bounding_box import BoundingBox, Detection

@dataclass
class ModelConfig():
    config_file: str
    checkpoint_file: str
    device: str = 'cuda:0'
    
@dataclass
class SingleImage:
    img: np.ndarray
    detections: Detection = field(default_factory=list)
    
    def __len__(self):
        return len(self.detections)

class BaseDetector(ABC):
    def __init__(self, model_config: Optional[ModelConfig] = None, score_threshold: float = 0.5, model = None):
        self.score_threshold = score_threshold
        
        if model is not None:
            # Use provided model directly
            self.model = model
        elif model_config is not None:
            # Load model from config
            print("Initializing detector, loading model...")
            register_all_modules()
            model_config.device = self._select_device(model_config.device)
            print(f"Using device: {model_config.device}")
            self.model = init_detector(model_config.config_file, model_config.checkpoint_file, device=model_config.device)
        else:
            raise ValueError("Either model_config or model must be provided")
    
    @abstractmethod
    def detect(self, img) -> Detection:
        pass
    
    def _select_device(self, device: str):
        # detect environment, if cuda is available use it, otherwise use cpu
        if device == 'auto':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        return device

    def _filter_detections(self, labels, bboxes, scores) -> Detection:
        mask = scores > self.score_threshold
        labels = labels[mask]
        bboxes = bboxes[mask]
        scores = scores[mask]
        
        detections = []
        for i in range(len(labels)):
            bbox = bboxes[i]
            sign = SignResolver.from_idx(labels[i])
            detections.append(BoundingBox(
                x1=float(bbox[0]),
                y1=float(bbox[1]),
                x2=float(bbox[2]),
                y2=float(bbox[3]),
                score=float(scores[i]),
                sign=sign
            ))
        
        return detections

class SingleImageDetector(BaseDetector):
    def detect(self, img) -> Detection:
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
    
    def detect(self, img) -> Detection:
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
        
        # Create single image detector using the already-loaded model
        single_detector = SingleImageDetector(model=self.model, score_threshold=self.score_threshold)
        
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
                # Transform BoundingBox coordinates
                transformed_det = BoundingBox(
                    x1=det.x1 + piece_offset_x,
                    y1=det.y1 + piece_offset_y,
                    x2=det.x2 + piece_offset_x,
                    y2=det.y2 + piece_offset_y,
                    score=det.score,
                    sign=det.sign
                )
                all_detections.append(transformed_det)
        
        return all_detections
    
    def get_cropped_images(self) -> List[SingleImage]:
        return self.cropped_images