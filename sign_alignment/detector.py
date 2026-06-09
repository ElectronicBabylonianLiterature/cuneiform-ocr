
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules

import torch
import numpy as np

from data_processing.divide_photos import divide_tablet_photo

from .sign import SignResolver
from .box import Box, Boxes
from .tablet import SubTablet

@dataclass
class ModelConfig:
    config_file: str
    checkpoint_file: str
    device: str = 'cuda:0'


class BaseDetector(ABC):
    def __init__(self, model_config: Optional[ModelConfig] = None, score_threshold: float = 0.5, model = None):
        self.score_threshold = score_threshold
        
        if model is not None:
            self.model = model
        elif model_config is not None:
            print("Initializing detector, loading model...")
            register_all_modules()
            model_config.device = self._select_device(model_config.device)
            print(f"Using device: {model_config.device}")
            self.model = init_detector(model_config.config_file, model_config.checkpoint_file, device=model_config.device)
        else:
            raise ValueError("Either model_config or model must be provided")
    
    @abstractmethod
    def detect(self, img) -> Boxes:
        pass
    
    def _select_device(self, device: str):
        if device == 'auto':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        return device

    def unload_model(self) -> None:
        """Release the detector model from GPU memory."""
        import gc
        if self.model is None:
            return
        self.model.cpu()
        del self.model
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Detector model unloaded from GPU.")

    def _filter_detections(self, labels, bboxes, scores) -> Boxes:
        mask = scores > self.score_threshold
        labels = labels[mask]
        bboxes = bboxes[mask]
        scores = scores[mask]
        
        detections = Boxes()
        for i in range(len(labels)):
            bbox = bboxes[i]
            sign = SignResolver.from_idx(labels[i])
            detections.append(Box(
                x1=float(bbox[0]),
                y1=float(bbox[1]),
                x2=float(bbox[2]),
                y2=float(bbox[3]),
                score=float(scores[i]),
                sign=sign
            ))
        
        return detections

class SingleImageDetector(BaseDetector):
    def detect(self, img) -> Boxes:
        result = inference_detector(self.model, img)
        OCR_result = result.pred_instances.cpu()
        
        labels = OCR_result['labels'].numpy()
        bboxes = OCR_result['bboxes'].numpy()
        scores = OCR_result['scores'].numpy()
        
        return self._filter_detections(labels, bboxes, scores)
    
class TabletImageDetector(BaseDetector):
    def __init__(self, model_config: ModelConfig, score_threshold: float = 0.5, 
                 visualize_crop: bool = False, logging_crop: bool = False, keep_crops: bool = False, is_crop_itself: bool = False):
        super().__init__(model_config, score_threshold)
        self.visualize_crop = visualize_crop
        self.logging_crop = logging_crop
        self.keep_crops = keep_crops
        self.is_crop_itself = is_crop_itself
        self.cropped_images = []
        self.crop_boxes = []
        self.crop_coordinates = []
        
    
    def detect(self, img) -> Boxes:
        if self.keep_crops:
            self.cropped_images = []
            self.crop_boxes = []

        if self.is_crop_itself:
            h, w = img.shape[:2]
            self.crop_coordinates = [{'x': 0, 'y': 0, 'w': w, 'h': h}]
            single_detector = SingleImageDetector(model=self.model, score_threshold=self.score_threshold)
            detections = single_detector.detect(img)
            if self.keep_crops:
                mask = np.full((h, w), 255, dtype=np.uint8)
                subtablet = SubTablet(img=img, mask=mask, name="crop_0")
                self.cropped_images.append(subtablet)
                self.crop_boxes.append(detections.copy(subtablet=subtablet))
            return detections

        cropped_images, crop_coordinates, masks = divide_tablet_photo(
            img, 
            visualize=self.visualize_crop, 
            logging=self.logging_crop, 
            return_coordinates=True,
            return_masks=True,
        )
        
        self.crop_coordinates = crop_coordinates
        
        single_detector = SingleImageDetector(model=self.model, score_threshold=self.score_threshold)
        
        all_detections = Boxes()
        
        for idx, img_piece in enumerate(cropped_images):
            piece_detections = single_detector.detect(img_piece)

            if self.keep_crops:
                subtablet = SubTablet(img=img_piece, mask=masks[idx], name=f"crop_{idx}")
                self.cropped_images.append(subtablet)
                self.crop_boxes.append(piece_detections.copy(subtablet=subtablet))
            
            piece_offset_x = crop_coordinates[idx]['x']
            piece_offset_y = crop_coordinates[idx]['y']
            
            for det in piece_detections:
                transformed_det = Box(
                    x1=det.x1 + piece_offset_x,
                    y1=det.y1 + piece_offset_y,
                    x2=det.x2 + piece_offset_x,
                    y2=det.y2 + piece_offset_y,
                    score=det.score,
                    sign=det.sign
                )
                all_detections.append(transformed_det)
        
        return all_detections
    
    def get_cropped_images(self) -> List[SubTablet]:
        return self.cropped_images

    def get_crop_boxes(self) -> List[Boxes]:
        return self.crop_boxes
