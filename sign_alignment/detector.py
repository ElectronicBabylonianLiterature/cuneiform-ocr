
from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
from typing import List, Optional

from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules

import torch
import numpy as np

from data_processing.divide_photos import divide_tablet_photo

from .sign import SignResolver
from .box import Box, Boxes, SignCandidate
from .tablet import SubTablet, Tablet

@dataclass
class ModelConfig:
    config_file: str
    checkpoint_file: str
    device: str = 'cuda:0'


class BaseDetector(ABC):
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        default_score_threshold: float = 0.5,
        is_load_now: bool = True,
        model=None,
    ):
        self.default_score_threshold = default_score_threshold
        self.model_config = model_config
        self.model = model
        self.result = {}
        if self.model is None and is_load_now:
            self.load_model()
    
    @abstractmethod
    def detect(self, tablet: Tablet, score_threshold: Optional[float] = None) -> Boxes:
        pass
    
    def _select_device(self, device: str):
        if device == 'auto':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        return device
    
    def load_model(self) -> None:
        print("Loading model...")
        register_all_modules()
        device = self._select_device(self.model_config.device)
        print(f"Using device: {device}")
        self.model = init_detector(
            self.model_config.config_file,
            self.model_config.checkpoint_file,
            device=device,
        )

    def unload_model(self) -> None:
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _filter_detections(self, labels, bboxes, scores, tablet: Tablet, score_threshold: float, deduplicate: bool = True) -> Boxes:
        mask = scores > score_threshold
        labels = labels[mask]
        bboxes = bboxes[mask]
        scores = scores[mask]

        groups = [[i] for i in range(len(labels))]
        if deduplicate:
            score_order = np.argsort(-scores, kind="stable")
            groups = []
            for idx in score_order:
                for group in groups:
                    if np.all(np.abs(bboxes[idx] - bboxes[group[0]]) <= 2.0):
                        group.append(int(idx))
                        break
                else:
                    groups.append([int(idx)])
        
        detections = Boxes(tablet=tablet)
        for group in groups:
            bbox = bboxes[group[0]]
            candidates = [
                SignCandidate(
                    sign=SignResolver.from_idx(int(labels[i])),
                    score=float(scores[i]),
                )
                for i in group
            ]
            detections.append(Box(
                x1=float(bbox[0]),
                y1=float(bbox[1]),
                x2=float(bbox[2]),
                y2=float(bbox[3]),
                candidates=candidates,
                tablet=tablet,
            ))
        
        return detections

class SingleImageDetector(BaseDetector):
    def detect(self, tablet: Tablet, score_threshold: Optional[float] = None) -> Boxes:
        if score_threshold is None:
            score_threshold = self.default_score_threshold
        img_hash = hashlib.sha256(tablet.img.tobytes()).digest()
        img_key = (tablet.img.shape, tablet.img.dtype.str, img_hash)
        if img_key not in self.result:
            self.result[img_key] = inference_detector(self.model, tablet.img)
        OCR_result = self.result[img_key].pred_instances.cpu()
        
        labels = OCR_result['labels'].numpy()
        bboxes = OCR_result['bboxes'].numpy()
        scores = OCR_result['scores'].numpy()
        
        return self._filter_detections(labels, bboxes, scores, tablet, score_threshold, deduplicate=True)
    
class TabletImageDetector(BaseDetector):
    def __init__(
        self,
        model_config: ModelConfig,
        default_score_threshold: float = 0.5,
        visualize_crop: bool = False,
        logging_crop: bool = False,
        keep_crops: bool = False,
        is_crop_itself: bool = False,
        is_load_now: bool = True,
    ):
        super().__init__(model_config, default_score_threshold, is_load_now=is_load_now)
        self.visualize_crop = visualize_crop
        self.logging_crop = logging_crop
        self.keep_crops = keep_crops
        self.is_crop_itself = is_crop_itself
        self.crop_tablets = []
        self.crop_boxes = []
        self.crop_coordinates = []
    
    def detect(self, tablet: Tablet, score_threshold: Optional[float] = None) -> Boxes:
        if score_threshold is None:
            score_threshold = self.default_score_threshold
        if self.model is None:
            self.load_model()

        if self.keep_crops:
            self.crop_tablets = []
            self.crop_boxes = []

        if self.is_crop_itself:
            h, w = tablet.shape
            self.crop_coordinates = [{'x': 0, 'y': 0, 'w': w, 'h': h}]
            single_detector = SingleImageDetector(model=self.model, default_score_threshold=score_threshold)
            detections = single_detector.detect(tablet)
            if self.keep_crops:
                crop_tablet = SubTablet(
                    img=tablet.img,
                    parent=tablet,
                    offset_in_parent=(0.0, 0.0),
                    mask=np.full((h, w), 255, dtype=np.uint8),
                    name="crop_0",
                )
                crop_detections = detections.to_tablet(crop_tablet)
                self.crop_tablets.append(crop_tablet)
                self.crop_boxes.append(crop_detections)
            return detections

        cropped_images, crop_coordinates, masks = divide_tablet_photo(
            tablet.img,
            visualize=self.visualize_crop, 
            logging=self.logging_crop, 
            return_coordinates=True,
            return_masks=True,
        )
        
        self.crop_coordinates = crop_coordinates
        
        single_detector = SingleImageDetector(model=self.model, default_score_threshold=score_threshold)
        
        all_detections = Boxes(tablet=tablet)
        
        for idx, img_piece in enumerate(cropped_images):
            piece_offset_x = crop_coordinates[idx]['x']
            piece_offset_y = crop_coordinates[idx]['y']
            crop_tablet = SubTablet(
                img=img_piece,
                parent=tablet,
                offset_in_parent=(piece_offset_x, piece_offset_y),
                name=f"crop_{idx}",
                mask=masks[idx],
            )
            piece_detections = single_detector.detect(crop_tablet)

            if self.keep_crops:
                self.crop_tablets.append(crop_tablet)
                self.crop_boxes.append(piece_detections)
            
            for det in piece_detections:
                all_detections.append(det.to_tablet(tablet))
        
        return all_detections
    
    def get_crop_tablets(self) -> List[SubTablet]:
        return self.crop_tablets

    def get_crop_boxes(self) -> List[Boxes]:
        return self.crop_boxes
