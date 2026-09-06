
from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
from typing import Optional

from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules

import cv2
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
        use_sahi: bool = False,
        sahi_model=None,
        box_slice_ratio: float = 0.15,
    ):
        self.default_score_threshold = default_score_threshold
        self.model_config = model_config
        self.model = model
        self.use_sahi = use_sahi
        self.sahi_model = sahi_model
        self.box_slice_ratio = box_slice_ratio
        self.slice_height = None
        self.slice_width = None
        self.result = {}
        model_is_missing = self.sahi_model is None if self.use_sahi else self.model is None
        if model_is_missing and is_load_now:
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
        if self.use_sahi:
            from sahi import AutoDetectionModel

            self.sahi_model = AutoDetectionModel.from_pretrained(
                model_type="mmdet",
                model_path=self.model_config.checkpoint_file,
                config_path=self.model_config.config_file,
                confidence_threshold=0.1,
                device=device,
            )
        else:
            self.model = init_detector(
                self.model_config.config_file,
                self.model_config.checkpoint_file,
                device=device,
            )

    def unload_model(self) -> None:
        self.model = None
        self.sahi_model = None
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

        if self.use_sahi:
            from sahi.predict import get_prediction, get_sliced_prediction

            def unpack_result(result):
                predictions = result.object_prediction_list
                labels = np.asarray(
                    [prediction.category.id for prediction in predictions], dtype=np.int64
                )
                bboxes = np.asarray(
                    [prediction.bbox.to_xyxy() for prediction in predictions], dtype=np.float32
                ).reshape(-1, 4)
                scores = np.asarray(
                    [prediction.score.value for prediction in predictions], dtype=np.float32
                )
                return labels, bboxes, scores

            img_key = (*img_key, float(score_threshold), self.box_slice_ratio)
            if img_key not in self.result:
                img_rgb = cv2.cvtColor(tablet.img, cv2.COLOR_BGR2RGB)
                initial_result = get_prediction(
                    img_rgb,
                    self.sahi_model,
                    confidence_threshold=0.0,
                )
                initial_labels, initial_bboxes, initial_scores = unpack_result(initial_result)
                initial_detections = self._filter_detections(
                    initial_labels,
                    initial_bboxes,
                    initial_scores,
                    tablet,
                    score_threshold,
                    deduplicate=True,
                )

                slice_size = max(1, int(initial_detections.avg_size / self.box_slice_ratio))
                self.slice_height = slice_size
                self.slice_width = slice_size
                sliced_result = get_sliced_prediction(
                    img_rgb,
                    self.sahi_model,
                    confidence_threshold=0.2,
                    slice_height=self.slice_height,
                    slice_width=self.slice_width,
                    overlap_height_ratio=0.2,
                    overlap_width_ratio=0.2,
                )
                self.result[img_key] = (sliced_result, self.slice_height, self.slice_width)

            result, self.slice_height, self.slice_width = self.result[img_key]
            labels, bboxes, scores = unpack_result(result)

            return self._filter_detections(
                labels, bboxes, scores, tablet, score_threshold, deduplicate=True
            )

        # non-SAHI detection
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
        is_crop_itself: bool = False,
        is_load_now: bool = True,
        use_sahi: bool = False,
        box_slice_ratio: float = 0.15,
    ):
        super().__init__(
            model_config,
            default_score_threshold,
            is_load_now=is_load_now,
            use_sahi=use_sahi,
            box_slice_ratio=box_slice_ratio,
        )
        self.visualize_crop = visualize_crop
        self.logging_crop = logging_crop
        self.is_crop_itself = is_crop_itself
        self.crop_tablets = []
        self.crop_boxes = []
        self.crop_coordinates = []
    
    def detect(self, tablet: Tablet, score_threshold: Optional[float] = None) -> Boxes:
        if score_threshold is None:
            score_threshold = self.default_score_threshold
        model_is_missing = self.sahi_model is None if self.use_sahi else self.model is None
        if model_is_missing:
            self.load_model()

        self.crop_tablets = []
        self.crop_boxes = []

        if self.is_crop_itself:
            h, w = tablet.shape
            self.crop_coordinates = [{'x': 0, 'y': 0, 'w': w, 'h': h}]
            single_detector = SingleImageDetector(
                model=self.model,
                sahi_model=self.sahi_model,
                use_sahi=self.use_sahi,
                box_slice_ratio=self.box_slice_ratio,
                default_score_threshold=score_threshold,
            )
            detections = single_detector.detect(tablet)
            self.slice_height = single_detector.slice_height
            self.slice_width = single_detector.slice_width
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
        
        single_detector = SingleImageDetector(
            model=self.model,
            sahi_model=self.sahi_model,
            use_sahi=self.use_sahi,
            box_slice_ratio=self.box_slice_ratio,
            default_score_threshold=score_threshold,
        )
        
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
            self.slice_height = single_detector.slice_height
            self.slice_width = single_detector.slice_width

            self.crop_tablets.append(crop_tablet)
            self.crop_boxes.append(piece_detections)
            
            for det in piece_detections:
                all_detections.append(det.to_tablet(tablet))
        
        return all_detections
