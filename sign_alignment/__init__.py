"""
Sign alignment package for cuneiform OCR.

This package provides tools for aligning and matching cuneiform signs
between images and text transcriptions.
"""

from .data_source import (
    LocalDataSource,
    EBLAPISource,
    SignTextParser,
    create_local_source,
    create_api_source,
)

from .sign import (
    Sign,
    SignResolver,
    CLASSES_NAME,
    CLASSES_ABZ,
)

from .bounding_box import (
    BoundingBox,
    Detection,
    GroundTruths,
)

from .detector import (
    ModelConfig,
    SingleImage,
    BaseDetector,
    SingleImageDetector,
    TabletImageDetector,
)

from .visualizer import (
    BboxVisualizer,
    TextVisualizer,
    HeatmapVisualizer,
)

from .heatmap import (
    create_2d_gaussian,
    create_2d_rectangle_blur,
    create_detection_heatmap,
    create_text_heatmap,
    match_heatmaps_ncc,
    compute_avg_dimensions,
    group_detections_into_lines,
    transform_gt_to_cropped_region,
    create_text_based_detections,
)

from .tablet import (
    SignBox,
    SubTablet,
)

from .optimizer import (
    ElasticChainOptimizer,
)

__all__ = [
    # Data sources
    'LocalDataSource',
    'EBLAPISource',
    'SignTextParser',
    'create_local_source',
    'create_api_source',
    
    # Sign utilities
    'Sign',
    'SignResolver',
    'CLASSES_NAME',
    'CLASSES_ABZ',
    
    # Bounding boxes
    'BoundingBox',
    'Detection',
    'GroundTruths',
    
    # Detection
    'ModelConfig',
    'SingleImage',
    'BaseDetector',
    'SingleImageDetector',
    'TabletImageDetector',
    
    # Visualization
    'BboxVisualizer',
    'TextVisualizer',
    'HeatmapVisualizer',
    
    # Heatmap utilities
    'create_2d_gaussian',
    'create_2d_rectangle_blur',
    'create_detection_heatmap',
    'create_text_heatmap',
    'match_heatmaps_ncc',
    'compute_avg_dimensions',
    'group_detections_into_lines',
    'transform_gt_to_cropped_region',
    'create_text_based_detections',
    
    # Tablet data structures
    'SignBox',
    'SubTablet',
    
    # Optimization
    'ElasticChainOptimizer',
]
