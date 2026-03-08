"""
Sign alignment package for cuneiform OCR.

This package provides tools for aligning and matching cuneiform signs
between images and text transcriptions.
"""

from .data_source import (
    LocalDataSource,
    EBLAPISource,
    SignTextParser,
    SignAPIResolver,
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

from data_processing.line_process import (
    detect_rows_dbscan,
    compute_row_similarity,
    match_rows_dp,
    create_row_mapping,
    match_signs_in_row_dp,
    align_text_row_to_detection,
    align_text_to_detection_rows,
)

from .tablet import (
    SignBox,
    SubTablet,
)

from .optimizer import (
    ElasticChainOptimizer,
    build_agnostic_heatmap,
)

from .psr_optimizer import (
    PointSetRegistrationOptimizer,
    initialize_text_subtablet,
    filter_boxes_by_mask,
)

__all__ = [
    # Data sources
    'LocalDataSource',
    'EBLAPISource',
    'SignTextParser',
    'SignAPIResolver',
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
    
    # Line/row processing
    'detect_rows_dbscan',
    'compute_row_similarity',
    'match_rows_dp',
    'create_row_mapping',
    'match_signs_in_row_dp',
    'align_text_row_to_detection',
    'align_text_to_detection_rows',
    
    # Tablet data structures
    'SignBox',
    'SubTablet',
    
    # Optimization
    'ElasticChainOptimizer',
    'build_agnostic_heatmap',
    
    # PSR Optimization
    'PointSetRegistrationOptimizer',
    'initialize_text_subtablet',
    'filter_boxes_by_mask',
]
