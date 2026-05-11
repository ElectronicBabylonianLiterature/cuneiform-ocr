"""
Sign alignment package for cuneiform OCR.

This package provides tools for aligning and matching cuneiform signs
between images and text transcriptions.
"""

from .data_source import (
    LocalDataSource,
    LocalTestDataSource,
    EBLAPISource,
    SubtabletEBLAPISource,
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
    CompositeVisualizer,
    build_sign_match_info,
)

from .heatmap import (
    compute_avg_dimensions,
    transform_gt_to_cropped_region,
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

from .psr_optimizer import (
    PointSetRegistrationOptimizer,
    initialize_text_subtablet,
    filter_boxes_by_mask,
)

from .hyperparam import (
    hyperparameter_search,
    SEARCH_AXES,
)

__all__ = [
    # Data sources
    'LocalDataSource',
    'LocalTestDataSource',
    'EBLAPISource',
    'SubtabletEBLAPISource',
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
    'CompositeVisualizer',
    'build_sign_match_info',
    
    # Geometry utilities
    'compute_avg_dimensions',
    'transform_gt_to_cropped_region',
    
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
    
    # PSR Optimization
    'PointSetRegistrationOptimizer',
    'initialize_text_subtablet',
    'filter_boxes_by_mask',

    # Hyperparameter search
    'hyperparameter_search',
    'SEARCH_AXES',
]
