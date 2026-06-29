"""
Sign alignment package for cuneiform OCR.

This package provides tools for aligning and matching cuneiform signs
between images and text transcriptions.
"""

from .data_source import (
    EBLMongoCanonicalSource,
    LocalDataSource,
    LocalTestDataSource,
    EBLAPISource,
    SignTextParser,
    SignAPIResolver,
)

from .sign import (
    Sign,
    SignResolver,
    CLASSES_NAME,
    CLASSES_ABZ,
)

from .box import Box, Boxes, boxes_in_crop

from .detector import (
    ModelConfig,
    BaseDetector,
    SingleImageDetector,
    TabletImageDetector,
)

from .dift_model import (
    DiftConfig,
    DiftModel,
)
from .dift_align import (
    DiftAlignmentConfig,
    DiftMatchConfig,
    DiftRuntime,
)

from .visualizer import (
    BboxVisualizer,
    TextVisualizer,
    CompositeVisualizer,
    build_sign_match_info,
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

from .tablet import SubTablet, Tablet

from .psr_optimizer import (
    PointSetRegistrationOptimizer,
)

from .hyperparam import (
    hyperparameter_search,
    SEARCH_AXES,
)

__all__ = [
    # Data sources
    'EBLMongoCanonicalSource',
    'LocalDataSource',
    'LocalTestDataSource',
    'EBLAPISource',
    'SignTextParser',
    'SignAPIResolver',
    
    # Sign utilities
    'Sign',
    'SignResolver',
    'CLASSES_NAME',
    'CLASSES_ABZ',
    
    # Boxes
    'Box',
    'Boxes',
    'boxes_in_crop',
    
    # Detection
    'ModelConfig',
    'BaseDetector',
    'SingleImageDetector',
    'TabletImageDetector',
    'DiftConfig',
    'DiftModel',
    'DiftAlignmentConfig',
    'DiftMatchConfig',
    'DiftRuntime',
    
    # Visualization
    'BboxVisualizer',
    'TextVisualizer',
    'CompositeVisualizer',
    'build_sign_match_info',
    
    # Line/row processing
    'detect_rows_dbscan',
    'compute_row_similarity',
    'match_rows_dp',
    'create_row_mapping',
    'match_signs_in_row_dp',
    'align_text_row_to_detection',
    'align_text_to_detection_rows',
    
    # Tablet image frames
    'Tablet',
    'SubTablet',
    
    # PSR Optimization
    'PointSetRegistrationOptimizer',

    # Hyperparameter search
    'hyperparameter_search',
    'SEARCH_AXES',
]
