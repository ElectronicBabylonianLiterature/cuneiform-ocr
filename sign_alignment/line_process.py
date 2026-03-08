"""
Compatibility stub: line_process has been moved to data_processing/line_process.py.
All symbols are re-exported from there for backward compatibility.
"""

from data_processing.line_process import (
    detect_rows_dbscan,
    compute_row_similarity,
    match_rows_dp,
    create_row_mapping,
    match_signs_in_row_dp,
    align_text_row_to_detection,
    align_text_to_detection_rows,
)

__all__ = [
    'detect_rows_dbscan',
    'compute_row_similarity',
    'match_rows_dp',
    'create_row_mapping',
    'match_signs_in_row_dp',
    'align_text_row_to_detection',
    'align_text_to_detection_rows',
]
