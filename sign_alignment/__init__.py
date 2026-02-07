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
]
