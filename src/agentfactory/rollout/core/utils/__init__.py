"""
Utility functions for rollout system

Contains helper functions for:
- Image content processing and resizing
- Message manipulation and formatting
"""

# -----------------------------------------------------------------------
# Image Processing Utilities
# -----------------------------------------------------------------------

from .image_content_helper import (
    calculate_resize_dimensions,
    smart_resize_image,
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    DEFAULT_FACTOR,
    DEFAULT_JPEG_QUALITY,
)

# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

__all__ = [
    # Image processing
    "calculate_resize_dimensions",
    "smart_resize_image",
    "DEFAULT_MAX_PIXELS",
    "DEFAULT_MIN_PIXELS", 
    "DEFAULT_FACTOR",
    "DEFAULT_JPEG_QUALITY",
] 