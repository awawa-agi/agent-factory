"""
Image processing module for handling images in rollout system

This module provides core image processing capabilities including:
- RolloutImage class for unified image representation
- Image resizing and format standardization  
- Async image loading from various sources
"""

import io
import base64
import math
from pathlib import Path
from typing import Tuple
from PIL import Image
import httpx

# -------------------------------------------------------------------
# Configuration and Constants
# -------------------------------------------------------------------

DEFAULT_MAX_PIXELS = 28 * 28 * 2048
DEFAULT_MIN_PIXELS = 56 * 56
DEFAULT_FACTOR = 28
DEFAULT_JPEG_QUALITY = 80
DEFAULT_MAX_ASPECT_RATIO = 100
DEFAULT_TIMEOUT = 30


# -------------------------------------------------------------------
# Image Loading Functions
# -------------------------------------------------------------------

def fetch_image_bytes(url: str, timeout: int = DEFAULT_TIMEOUT) -> bytes:
    """
    Fetch image bytes from various URL sources.
    
    Supports:
    - Data URLs (base64 encoded)
    - File URLs (local filesystem)
    - HTTP/HTTPS URLs (remote download)
    
    Args:
        url: The image URL to fetch from
        timeout: HTTP request timeout in seconds
        
    Returns:
        Raw image bytes
        
    Raises:
        ValueError: If URL scheme is unsupported or fetch fails
        FileNotFoundError: If local file doesn't exist
    """
    try:
        if url.startswith("data:"):
            # Handle data URLs
            _, data = url.split(",", 1)
            return base64.b64decode(data)
        elif url.startswith("file://"):
            # Handle file URLs
            file_path = Path(url[7:])  # Remove 'file://' prefix
            if not file_path.exists():
                raise FileNotFoundError(f"Image file not found: {file_path}")
            return file_path.read_bytes()
        elif url.startswith(("http://", "https://")):
            # Handle HTTP URLs
            with httpx.Client(timeout=timeout) as client:
                response = client.get(url)
                response.raise_for_status()
                return response.content
        else:
            raise ValueError(f"Unsupported URL scheme: {url}")
    except Exception as e:
        raise ValueError(f"Failed to fetch image from {url}: {e}") from e


# -------------------------------------------------------------------
# Dimension Calculation Utilities
# -------------------------------------------------------------------

def calculate_resize_dimensions(
    width: int,
    height: int, 
    factor: int = DEFAULT_FACTOR, 
    min_pixels: int = DEFAULT_MIN_PIXELS, 
    max_pixels: int = DEFAULT_MAX_PIXELS,
    max_aspect_ratio: int = DEFAULT_MAX_ASPECT_RATIO,
) -> Tuple[int, int]:
    """
    Calculate resized image dimensions with constraints
    
    Based on Qwen2.5-VL approach, ensures:
    1. Height and width divisible by factor
    2. Total pixels within [min_pixels, max_pixels] range
    3. Preserves aspect ratio as much as possible
    
    Args:
        width: Original width
        height: Original height
        factor: Alignment factor for dimensions
        min_pixels: Minimum allowed pixels
        max_pixels: Maximum allowed pixels
        
    Returns:
        Tuple of (new_width, new_height)
        
    Raises:
        ValueError: If dimensions are too small or aspect ratio too extreme
    """
    # Validate input dimensions
    if height < factor or width < factor:
        raise ValueError(
            f"Height ({height}) and width ({width}) must be larger than "
            f"factor ({factor})"
        )
    
    # Check aspect ratio constraints
    aspect_ratio = max(height, width) / min(height, width)
    if aspect_ratio > max_aspect_ratio:
        raise ValueError(
            f"Aspect ratio too extreme: {aspect_ratio:.1f} > {max_aspect_ratio}"
        )
    
    # Calculate initial dimensions aligned to factor
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    
    # Adjust for maximum pixel constraint
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    
    # Adjust for minimum pixel constraint
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return w_bar, h_bar


def encode_image_to_data_url(image: Image.Image, quality: int = DEFAULT_JPEG_QUALITY) -> str:
    """Encode image to data url format"""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode()}"


def smart_resize_image(
    image: Image.Image,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    factor: int = DEFAULT_FACTOR,
    max_aspect_ratio: int = DEFAULT_MAX_ASPECT_RATIO,
) -> Image.Image:
    """
    Smart resize image with constraints
    
    Automatically calculates optimal dimensions and resizes if needed.
    Returns original image if no resize is required.
    
    Args:
        jpeg_bytes: Original image data
        current_width: Current image width
        current_height: Current image height
        max_pixels: Maximum allowed pixels
        min_pixels: Minimum allowed pixels
        factor: Alignment factor for dimensions
        
    Returns:
        Tuple of (jpeg_bytes, width, height)
    """
    # Calculate optimal dimensions
    new_width, new_height = calculate_resize_dimensions(
        image.width, 
        image.height, 
        factor, 
        min_pixels, 
        max_pixels,
        max_aspect_ratio,
    )
    
    # Return original if no resize needed
    if (new_width, new_height) == (image.width, image.height):
        return image
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)