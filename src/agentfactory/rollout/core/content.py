from __future__ import annotations

import io
from pathlib import Path
from typing import (
    Any, Dict, Optional, Tuple, Union, Literal,
    TYPE_CHECKING
)

from PIL import Image
from pydantic import BaseModel, Field, PrivateAttr

if TYPE_CHECKING:
    from .utils.image_content_helper import smart_resize_image, encode_image_to_data_url, fetch_image_bytes

class ImageDefaults:
    """Default configuration for image processing."""
    QUALITY = 85
    TIMEOUT = 30
    MAX_PIXELS = 28 * 28 * 2048
    MIN_PIXELS = 56 * 56
    RESIZE_FACTOR = 28
    THUMBNAIL_SIZE = 256
    ASYNC_CONCURRENCY = 8
    MAX_ASPECT_RATIO = 100

# ---------------------------------------------------------------------------
# Content models
# ---------------------------------------------------------------------------

class TextContent(BaseModel):
    """Plain text content within a message."""
    type: Literal["text"] = Field(default="text", frozen=True)
    text: str

    def __str__(self) -> str:
        return self.text


class ImageUrl(BaseModel):
    """Container for image URL with validation."""
    url: str

    def is_data_url(self) -> bool:
        return self.url.startswith("data:")
    
    def is_file_url(self) -> bool:
        return self.url.startswith("file://")
    
    def is_http_url(self) -> bool:
        return self.url.startswith(("http://", "https://"))


class ImageContent(BaseModel):
    """Image content with lazy loading and caching capabilities."""
    
    model_config = {"frozen": True}
    
    type: Literal["image_url"] = Field(default="image_url", frozen=True)
    image_url: ImageUrl
    
    # Private cached properties
    _raw_bytes: Optional[bytes] = PrivateAttr(default=None)
    _dimensions: Optional[Tuple[int, int]] = PrivateAttr(default=None)
    _is_loaded: bool = PrivateAttr(default=False)

    @property
    def url(self) -> str:
        """Convenience accessor for the image URL."""
        return self.image_url.url

    @property
    def width(self) -> Optional[int]:
        """Image width if known."""
        return self._dimensions[0] if self._dimensions else None

    @property
    def height(self) -> Optional[int]:
        """Image height if known.""" 
        return self._dimensions[1] if self._dimensions else None

    @property
    def size(self) -> Optional[Tuple[int, int]]:
        """Image dimensions as (width, height) tuple."""
        return self._dimensions

    @property
    def is_loaded(self) -> bool:
        """Whether image data has been fetched and cached."""
        return self._is_loaded

    # ---------------------------------------------------------------------------
    # Synchronous image operations
    # ---------------------------------------------------------------------------

    def fetch_bytes(self) -> bytes:
        """Download or load image bytes with caching."""
        if self._raw_bytes is not None:
            return self._raw_bytes

        from .utils.image_content_helper import fetch_image_bytes
        
        self._raw_bytes = fetch_image_bytes(self.url, timeout=ImageDefaults.TIMEOUT)
        self._is_loaded = True
        return self._raw_bytes

    def to_pil(self) -> Image.Image:
        """Convert to PIL Image object (creates new instance each time)."""
        image_bytes = self.fetch_bytes()
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Cache dimensions on first load
        if self._dimensions is None:
            self._dimensions = pil_image.size
            
        return pil_image

    def save_to_file(self, path: Union[str, Path], **kwargs) -> Path:
        """Save image to file with optional format conversion."""
        output_path = Path(path)
        self.to_pil().save(output_path, **kwargs)
        return output_path

    # ---------------------------------------------------------------------------
    # Image transformation methods
    # ---------------------------------------------------------------------------

    def resize_smart(
        self,
        max_pixels: int = ImageDefaults.MAX_PIXELS,
        min_pixels: int = ImageDefaults.MIN_PIXELS,
        factor: int = ImageDefaults.RESIZE_FACTOR,
        max_aspect_ratio: int = ImageDefaults.MAX_ASPECT_RATIO,
    ) -> "ImageContent":
        """Create resized copy using smart resizing algorithm."""
        from .utils.image_content_helper import smart_resize_image, encode_image_to_data_url
        
        original_image = self.to_pil().convert("RGB")
        resized_image = smart_resize_image(
            image=original_image,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            factor=factor,
            max_aspect_ratio=max_aspect_ratio,
        )
        
        # Create new instance with resized image as data URL
        data_url = encode_image_to_data_url(resized_image, quality=ImageDefaults.QUALITY)
        return self.model_copy(
            update={"image_url": ImageUrl(url=data_url)},
            deep=True
        )

    def create_thumbnail(self, size: int = ImageDefaults.THUMBNAIL_SIZE) -> "ImageContent":
        """Create square thumbnail maintaining aspect ratio."""
        from .utils.image_content_helper import encode_image_to_data_url
        
        image = self.to_pil().convert("RGB")
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        
        data_url = encode_image_to_data_url(image, quality=ImageDefaults.QUALITY)
        return self.model_copy(
            update={"image_url": ImageUrl(url=data_url)},
            deep=True
        )

    def data_url_format(self, quality: int = ImageDefaults.QUALITY) -> "ImageContent":
        """Convert image to data URL format if not already."""
        if self.image_url.is_data_url():
            return self
        
        from .utils.image_content_helper import encode_image_to_data_url
        
        pil_image = self.to_pil().convert("RGB")
        data_url = encode_image_to_data_url(pil_image, quality=quality)
        
        return self.model_copy(
            update={"image_url": ImageUrl(url=data_url)},
            deep=True
        )

class McpPromptContent(BaseModel):
    """MCP prompt content."""
    type: Literal["mcp_prompt"] = Field(default="mcp_prompt", frozen=True)
    prompt_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)