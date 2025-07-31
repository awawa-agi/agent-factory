from typing import Optional
from pathlib import Path
from loguru import logger
from .base_profiler import BaseProfiler

class VizTracerProfiler(BaseProfiler):
    """VizTracer backend implementation for profiling."""
    
    def __init__(self, config: dict):
        super().__init__("viztracer")
        self._tracer = None
        self._config = config
        
    def _ensure_tracer(self):
        """Lazily import and initialize viztracer."""
        if self._tracer is None:
            try:
                from viztracer import VizTracer
                # Use config from factory
                self._tracer = VizTracer(**self._config)
                logger.debug(f"VizTracer profiler initialized with config: {self._config}")
            except ImportError as e:
                logger.error(f"Failed to import viztracer: {e}")
                raise ImportError(
                    "viztracer is required but not installed. "
                    "Please install it with: pip install viztracer"
                ) from e
                
    def start(self) -> None:
        """Start viztracer profiling."""
        self._ensure_tracer()
        if not self._is_profiling:
            self._tracer.start()
            self._is_profiling = True
            logger.trace("VizTracer profiling started")
            
    def stop(self) -> None:
        """Stop viztracer profiling."""
        if self._is_profiling and self._tracer:
            self._tracer.stop()
            self._is_profiling = False
            logger.trace("VizTracer profiling stopped")
            
    def get_html_output(self) -> str:
        """Get profiling results as HTML string."""
        if not self._tracer:
            raise RuntimeError("Tracer not initialized or never started")
        
        try:
            # VizTracer doesn't have a direct method to get HTML as string
            # We need to save to a temporary file and read it back
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Save to temporary HTML file
                self._tracer.save(output_file=temp_path)
                
                # Read the HTML content
                with open(temp_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                logger.trace(f"Generated HTML profile output ({len(html_content)} chars)")
                return html_content
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            logger.error(f"Failed to generate HTML output: {e}")
            raise
            
    def save_html(self, file_path: Path) -> None:
        """Save profiling results as HTML file."""
        if not self._tracer:
            raise RuntimeError("Tracer not initialized or never started")
            
        try:
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # VizTracer can save directly to HTML
            self._tracer.save(output_file=str(file_path))
            logger.debug(f"Saved viztracer profile to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save HTML profile to {file_path}: {e}")
            raise