from typing import Optional
from pathlib import Path
from loguru import logger
from .base_profiler import BaseProfiler

class PyinstrumentProfiler(BaseProfiler):
    """Pyinstrument backend implementation for profiling."""
    
    def __init__(self, config: dict):
        super().__init__("pyinstrument")
        self._profiler = None
        self._config = config
        
    def _ensure_profiler(self):
        """Lazily import and initialize pyinstrument profiler."""
        if self._profiler is None:
            try:
                from pyinstrument import Profiler
                # Use config from factory
                self._profiler = Profiler(**self._config)
                logger.debug(f"Pyinstrument profiler initialized with config: {self._config}")
            except ImportError as e:
                logger.error(f"Failed to import pyinstrument: {e}")
                raise ImportError(
                    "pyinstrument is required but not installed. "
                    "Please install it with: pip install pyinstrument"
                ) from e
                
    def start(self) -> None:
        """Start pyinstrument profiling. Resets any previous profiling data."""
        self._ensure_profiler()
        if not self._is_profiling:
            # Reset to clear any previous profiling data
            self._profiler.reset()
            self._profiler.start()
            self._is_profiling = True
            logger.trace("Pyinstrument profiling started (previous data cleared)")
            
    def stop(self) -> None:
        """Stop pyinstrument profiling."""
        if self._is_profiling and self._profiler:
            self._profiler.stop()
            self._is_profiling = False
            logger.trace("Pyinstrument profiling stopped")
            
    def get_html_output(self) -> str:
        """Get profiling results as HTML string."""
        if not self._profiler:
            raise RuntimeError("Profiler not initialized or never started")
        
        try:
            html_output = self._profiler.output_html()
            logger.trace(f"Generated HTML profile output ({len(html_output)} chars)")
            return html_output
        except Exception as e:
            logger.error(f"Failed to generate HTML output: {e}")
            raise
            
    def save_html(self, file_path: Path) -> None:
        """Save profiling results as HTML file."""
        if not self._profiler:
            raise RuntimeError("Profiler not initialized or never started")
            
        try:
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            html_content = self.get_html_output()
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.debug(f"Saved pyinstrument profile to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save HTML profile to {file_path}: {e}")
            raise