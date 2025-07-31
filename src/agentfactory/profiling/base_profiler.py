from abc import ABC, abstractmethod
from typing import Optional, Any, Dict
from pathlib import Path

class BaseProfiler(ABC):
    """Abstract base class for profilers supporting multiple backends."""
    
    def __init__(self, name: str):
        self.name = name
        self._is_profiling = False
        
    @abstractmethod
    def start(self) -> None:
        """Start profiling."""
        pass
        
    @abstractmethod  
    def stop(self) -> None:
        """Stop profiling."""
        pass
        
    @abstractmethod
    def get_html_output(self) -> str:
        """Get profiling results as HTML string."""
        pass
        
    @abstractmethod
    def save_html(self, file_path: Path) -> None:
        """Save profiling results as HTML file."""
        pass
        
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        
    @property
    def is_profiling(self) -> bool:
        """Check if profiler is currently active."""
        return self._is_profiling