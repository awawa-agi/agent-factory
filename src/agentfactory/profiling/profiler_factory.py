from typing import Optional, Dict, Any
from loguru import logger
from pathlib import Path
import glob
from .base_profiler import BaseProfiler

def _get_agentfactory_files():
    """Get all Python files in agentfactory package as absolute paths."""
    # Get the agentfactory package directory
    current_file = Path(__file__)
    agentfactory_dir = current_file.parent.parent  # Go up to src/agentfactory/
    
    # Find all Python files recursively
    py_files = glob.glob(str(agentfactory_dir / "**" / "*.py"), recursive=True)
    return [str(Path(f).resolve()) for f in py_files]

# Centralized profiler configurations
PROFILER_CONFIGS = {
    "pyinstrument": {
        "interval": 0.2,  # Sample every 200ms instead of default 1ms
        "async_mode": "disabled",  # Disable async mode to reduce complexity
        "use_timing_thread": True,
    },
    "viztracer": {
        "ignore_c_function": True,  # Ignore C functions to avoid CUDA issues
        "ignore_frozen": True,      # Ignore frozen modules
        "max_stack_depth": 3,       # 3 layers of stack depth
        "verbose": 0,              # Disable verbose output messages
        "tracer_entries": 500000,  # Large buffer to avoid losing early data
        "min_duration": 0.2,       # Only log functions taking >200ms for major operations only
        "log_async": False,        # Disable async task tracking to reduce data
        "log_func_args": False,    # Disable function argument logging to reduce size
        "log_func_retval": False,  # Disable return value logging to reduce size
        "include_files": _get_agentfactory_files(),  # Dynamically get all agentfactory Python files
    }
}

def create_profiler(profiler_name: str, custom_config: Optional[Dict[str, Any]] = None) -> Optional[BaseProfiler]:
    """Factory function to create a single profiler based on backend name.
    
    Args:
        profiler_name: Profiler backend name (e.g., 'pyinstrument', 'viztracer')
        custom_config: Optional custom configuration to override defaults
        
    Returns:
        BaseProfiler instance or None if invalid/unavailable
    """
    if not profiler_name:
        return None
        
    # Get default config and merge with custom config if provided
    config = PROFILER_CONFIGS.get(profiler_name, {}).copy()
    if custom_config:
        config.update(custom_config)
        
    try:
        if profiler_name == "pyinstrument":
            from .pyinstrument_profiler import PyinstrumentProfiler
            profiler = PyinstrumentProfiler(config)
            logger.debug(f"Created {profiler_name} profiler with config: {config}")
            return profiler
        elif profiler_name == "viztracer":
            from .viztracer_profiler import VizTracerProfiler
            profiler = VizTracerProfiler(config)
            logger.debug(f"Created {profiler_name} profiler with config: {config}")
            return profiler
        else:
            logger.warning(f"Unknown profiler backend: {profiler_name}")
            return None
            
    except ImportError as e:
        logger.warning(f"Failed to import {profiler_name} profiler: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to create {profiler_name} profiler: {e}")
        return None


class ProfilerManager:
    """Manager for creating and cleaning up profiler instances to avoid memory leaks.
    
    Uses a use-and-dispose pattern where each profiling session gets a fresh instance
    that is properly cleaned up after use.
    """
    
    def __init__(self, profiler_name: str, custom_config: Optional[Dict[str, Any]] = None):
        self.profiler_name = profiler_name
        self.custom_config = custom_config
        self._current_profiler: Optional[BaseProfiler] = None
        
    def start_profiling(self) -> bool:
        """Create a new profiler instance and start profiling.
        
        Returns:
            True if profiling started successfully, False otherwise
        """
        # Clean up any existing profiler first
        self._cleanup_current_profiler()
        
        # Create fresh profiler instance
        self._current_profiler = create_profiler(self.profiler_name, self.custom_config)
        
        if self._current_profiler:
            try:
                self._current_profiler.start()
                logger.trace(f"Started new {self.profiler_name} profiler instance")
                return True
            except Exception as e:
                logger.error(f"Failed to start {self.profiler_name} profiler: {e}")
                self._cleanup_current_profiler()
                return False
        else:
            logger.warning(f"Failed to create {self.profiler_name} profiler")
            return False
            
    def stop_and_get_results(self) -> Optional[str]:
        """Stop profiling and get HTML results, then cleanup the profiler instance.
        
        Returns:
            HTML profiling results or None if failed
        """
        if not self._current_profiler:
            logger.warning("No active profiler to stop")
            return None
            
        try:
            # Stop profiling
            self._current_profiler.stop()
            
            # Get results
            html_content = self._current_profiler.get_html_output()
            logger.trace(f"Got {self.profiler_name} profile results ({len(html_content)} chars)")
            
            # Clean up immediately to free memory
            self._cleanup_current_profiler()
            
            return html_content
            
        except Exception as e:
            logger.error(f"Failed to get {self.profiler_name} profile results: {e}")
            self._cleanup_current_profiler()
            return None
            
    def _cleanup_current_profiler(self):
        """Clean up the current profiler instance to free memory."""
        if self._current_profiler:
            try:
                # Try to stop if still running
                if hasattr(self._current_profiler, '_is_profiling') and self._current_profiler._is_profiling:
                    self._current_profiler.stop()
                
                # Clear the reference to allow garbage collection
                self._current_profiler = None
                logger.trace(f"Cleaned up {self.profiler_name} profiler instance")
                
            except Exception as e:
                logger.warning(f"Error during {self.profiler_name} profiler cleanup: {e}")
                self._current_profiler = None
                
    def get_profiler_name(self) -> str:
        """Get the profiler backend name."""
        return self.profiler_name
        
    def __del__(self):
        """Ensure cleanup on deletion."""
        self._cleanup_current_profiler()