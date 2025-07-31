"""
Rollout Server V2 Package

This module provides a system for running large language model rollouts
with reward calculation capabilities.
"""

# -----------------------------------------------------------------------
# Core Components  
# -----------------------------------------------------------------------

# Import core configuration and message classes
from .core import (
    RolloutConfig,
    McpConfig,
    ApiConfig,
    GenerationConfig,
    ToolCallingConfig,
    SingleRolloutRequest, 
    SingleRolloutResult,
    BatchRolloutRequest,
    BatchRolloutResult,
    Messages,
    TextContent,
    ImageContent,
)

# -----------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------

from .core.utils import (
    calculate_resize_dimensions,
    smart_resize_image,
)

# -----------------------------------------------------------------------
# Main Execution Components
# -----------------------------------------------------------------------

# Import the reward manager before any modules that use it
from .rewards_manager import RewardManager

# Import rewards module to ensure functions are registered
from . import builtin_rewards

# Import the rest of the modules
from .rollout import run_single_rollout, run_batch_rollouts, run_batch_rollouts_async
from .monitor import ExecutionMonitor
from .mcp_client import MultiServerClientWithLimit

# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

__all__ = [
    # Core configuration and data structures
    'RolloutConfig',
    'McpConfig',
    'ApiConfig',
    'GenerationConfig',
    'ToolCallingConfig',
    'SingleRolloutRequest', 
    'SingleRolloutResult',
    'BatchRolloutRequest',
    'BatchRolloutResult',
    'Messages',
    'TextContent',
    'ImageContent', 
    
    # Utility functions
    'calculate_resize_dimensions',
    'smart_resize_image',
    
    # Main execution components
    'RewardManager',
    'run_single_rollout',
    'run_batch_rollouts',
    'run_batch_rollouts_async',
    'ExecutionMonitor',
    'MultiServerClientWithLimit',
] 