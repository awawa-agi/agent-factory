"""
Core module for rollout system

Contains essential configuration and data structures:
- Configuration classes for rollout execution
- Message handling and content management
"""

# -----------------------------------------------------------------------
# Configuration Classes
# -----------------------------------------------------------------------

from .config import (
    RolloutConfig,
    McpConfig,
    ApiConfig,
    GenerationConfig,
    ToolCallingConfig,
    SingleRolloutRequest,
    SingleRolloutResult,
    BatchRolloutRequest,
    BatchRolloutResult,
)

# -----------------------------------------------------------------------
# Message Classes
# -----------------------------------------------------------------------

from .messages import (
    Messages,
    TextContent,
    ImageContent,
)

# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

__all__ = [
    # Configuration
    "RolloutConfig",
    "McpConfig", 
    "ApiConfig",
    "GenerationConfig",
    "ToolCallingConfig",
    "SingleRolloutRequest",
    "SingleRolloutResult",
    "BatchRolloutRequest",
    "BatchRolloutResult",
    
    # Messages
    "Messages",
    "TextContent",
    "ImageContent",
] 