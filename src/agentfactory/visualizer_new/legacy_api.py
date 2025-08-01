"""
Legacy API compatibility layer for AgentFactory visualizer
Provides backward compatibility with the original visualizer interface
"""

from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from .app_generator import AgentFactoryApp
from .data.data_processor import UnifiedDataProcessor
from .data.token_adapter import TokenAdapter
from ..rollout.core.config import SingleRolloutResult
from ..rollout.core.messages import Messages


def grouped_results_to_html(
    grouped_results: Dict[str, List[SingleRolloutResult]],
    title: str = "AgentFactory Grouped Rollout Collection",
    image_max_pixels: int = 28 * 28 * 128,
    show_system_message: bool = True,
    show_metadata: bool = True,
    template_dir: Optional[Path] = None
) -> str:
    """
    Legacy API: Convert grouped SingleRolloutResults to HTML with two-level navigation
    Now uses the new unified visualizer system
    """
    processor = UnifiedDataProcessor()
    app_data = processor.process_rollout_results(grouped_results)
    
    return AgentFactoryApp.create_app(
        app_data=app_data,
        view_type="conversation",
        title=title,
        template_dir=template_dir,
        image_max_pixels=image_max_pixels,
        show_system_message=show_system_message,
        show_metadata=show_metadata
    )


def multiple_results_to_html(
    results: List[SingleRolloutResult],
    title: str = "AgentFactory Rollout Collection",
    image_max_pixels: int = 28 * 28 * 128,
    show_system_message: bool = True,
    show_metadata: bool = True,
    template_dir: Optional[Path] = None
) -> str:
    """
    Legacy API: Convert multiple SingleRolloutResults to HTML with selector
    Now uses the new unified visualizer system
    """
    # Convert to grouped format for the new system
    grouped_results = {"default_group": results}
    
    return grouped_results_to_html(
        grouped_results=grouped_results,
        title=title,
        image_max_pixels=image_max_pixels,
        show_system_message=show_system_message,
        show_metadata=show_metadata,
        template_dir=template_dir
    )


def rollout_result_to_html(
    result: SingleRolloutResult,
    title: str = "AgentFactory Rollout Record",
    image_max_pixels: int = 28 * 28 * 128,
    show_system_message: bool = True,
    show_metadata: bool = True,
    template_dir: Optional[Path] = None
) -> str:
    """
    Legacy API: Convert SingleRolloutResult to HTML format
    Now uses the new unified visualizer system
    """
    return multiple_results_to_html(
        results=[result],
        title=title,
        image_max_pixels=image_max_pixels,
        show_system_message=show_system_message,
        show_metadata=show_metadata,
        template_dir=template_dir
    )


def messages_to_html(
    messages: Messages,
    reward_components: Optional[Dict[str, float]] = None,
    advantage: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
    token_stats=None,  # TokenStats type
    execution_time: Optional[float] = None,
    title: str = "LLM Conversation Display",
    image_max_pixels: int = 28 * 28 * 128,
    show_system_message: bool = True,
    show_metadata: bool = True,
    template_dir: Optional[Path] = None
) -> str:
    """
    Legacy API: Convert Messages to HTML format
    Now uses the new unified visualizer system
    """
    # Create a mock SingleRolloutResult to use the existing conversion logic
    from ..rollout.core.config import SingleRolloutResult
    
    mock_result = SingleRolloutResult(
        id="messages_conversion",
        messages=messages,
        reward_components=reward_components,
        advantage=advantage,
        metadata=metadata,
        token_stats=token_stats,
        execution_time=execution_time,
        is_success=True,
        weighted_reward=sum(reward_components.values()) if reward_components else 0.0
    )
    
    return rollout_result_to_html(
        result=mock_result,
        title=title,
        image_max_pixels=image_max_pixels,
        show_system_message=show_system_message,
        show_metadata=show_metadata,
        template_dir=template_dir
    )


def generate_token_visualizer_html(
    sequences_data: Union[List[Dict[str, Any]], Dict[str, Any], List[List[Any]]],
    title: str = "🧊 Token Visualizer 🧊",
    collapse_min_length: int = 3
) -> str:
    """
    Legacy API: Generate interactive HTML visualization for token-level data
    Now uses the new unified visualizer system
    """
    # Handle legacy formats
    if not sequences_data:
        sequences_data = [{
            "tokens": ["<empty>"],
            "logprobs": [0.0],
            "entropies": [0.0],
            "display_id": "Empty"
        }]
    
    # Convert legacy formats to new format
    adapted_data = TokenAdapter.adapt_legacy_format(sequences_data)
    
    processor = UnifiedDataProcessor()
    app_data = processor.process_token_data(adapted_data)
    
    return AgentFactoryApp.create_app(
        app_data=app_data,
        view_type="token",
        title=title,
        collapse_min_length=collapse_min_length
    )


def generate_multi_sequence_visualizer_html(
    sequences_data: List[Dict[str, Any]],
    title: str = "🧊 Multi-Sequence Token Visualizer 🧊",
    collapse_min_length: int = 3
) -> str:
    """
    Legacy API: Generate multi-sequence token visualizer
    Now redirects to the unified token visualizer
    """
    return generate_token_visualizer_html(
        sequences_data=sequences_data,
        title=title,
        collapse_min_length=collapse_min_length
    )


# Additional helper functions for backward compatibility
def process_token_with_newlines(token: str):
    """Legacy helper function"""
    return TokenAdapter._process_token_with_newlines(token)


# Export all legacy API functions
__all__ = [
    'grouped_results_to_html',
    'multiple_results_to_html', 
    'rollout_result_to_html',
    'messages_to_html',
    'generate_token_visualizer_html',
    'generate_multi_sequence_visualizer_html',
    'process_token_with_newlines'
]