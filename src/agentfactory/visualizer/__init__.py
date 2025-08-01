"""
AgentFactory Visualizer Package

Provides HTML conversion and visualization tools for rollout results.

By default, uses the new unified visualizer system (visualizer_new).
Set environment variable AGENTFACTORY_USE_LEGACY_VISUALIZER=1 to use the original system.
"""

import os

# Check if legacy visualizer should be used
USE_LEGACY = os.getenv('AGENTFACTORY_USE_LEGACY_VISUALIZER', '0').lower() in ('1', 'true', 'yes')

if USE_LEGACY:
    # Use original visualizer
    from .html_converter import (
        messages_to_html,
        rollout_result_to_html,
        multiple_results_to_html,
        grouped_results_to_html
    )
    from .token_visualizer import (
        generate_token_visualizer_html,
        process_token_with_newlines,
        generate_multi_sequence_visualizer_html
    )
else:
    # Use new unified visualizer system
    try:
        from ..visualizer_new.legacy_api import (
            messages_to_html,
            rollout_result_to_html,
            multiple_results_to_html,
            grouped_results_to_html,
            generate_token_visualizer_html,
            process_token_with_newlines,
            generate_multi_sequence_visualizer_html
        )
    except ImportError:
        # Fallback to legacy if new system is not available
        from .html_converter import (
            messages_to_html,
            rollout_result_to_html,
            multiple_results_to_html,
            grouped_results_to_html
        )
        from .token_visualizer import (
            generate_token_visualizer_html,
            process_token_with_newlines,
            generate_multi_sequence_visualizer_html
        )

# Also export the new system's API for direct access
try:
    from ..visualizer_new import (
        create_conversation_app,
        create_token_app,
        create_unified_app
    )
    __new_api_available__ = True
except ImportError:
    __new_api_available__ = False

__all__ = [
    'messages_to_html',
    'rollout_result_to_html', 
    'multiple_results_to_html',
    'grouped_results_to_html',
    'generate_token_visualizer_html',
    'process_token_with_newlines',
    'generate_multi_sequence_visualizer_html'
]

# Add new API functions if available
if __new_api_available__:
    __all__.extend([
        'create_conversation_app',
        'create_token_app', 
        'create_unified_app'
    ])