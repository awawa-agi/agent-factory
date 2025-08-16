"""
AgentFactory VisualizerNew Package

Modern unified visualization system for rollout results and token analysis.
Features glass-morphism design with seamless navigation between conversation 
and token views.
"""

from .app_generator import AgentFactoryApp
from .data.data_processor import UnifiedDataProcessor
from .components.template_manager import TemplateManager

# Main API functions - unified interface
def create_conversation_app(results, title="AgentFactory Conversation Viewer", background="waves.webp", **kwargs):
    """Create a conversation visualization app
    
    Args:
        results: Rollout results data
        title: App title
        background: Background filename (e.g., "waves.webp", "spiral.webp", "flow.webp", "custom.jpg")
        **kwargs: Additional options
    """
    processor = UnifiedDataProcessor()
    app_data = processor.process_rollout_results(results)
    return AgentFactoryApp.create_app(app_data, view_type="conversation", title=title, background=background, **kwargs)

def create_token_app(token_data, title="AgentFactory Token Analyzer", background="waves.webp", **kwargs):
    """Create a token visualization app
    
    Args:
        token_data: Token sequence data
        title: App title
        background: Background filename (e.g., "waves.webp", "spiral.webp", "flow.webp", "custom.jpg")
        **kwargs: Additional options
    """  
    processor = UnifiedDataProcessor()
    app_data = processor.process_token_data(token_data)
    return AgentFactoryApp.create_app(app_data, view_type="token", title=title, background=background, **kwargs)

def create_unified_app(rollout_results=None, token_data=None, title="AgentFactory Viewer", background="waves.webp", **kwargs):
    """Create a unified app with both conversation and token views
    
    Args:
        rollout_results: Rollout results data (optional)
        token_data: Token sequence data (optional)
        title: App title
        background: Background filename (e.g., "waves.webp", "spiral.webp", "flow.webp", "custom.jpg")
        **kwargs: Additional options
    """
    processor = UnifiedDataProcessor()
    app_data = processor.process_multiple_data_sources(rollout_results, token_data)
    return AgentFactoryApp.create_app(app_data, view_type="unified", title=title, background=background, **kwargs)

# Backward compatibility - legacy API functions
def grouped_results_to_html(grouped_results, title="AgentFactory Grouped Rollout Collection", **kwargs):
    """Legacy API - creates new unified app"""
    return create_conversation_app(grouped_results, title=title, **kwargs)

def multiple_results_to_html(results, title="AgentFactory Rollout Collection", **kwargs):
    """Legacy API - creates new unified app"""
    grouped = {"default_group": results} if isinstance(results, list) else results
    return create_conversation_app(grouped, title=title, **kwargs)

def generate_token_visualizer_html(sequences_data, title="Token Visualizer", **kwargs):
    """Legacy API - creates new token app"""
    return create_token_app(sequences_data, title=title, **kwargs)

def list_available_backgrounds():
    """List all available background files"""
    template_manager = TemplateManager()
    return template_manager.list_available_backgrounds()

__all__ = [
    'create_conversation_app',
    'create_token_app', 
    'create_unified_app',
    'grouped_results_to_html',
    'multiple_results_to_html',
    'generate_token_visualizer_html',
    'list_available_backgrounds',
    'AgentFactoryApp',
    'UnifiedDataProcessor',
    'TemplateManager'
]