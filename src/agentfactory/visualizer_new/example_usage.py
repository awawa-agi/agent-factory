"""
Example usage of the new AgentFactory VisualizerNew system
Demonstrates the unified interface and new features
"""

from typing import Dict, List, Any
from pathlib import Path

# Import the new visualizer API
from . import create_conversation_app, create_token_app, create_unified_app
from .legacy_api import grouped_results_to_html, generate_token_visualizer_html

# Mock data for demonstration
def create_mock_rollout_data() -> Dict[str, List[Dict[str, Any]]]:
    """Create mock rollout data for testing"""
    return {
        'test_prompt_1': [
            {
                'id': 'rollout_0',
                'is_success': True,
                'weighted_reward': 0.75,
                'reward_components': {
                    'task_completion': 1.0,
                    'efficiency': 0.8,
                    'safety': 0.9
                },
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a helpful assistant.',
                        'timestamp': '2025-07-05 14:30:00'
                    },
                    {
                        'role': 'user', 
                        'content': 'Solve this math problem: 2 + 2 = ?',
                        'timestamp': '2025-07-05 14:30:01'
                    },
                    {
                        'role': 'assistant',
                        'content': 'The answer is 4.',
                        'timestamp': '2025-07-05 14:30:02',
                        'rewards': {
                            'accuracy': 1.0,
                            'clarity': 0.9
                        }
                    }
                ],
                'metadata': {
                    'inputTokens': 20,
                    'outputTokens': 10,
                    'totalTokens': 30,
                    'executionTime': '1.2s'
                }
            }
        ]
    }

def create_mock_token_data() -> List[Dict[str, Any]]:
    """Create mock token data for testing"""
    return [
        {
            'sequence_id': 'seq_0',
            'display_id': 'Example Sequence',
            'tokens': ['The', ' answer', ' is', ' 4', '.'],
            'logprobs': [-0.1, -0.3, -0.2, -0.5, -0.1],
            'entropies': [1.2, 2.1, 1.8, 2.5, 1.0],
            'advantages': [0.1, -0.2, 0.0, -0.3, 0.2],
            'assistant_masks': [1, 1, 1, 1, 1]
        }
    ]

def demo_new_api():
    """Demonstrate the new unified API"""
    print("🌸 AgentFactory VisualizerNew Demo 🌸")
    print("=" * 50)
    
    # Create mock data
    rollout_data = create_mock_rollout_data()
    token_data = create_mock_token_data()
    
    print("\n1. Creating conversation app...")
    conversation_html = create_conversation_app(
        results=rollout_data,
        title="Demo Conversation Viewer"
    )
    print(f"   Generated HTML: {len(conversation_html)} characters")
    
    print("\n2. Creating token app...")
    token_html = create_token_app(
        token_data=token_data,
        title="Demo Token Analyzer"
    )
    print(f"   Generated HTML: {len(token_html)} characters")
    
    print("\n3. Creating unified app...")
    unified_html = create_unified_app(
        rollout_results=rollout_data,
        token_data=token_data,
        title="Demo Unified Viewer"
    )
    print(f"   Generated HTML: {len(unified_html)} characters")
    
    return conversation_html, token_html, unified_html

def demo_legacy_compatibility():
    """Demonstrate backward compatibility with legacy API"""
    print("\n4. Testing legacy API compatibility...")
    
    # This should work exactly like the old system
    rollout_data = create_mock_rollout_data()
    token_data = create_mock_token_data()
    
    # Legacy conversation API
    legacy_conversation = grouped_results_to_html(
        grouped_results=rollout_data,
        title="Legacy Conversation Viewer"
    )
    print(f"   Legacy conversation HTML: {len(legacy_conversation)} characters")
    
    # Legacy token API
    legacy_token = generate_token_visualizer_html(
        sequences_data=token_data,
        title="Legacy Token Visualizer"
    )
    print(f"   Legacy token HTML: {len(legacy_token)} characters")
    
    return legacy_conversation, legacy_token

def save_demo_files(output_dir: Path = None):
    """Save demo HTML files for manual inspection"""
    if output_dir is None:
        output_dir = Path(__file__).parent / "demo_output"
    
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n5. Saving demo files to {output_dir}...")
    
    # Generate all demo content
    conversation_html, token_html, unified_html = demo_new_api()
    legacy_conversation, legacy_token = demo_legacy_compatibility()
    
    # Save files
    files_to_save = [
        ("conversation_app.html", conversation_html),
        ("token_app.html", token_html),
        ("unified_app.html", unified_html),
        ("legacy_conversation.html", legacy_conversation),
        ("legacy_token.html", legacy_token)
    ]
    
    for filename, content in files_to_save:
        file_path = output_dir / filename
        file_path.write_text(content, encoding='utf-8')
        print(f"   Saved: {filename}")
    
    print(f"\n✅ Demo complete! Open files in {output_dir} to see the results.")

def compare_old_vs_new():
    """Compare the output sizes and features of old vs new system"""
    print("\n6. Comparing old vs new system...")
    
    # Mock the same data in both systems
    rollout_data = create_mock_rollout_data()
    
    # New system
    new_html = create_conversation_app(rollout_data, title="New System")
    
    # Legacy system (through compatibility layer)
    legacy_html = grouped_results_to_html(rollout_data, title="Legacy System")
    
    print(f"   New system HTML size: {len(new_html):,} characters")
    print(f"   Legacy system HTML size: {len(legacy_html):,} characters")
    print(f"   Size difference: {len(new_html) - len(legacy_html):+,} characters")
    
    # Feature comparison
    new_features = [
        "✅ Unified conversation + token views",
        "✅ Modern glass-morphism design", 
        "✅ Enhanced keyboard navigation",
        "✅ Improved mobile responsiveness",
        "✅ Better performance with large datasets",
        "✅ Modular component architecture",
        "✅ Advanced animation system"
    ]
    
    print("\n   New system features:")
    for feature in new_features:
        print(f"   {feature}")

if __name__ == "__main__":
    # Run the demo
    demo_new_api()
    demo_legacy_compatibility()
    compare_old_vs_new()
    
    # Uncomment to save demo files
    # save_demo_files()