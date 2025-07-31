"""
HTML section generators for rollout results
"""
import re
import json
import html as html_escape
from typing import Dict, Any, Optional, List
from ..rollout.core.messages import Messages, BaseMessage, ImageContent, TextContent, MessageUnion, MessageRole, AssistantMessage
from ..rollout.core.config import TokenStats


def _dumps_compact(obj, indent=2, max_len=8):
    """Compact JSON serialization"""
    txt = json.dumps(obj, indent=indent, ensure_ascii=False)
    pat = re.compile(r'\[\s+([\d",:\s.-]+?)\s+\]')
    def _join(m):
        items = ' '.join(x.strip() for x in m.group(1).splitlines())
        if items.count(',') < max_len:
            return '[' + items + ']'
        return m.group(0)
    return pat.sub(_join, txt)


def generate_rewards_section(reward_components: Dict[str, float], advantage: Optional[float] = None) -> str:
    """Generate rewards section"""
    rewards_items = []
    for key, value in reward_components.items():
        rewards_items.append(f'''
            <div class="reward-item">
                <span class="reward-label">{html_escape.escape(key)}</span>
                <span class="reward-value">{value:.3f}</span>
            </div>
        ''')
    
    if advantage is not None:
        rewards_items.append(f'''
            <div class="reward-item">
                <span class="reward-label">Advantage</span>
                <span class="reward-value">{advantage:.3f}</span>
            </div>
        ''')
    
    # Calculate total count for CSS class
    total_count = len(reward_components) + (1 if advantage is not None else 0)
    count_class = f"count-{min(total_count, 6)}"  # Cap at 6 for CSS class
    
    return f'''
    <div class="rewards-section">
        <div class="rewards-title">🏆 Reward Scores</div>
        <div class="rewards-grid {count_class}">
            {''.join(rewards_items)}
        </div>
    </div>
    '''


def generate_system_section(system_msg: BaseMessage, result_id: int | str = 0) -> str:
    """Generate system message section"""
    content = system_msg.get_text().strip()
    content_id = f"systemContent-{result_id}" if result_id != 0 and result_id != "0" else "systemContent"
    return f'''
    <div class="system-section">
        <div class="system-header" onclick="toggleSystem('{content_id}', this)">
            <span class="collapse-icon collapsed">▼</span>
            <strong>🔧 System Prompt</strong>
        </div>
        <div class="system-content hidden" id="{content_id}">
{html_escape.escape(content)}
        </div>
    </div>
    '''


def generate_conversation_section(messages: Messages) -> str:
    """Generate conversation section"""
    message_htmls = []
    
    for msg in messages:
        if msg.role == "system":
            continue
        
        message_html = generate_message_html(msg)
        message_htmls.append(message_html)
    
    return f'''
    <div class="conversation">
        {''.join(message_htmls)}
    </div>
    '''


def _generate_assistant_rewards_html(msg: AssistantMessage) -> str:
    """Generate HTML for AssistantMessage rewards and advantages"""
    rewards_items = []
    
    # Add turn reward components
    if msg.turn_reward_components:
        for key, value in msg.turn_reward_components.items():
            rewards_items.append(f'''
                <div class="reward-item">
                    <span class="reward-label">{html_escape.escape(key)}</span>
                    <span class="reward-value">{value:.3f}</span>
                </div>
            ''')
    
    # Add weighted turn reward
    if msg.weighted_turn_reward is not None:
        rewards_items.append(f'''
            <div class="reward-item">
                <span class="reward-label">Weighted Turn Reward</span>
                <span class="reward-value">{msg.weighted_turn_reward:.3f}</span>
            </div>
        ''')
    
    # Add MT-GRPO advantages
    advantage_fields = [
        ("Turn Advantage", msg.turn_level_advantage),
        ("Outcome Advantage", msg.outcome_level_advantage),
        ("EMT-GRPO Advantage", msg.emt_grpo_advantage)
    ]
    
    for label, value in advantage_fields:
        if value is not None:
            rewards_items.append(f'''
                <div class="reward-item">
                    <span class="reward-label">{label}</span>
                    <span class="reward-value">{value:.3f}</span>
                </div>
            ''')
    
    # Only generate HTML if there are items to display
    if not rewards_items:
        return ""
    
    return f'''
    <div class="message-rewards">
        <div class="message-rewards-title">📊 Turn Metrics</div>
        <div class="message-rewards-grid">
            {''.join(rewards_items)}
        </div>
    </div>
    '''


def generate_message_html(msg: MessageUnion) -> str:
    """Generate HTML for a single message"""
    role_configs = {
        "user": {"badge": "User", "icon": "👤"},
        "assistant": {"badge": "Assistant", "icon": "🤖"},
        "tool": {"badge": "Tool", "icon": "🔧"}
    }
    
    config = role_configs.get(msg.role, {"badge": msg.role.title(), "icon": "💬"})
    
    content_parts = []
    for content_item in msg.content:
        if content_item.type == "text":
            text = html_escape.escape(content_item.text).replace('\n', '<br>')
            content_parts.append(text)
        elif content_item.type == "image_url":
            content_parts.append(f'<img src="{content_item.url}" class="message-image" alt="Image">')
    
    content_html = '<br>'.join(content_parts)
    
    if msg.role == MessageRole.TOOL:
        tool_info = f"<strong>🔧 {msg.tool_name}()</strong><br>"
        tool_structured_output_html = ""
        if hasattr(msg, 'structured_output') and msg.structured_output:
            args_str = _dumps_compact(msg.structured_output)
            tool_structured_output_html = (
                f"<div class='tool-output'>{html_escape.escape(args_str)}</div>"
            )
        content_html = tool_info + content_html + tool_structured_output_html
    
    # Handle AssistantMessage rewards and advantages
    assistant_rewards_html = ""
    if isinstance(msg, AssistantMessage):
        assistant_rewards_html = _generate_assistant_rewards_html(msg)
    
    timestamp = "2025-07-05 14:30:00"
    
    return f'''
    <div class="message {msg.role}">
        <div class="message-header">
            <span class="role-badge">{config["badge"]}</span>
            <span class="timestamp">{timestamp}</span>
        </div>
        <div class="message-content">
            {content_html}
            {assistant_rewards_html}
        </div>
    </div>
    '''


def generate_metadata_section(token_stats: Optional[TokenStats],
                             metadata: Optional[Dict[str, Any]],
                             execution_time: Optional[float],
                             messages: Messages) -> str:
    """Generate metadata section"""
    
    cards = []
    
    if token_stats:
        total_tokens = token_stats.total_tokens
        completion_tokens = token_stats.num_completion_tokens or 0
        
        # Handle case where total_tokens is None (tokenize failed)
        if total_tokens is None:
            input_tokens = "N/A"
            total_tokens_display = "N/A"
        else:
            input_tokens = total_tokens - completion_tokens
            total_tokens_display = f"{total_tokens:,}"
        
        cards.append(f'''
        <div class="metadata-card">
            <h4>📊 Token Usage</h4>
            <div class="stat-item">
                <span class="stat-label">Input Tokens</span>
                <span class="stat-value">{input_tokens if isinstance(input_tokens, str) else f"{input_tokens:,}"}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Output Tokens</span>
                <span class="stat-value">{completion_tokens:,}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Total Tokens</span>
                <span class="stat-value">{total_tokens_display}</span>
            </div>
        </div>
        ''')
    
    role_counts = {"user": 0, "assistant": 0, "tool": 0}
    tool_calls = []
    
    for msg in messages:
        if msg.role in role_counts:
            role_counts[msg.role] += 1
        if msg.role == "tool":
            tool_calls.append(msg.tool_name)
    
    cards.append(f'''
    <div class="metadata-card">
        <h4>💬 Conversation Stats</h4>
        <div class="stat-item">
            <span class="stat-label">User Messages</span>
            <span class="stat-value">{role_counts["user"]}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">AI Responses</span>
            <span class="stat-value">{role_counts["assistant"]}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Tool Calls</span>
            <span class="stat-value">{role_counts["tool"]}</span>
        </div>
        {f'<div class="stat-item"><span class="stat-label">Execution Time</span><span class="stat-value">{execution_time:.1f}s</span></div>' if execution_time else ''}
    </div>
    ''')
    
    if tool_calls:
        tool_call_items = [f'<div class="tool-call">🔧 {tool}()</div>' for tool in tool_calls]
        cards.append(f'''
        <div class="metadata-card">
            <h4>🛠️ Tool Call Details</h4>
            {''.join(tool_call_items)}
        </div>
        ''')
    
    if metadata:
        metadata_items = []
        for key, value in metadata.items():
            if isinstance(value, (dict, list)):
                value_str = json.dumps(value, ensure_ascii=False, indent=2)
                metadata_items.append(f'''
                <div class="stat-item">
                    <span class="stat-label">{html_escape.escape(key)}</span>
                    <pre class="stat-value">{html_escape.escape(value_str)}</pre>
                </div>
                ''')
            else:
                metadata_items.append(f'''
                <div class="stat-item">
                    <span class="stat-label">{html_escape.escape(key)}</span>
                    <span class="stat-value">{html_escape.escape(str(value))}</span>
                </div>
                ''')
        
        if metadata_items:
            cards.append(f'''
            <div class="metadata-card">
                <h4>ℹ️ Additional Info</h4>
                {''.join(metadata_items)}
            </div>
            ''')
    
    return f'''
    <div class="metadata-section">
        <div class="metadata-title">📈 Statistics</div>
        <div class="metadata-grid">
            {''.join(cards)}
        </div>
    </div>
    '''