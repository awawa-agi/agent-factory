"""
Beautiful Glass Effect HTML Converter (Refactored)
Supporting conversation record conversion with sakura pink theme and glass effect design
"""
import html as html_escape
import json
from typing import Dict, Any, Optional, List
from pathlib import Path

from ..rollout.core.messages import Messages, ImageContent
from ..rollout.core.config import SingleRolloutResult, TokenStats
from .template_manager import TemplateManager
from .section_generators import (
    generate_rewards_section,
    generate_system_section, 
    generate_conversation_section,
    generate_metadata_section
)


def _process_messages_images(messages: Messages, image_max_pixels: int) -> Messages:
    """Process and resize images in messages"""
    processed = Messages(root=[])
    
    for msg in messages:
        new_content = []
        for content_item in msg.content:
            if content_item.type == "image_url":
                img_copy = ImageContent(image_url=content_item.image_url)
                img_copy = img_copy.resize_smart(max_pixels=image_max_pixels, factor=1)
                new_content.append(img_copy)
            else:
                new_content.append(content_item)
        
        msg_dict = msg.model_dump()
        msg_dict['content'] = new_content
        processed.append(msg_dict)
    
    return processed


def _get_system_message(messages: Messages):
    """Extract system message"""
    for msg in messages:
        if msg.role == "system":
            return msg
    return None


def messages_to_html(messages: Messages,
                    reward_components: Optional[Dict[str, float]] = None,
                    advantage: Optional[float] = None,
                    metadata: Optional[Dict[str, Any]] = None,
                    token_stats: Optional[TokenStats] = None,
                    execution_time: Optional[float] = None,
                    title: str = "LLM Conversation Display",
                    image_max_pixels: int = 28 * 28 * 128,
                    show_system_message: bool = True,
                    show_metadata: bool = True,
                    template_dir: Optional[Path] = None) -> str:
    """Convert Messages to beautiful glass effect HTML format"""
    
    template_manager = TemplateManager(template_dir)
    processed_messages = _process_messages_images(messages, image_max_pixels)
    
    html_parts = [
        template_manager.get_html_head(html_escape.escape(title)),
        template_manager.get_header_section(html_escape.escape(title))
    ]
    
    if reward_components:
        html_parts.append(generate_rewards_section(reward_components, advantage))
    
    system_msg = _get_system_message(processed_messages)
    if system_msg and show_system_message:
        html_parts.append(generate_system_section(system_msg))
    
    html_parts.append(generate_conversation_section(processed_messages))
    
    if show_metadata and any([token_stats, metadata, execution_time]):
        html_parts.append(generate_metadata_section(
            token_stats, metadata, execution_time, processed_messages
        ))
    
    html_parts.append('</div>')  # Close chat-container
    html_parts.append(template_manager.get_html_footer())
    
    return ''.join(html_parts)


def rollout_result_to_html(result: SingleRolloutResult,
                          title: str = "AgentFactory Rollout Record",
                          image_max_pixels: int = 28 * 28 * 128,
                          show_system_message: bool = True,
                          show_metadata: bool = True,
                          template_dir: Optional[Path] = None) -> str:
    """Convert SingleRolloutResult to beautiful glass effect HTML format"""
    return messages_to_html(
        messages=result.messages or Messages(root=[]),
        reward_components=result.reward_components,
        advantage=result.advantage,
        metadata=result.metadata,
        token_stats=result.token_stats,
        execution_time=result.execution_time,
        title=title,
        image_max_pixels=image_max_pixels,
        show_system_message=show_system_message,
        show_metadata=show_metadata,
        template_dir=template_dir
    )


def multiple_results_to_html(results: List[SingleRolloutResult],
                           title: str = "AgentFactory Rollout Collection",
                           image_max_pixels: int = 28 * 28 * 128,
                           show_system_message: bool = True,
                           show_metadata: bool = True,
                           template_dir: Optional[Path] = None) -> str:
    """Convert multiple SingleRolloutResults to beautiful glass effect HTML with selector"""
    
    if not results:
        return _get_simple_html("No conversation records", template_dir)
    
    template_manager = TemplateManager(template_dir)
    
    html_parts = [
        template_manager.get_html_head(html_escape.escape(title)),
        template_manager.get_header_section(html_escape.escape(title)),
        template_manager.get_selector_section(title, results)
    ]
    
    for i, result in enumerate(results):
        processed_messages = _process_messages_images(
            result.messages or Messages(root=[]), image_max_pixels
        )
        
        sections = []
        
        if result.reward_components:
            sections.append(generate_rewards_section(result.reward_components, result.advantage))
        
        system_msg = _get_system_message(processed_messages)
        if system_msg and show_system_message:
            sections.append(generate_system_section(system_msg, result_id=i))
        
        sections.append(generate_conversation_section(processed_messages))
        
        if show_metadata:
            sections.append(generate_metadata_section(
                result.token_stats, result.metadata, result.execution_time, processed_messages
            ))
        
        visibility = "" if i == 0 else " style='display: none;'"
        html_parts.append(f'<div class="result-content" id="result-{i}"{visibility}>{"".join(sections)}</div>')
    
    html_parts.append('</div>')  # Close chat-container
    html_parts.append(template_manager.get_html_footer())
    
    return ''.join(html_parts)


def grouped_results_to_html(grouped_results: Dict[str, List[SingleRolloutResult]],
                           title: str = "AgentFactory Grouped Rollout Collection",
                           image_max_pixels: int = 28 * 28 * 128,
                           show_system_message: bool = True,
                           show_metadata: bool = True,
                           template_dir: Optional[Path] = None) -> str:
    """Convert pre-grouped SingleRolloutResults to HTML with two-level navigation"""
    
    if not grouped_results:
        return _get_simple_html("No conversation records", template_dir)
    
    groups = grouped_results
    
    template_manager = TemplateManager(template_dir)
    
    html_parts = [
        template_manager.get_html_head(html_escape.escape(title)),
        template_manager.get_header_section(html_escape.escape(title)),
        _generate_grouped_selector_section(groups, title),
        _generate_group_stats_section(),
    ]
    
    # Generate content for each group and result
    for group_id, group_results in groups.items():
        for i, result in enumerate(group_results):
            processed_messages = _process_messages_images(
                result.messages or Messages(root=[]), image_max_pixels
            )
            
            sections = []
            
            if result.reward_components:
                sections.append(generate_rewards_section(result.reward_components, result.advantage))
            
            system_msg = _get_system_message(processed_messages)
            if system_msg and show_system_message:
                sections.append(generate_system_section(system_msg, result_id=f"{group_id}_{i}"))
            
            sections.append(generate_conversation_section(processed_messages))
            
            if show_metadata:
                sections.append(generate_metadata_section(
                    result.token_stats, result.metadata, result.execution_time, processed_messages
                ))
            
            # Add visibility control based on whether this is the first result
            is_first_group = group_id == next(iter(groups.keys()))
            is_first_result = i == 0
            visibility = "" if (is_first_group and is_first_result) else " style='display: none;'"
            
            html_parts.append(f'<div class="grouped-result-content" data-group="{group_id}" data-result="{i}"{visibility}>{"".join(sections)}</div>')
    
    html_parts.append('</div>')  # Close chat-container
    html_parts.append(_get_grouped_footer_script(groups))
    html_parts.append(template_manager.get_html_footer())
    
    return ''.join(html_parts)



def _generate_grouped_selector_section(groups: Dict[str, List[SingleRolloutResult]], title: str) -> str:
    """Generate grouped selector section HTML"""
    group_options = []
    for group_id, group_results in groups.items():
        success_count = sum(1 for r in group_results if r.is_success)
        total_count = len(group_results)
        success_rate = f"{success_count}/{total_count}"
        avg_reward = sum(r.weighted_reward or 0 for r in group_results if r.weighted_reward is not None) / len(group_results)
        
        group_options.append(f'''
            <option value="{html_escape.escape(group_id)}" data-results='{len(group_results)}'>
                {html_escape.escape(group_id)} - Success: {success_rate}, Avg Reward: {avg_reward:.3f}
            </option>
        ''')
    
    return f'''
    <div class="grouped-controls" style="
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(255, 182, 193, 0.2);
    ">
        <div style="display: flex; gap: 20px; align-items: center; flex-wrap: wrap;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <label for="groupSelect" style="font-weight: 600; color: #8B4A6B;">📁 Group:</label>
                <select id="groupSelect" style="
                    background: rgba(255, 255, 255, 0.3);
                    border: 1px solid rgba(255, 255, 255, 0.5);
                    border-radius: 8px;
                    padding: 8px 12px;
                    color: #8B4A6B;
                    font-weight: 500;
                    min-width: 300px;
                ">
                    {''.join(group_options)}
                </select>
            </div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <label for="resultSelect" style="font-weight: 600; color: #8B4A6B;">📄 Result:</label>
                <select id="resultSelect" style="
                    background: rgba(255, 255, 255, 0.3);
                    border: 1px solid rgba(255, 255, 255, 0.5);
                    border-radius: 8px;
                    padding: 8px 12px;
                    color: #8B4A6B;
                    font-weight: 500;
                    min-width: 200px;
                "></select>
            </div>
        </div>
    </div>
    '''


def _generate_group_stats_section() -> str:
    """Generate group statistics section HTML"""
    return f'''
    <div id="groupStats" style="
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(255, 182, 193, 0.1);
    ">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px;">
            <div style="text-align: center;">
                <div style="font-size: 1.2em; font-weight: 700; color: #8B4A6B;" id="groupSuccessRate">-</div>
                <div style="font-size: 0.9em; color: #8B4A6B; opacity: 0.8;">Success Rate</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.2em; font-weight: 700; color: #8B4A6B;" id="groupAvgReward">-</div>
                <div style="font-size: 0.9em; color: #8B4A6B; opacity: 0.8;">Avg Reward</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.2em; font-weight: 700; color: #8B4A6B;" id="groupTotalResults">-</div>
                <div style="font-size: 0.9em; color: #8B4A6B; opacity: 0.8;">Total Results</div>
            </div>
        </div>
    </div>
    '''


def _get_grouped_footer_script(groups: Dict[str, List[SingleRolloutResult]]) -> str:
    """Generate JavaScript for grouped navigation"""
    # Prepare groups data for JavaScript
    groups_data = {}
    for group_id, group_results in groups.items():
        groups_data[group_id] = [
            {
                'id': result.id,
                'is_success': result.is_success,
                'weighted_reward': result.weighted_reward or 0,
                'execution_time': result.execution_time or 0,
            }
            for result in group_results
        ]
    
    groups_json = json.dumps(groups_data, ensure_ascii=False)
    
    return f'''
    <script>
        const groupsData = {groups_json};
        let currentGroup = Object.keys(groupsData)[0];
        let currentResult = 0;

        document.addEventListener('DOMContentLoaded', function() {{
            setupGroupedEventListeners();
            updateResultSelector();
            updateGroupStats();
        }});

        function setupGroupedEventListeners() {{
            document.getElementById('groupSelect').addEventListener('change', function(e) {{
                currentGroup = e.target.value;
                currentResult = 0;
                updateResultSelector();
                updateGroupStats();
                showGroupedResult();
            }});

            document.getElementById('resultSelect').addEventListener('change', function(e) {{
                currentResult = parseInt(e.target.value);
                showGroupedResult();
            }});
        }}

        function updateResultSelector() {{
            const resultSelect = document.getElementById('resultSelect');
            const groupResults = groupsData[currentGroup];
            
            if (!groupResults) return;
            
            resultSelect.innerHTML = groupResults.map((result, i) => {{
                const successIcon = result.is_success ? '✓' : '✗';
                const rewardText = result.weighted_reward !== null && result.weighted_reward !== undefined ? result.weighted_reward.toFixed(3) : 'N/A';
                return `<option value="${{i}}">Gen ${{i}} (${{successIcon}}) - Reward: ${{rewardText}}</option>`;
            }}).join('');
            
            resultSelect.value = currentResult.toString();
        }}

        function updateGroupStats() {{
            const groupResults = groupsData[currentGroup];
            if (!groupResults) return;
            
            const successCount = groupResults.filter(r => r.is_success).length;
            const totalCount = groupResults.length;
            const avgReward = groupResults.reduce((sum, r) => sum + r.weighted_reward, 0) / totalCount;
            
            document.getElementById('groupSuccessRate').textContent = `${{successCount}}/${{totalCount}} (${{(successCount/totalCount*100).toFixed(1)}}%)`;
            document.getElementById('groupAvgReward').textContent = avgReward.toFixed(3);
            document.getElementById('groupTotalResults').textContent = totalCount;
        }}

        function showGroupedResult() {{
            // Hide all results
            document.querySelectorAll('.grouped-result-content').forEach(el => {{
                el.style.display = 'none';
            }});
            
            // Show selected result
            const targetElement = document.querySelector(`[data-group="${{currentGroup}}"][data-result="${{currentResult}}"]`);
            if (targetElement) {{
                targetElement.style.display = 'block';
            }}
        }}
    </script>
    '''


def _get_simple_html(message: str, template_dir: Optional[Path] = None) -> str:
    """Get simple error message HTML"""
    template_manager = TemplateManager(template_dir)
    css_content = template_manager.load_css()
    
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Conversation Record</title>
    <style>
{css_content}
    </style>
</head>
<body>
    <div class="container">
        <div class="message" style="
            text-align: center;
            padding: 50px;
            background: rgba(255, 255, 255, 0.3);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 8px 32px rgba(255, 182, 193, 0.3);
            color: #8B4A6B;
        ">
            {html_escape.escape(message)}
        </div>
    </div>
</body>
</html>'''