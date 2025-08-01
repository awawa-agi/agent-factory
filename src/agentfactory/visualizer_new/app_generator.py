"""
Main application generator for AgentFactory VisualizerNew
Creates unified HTML applications with modern glass-morphism design
"""

import json
import html as html_escape
from typing import Dict, Any, Optional, List
from pathlib import Path

from .data.data_processor import AppData, GroupData, SampleData
from .components.template_manager import TemplateManager


class AgentFactoryApp:
    """Main application generator for the unified visualizer"""
    
    def __init__(self, template_dir: Optional[Path] = None):
        self.template_manager = TemplateManager(template_dir)
    
    @classmethod
    def create_app(cls, app_data: AppData, view_type: str = "conversation", 
                   title: str = "AgentFactory Viewer", **kwargs) -> str:
        """
        Create a complete HTML application
        
        Args:
            app_data: Processed application data
            view_type: Type of view ("conversation", "token", "unified")
            title: Application title
            **kwargs: Additional options
        """
        generator = cls(kwargs.get('template_dir'))
        return generator.generate_app(app_data, view_type, title, **kwargs)
    
    def generate_app(self, app_data: AppData, view_type: str, title: str, **kwargs) -> str:
        """Generate the complete HTML application"""
        
        # Get background option
        background = kwargs.get('background', 'waves.webp')
        
        # Prepare data for JavaScript
        js_data = self._prepare_js_data(app_data)
        
        # Build HTML parts
        html_parts = [
            self._generate_html_head(title, background),
            self._generate_app_body(app_data, title, js_data, view_type),
            self._generate_html_footer(js_data, app_data)
        ]
        
        return ''.join(html_parts)
    
    def _generate_html_head(self, title: str, background: str = "waves.webp") -> str:
        """Generate HTML head section with embedded styles"""
        css_content = self.template_manager.load_all_css(background)
        
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html_escape.escape(title)}</title>
    <style>
{css_content}
    </style>
</head>
<body class="app">'''
    
    def _generate_app_body(self, app_data: AppData, title: str, js_data: str, view_type: str) -> str:
        """Generate the main application body"""
        
        return f'''
    <!-- App Header -->
    <header class="app-header">
        <div class="header-content">
            <h1 class="header-title">🌸 {html_escape.escape(title)} 🌸</h1>
        </div>
    </header>

    <!-- Main App Container -->
    <div class="app-container">
        
        <!-- Group Navigation & Statistics -->
        <div class="content-section">
            <div class="glass-card p-6 animate-fade-in-up">
                
                <!-- Group Navigation Controls -->
                <div class="nav-controls mb-6">
                    <div class="flex items-center gap-3">
                        <label class="text-sm font-medium text-pink-700">Group:</label>
                        <button id="groupPrevBtn" class="nav-button">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polyline points="15,18 9,12 15,6"></polyline>
                            </svg>
                        </button>
                        <select id="groupSelect" class="app-select">
                            {self._generate_group_options(app_data)}
                        </select>
                        <button id="groupNextBtn" class="nav-button">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polyline points="9,18 15,12 9,6"></polyline>
                            </svg>
                        </button>
                    </div>
                    <span id="groupIndicator" class="nav-indicator">1 of {len(app_data.groups)}</span>
                </div>
                
                <!-- Statistics Grid -->
                <div class="stats-grid">
                    <div class="stats-card">
                        <div class="stats-card-header">
                            <svg class="stats-card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <line x1="18" y1="20" x2="18" y2="10"></line>
                                <line x1="12" y1="20" x2="12" y2="4"></line>
                                <line x1="6" y1="20" x2="6" y2="14"></line>
                            </svg>
                            <span class="stats-card-label">Success Rate</span>
                        </div>
                        <span id="successRate" class="stats-card-value">-</span>
                    </div>
                    
                    <div class="stats-card">
                        <div class="stats-card-header">
                            <svg class="stats-card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="8" r="7"></circle>
                                <polyline points="8.21,13.89 7,23 12,20 17,23 15.79,13.88"></polyline>
                            </svg>
                            <span class="stats-card-label">Avg Reward</span>
                        </div>
                        <span id="avgReward" class="stats-card-value">-</span>
                    </div>
                    
                    <div class="stats-card">
                        <div class="stats-card-header">
                            <svg class="stats-card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polyline points="22,12 18,12 15,21 9,3 6,12 2,12"></polyline>
                            </svg>
                            <span class="stats-card-label">Total Results</span>
                        </div>
                        <span id="totalResults" class="stats-card-value">-</span>
                    </div>

                    <div class="stats-card">
                        <div class="stats-card-header">
                            <svg class="stats-card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="10"></circle>
                                <polyline points="12,6 12,12 16,14"></polyline>
                            </svg>
                            <span class="stats-card-label">Exec Time</span>
                        </div>
                        <span id="executionTime" class="stats-card-value">-</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Sample Selector & Reward Scores -->
        <div class="content-section">
            <div class="glass-card p-4 animate-fade-in-up">
                
                <!-- Sample Navigation Controls -->
                <div class="nav-controls mb-4">
                    <div class="flex items-center gap-3">
                        <label class="text-sm font-medium text-pink-700">Generation:</label>
                        <button id="samplePrevBtn" class="nav-button">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polyline points="15,18 9,12 15,6"></polyline>
                            </svg>
                        </button>
                        <select id="sampleSelect" class="app-select app-select-sm">
                            <!-- Populated by JavaScript -->
                        </select>
                        <button id="sampleNextBtn" class="nav-button">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polyline points="9,18 15,12 9,6"></polyline>
                            </svg>
                        </button>
                    </div>
                    <span id="sampleIndicator" class="nav-indicator">-</span>
                </div>

                <!-- Reward Scores -->
                <div class="reward-scores">
                    <div class="reward-scores-header">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="12" cy="8" r="7"></circle>
                            <polyline points="8.21,13.89 7,23 12,20 17,23 15.79,13.88"></polyline>
                        </svg>
                        Reward Scores
                    </div>
                    <div id="rewardScoresGrid" class="reward-scores-grid">
                        <!-- Populated by JavaScript -->
                    </div>
                </div>
            </div>
        </div>

        <!-- Main Content Area -->
        <div class="content-section">
            <div class="glass-card p-6 animate-fade-in-up">
                <div id="mainContent">
                    <!-- Content populated by JavaScript -->
                    <div class="empty-state">
                        <div class="empty-state-icon">
                            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <line x1="18" y1="20" x2="18" y2="10"></line>
                                <line x1="12" y1="20" x2="12" y2="4"></line>
                                <line x1="6" y1="20" x2="6" y2="14"></line>
                            </svg>
                        </div>
                        <p class="empty-state-text">Loading conversation details...</p>
                    </div>
                </div>
            </div>
        </div>
        
    </div>
'''
    
    def _generate_html_footer(self, js_data: str, app_data: AppData) -> str:
        """Generate HTML footer with JavaScript and compressed data"""
        js_content = self.template_manager.load_all_js()
        
        # Generate compressed data scripts for token groups
        compressed_scripts = ""
        if app_data.view_type == "token":
            compressed_scripts = self._generate_compressed_data_scripts(app_data)
        
        return f'''
    {compressed_scripts}
    
    <!-- JavaScript -->
    <script>
        // Inject app data for JavaScript
        window.appData = {js_data};
    </script>
    <script>
{js_content}
    </script>
</body>
</html>'''
    
    def _generate_group_options(self, app_data: AppData) -> str:
        """Generate HTML options for group selector"""
        options = []
        
        for group_id, group_data in app_data.groups.items():
            # Create display text with stats
            display_text = f"{group_id} - Success: {group_data.stats.success_rate}, Avg Reward: {group_data.stats.avg_reward:.3f}"
            options.append(
                f'<option value="{html_escape.escape(group_id)}">{html_escape.escape(display_text)}</option>'
            )
        
        return '\n'.join(options)
    
    def _prepare_js_data(self, app_data: AppData) -> str:
        """Prepare app data for JavaScript consumption"""
        
        # Convert AppData to dictionary format suitable for JSON
        js_data = {
            "view_type": app_data.view_type,
            "groups": {}
        }
        
        for group_id, group_data in app_data.groups.items():
            js_data["groups"][group_id] = {
                "group_id": group_data.group_id,
                "stats": {
                    "total_results": group_data.stats.total_results,
                    "success_rate": group_data.stats.success_rate,
                    "avg_reward": group_data.stats.avg_reward,
                    "execution_time": group_data.stats.execution_time
                },
                "samples": []
            }
            
            for sample in group_data.samples:
                sample_data = {
                    "sample_id": sample.sample_id,
                    "name": sample.name,
                    "is_success": sample.is_success,
                    "weighted_reward": sample.weighted_reward,
                    "rewards": sample.rewards or {},
                    "system_prompt": sample.system_prompt,
                    "metadata": sample.metadata or {}
                }
                
                # Add conversation data if present
                if sample.conversation:
                    sample_data["conversation"] = sample.conversation
                
                # Add token data if present  
                if sample.token_data:
                    sample_data["token_data"] = sample.token_data
                
                js_data["groups"][group_id]["samples"].append(sample_data)
        
        return json.dumps(js_data, ensure_ascii=False, indent=2)
    
    def _generate_compressed_data_scripts(self, app_data: AppData) -> str:
        """Generate compressed data scripts for token groups"""
        scripts = []
        
        # Generate group stats (lightweight, uncompressed)
        group_stats = {}
        for group_id, group_data in app_data.groups.items():
            # Extract compressed group data if available
            if group_data.samples and group_data.samples[0].token_data:
                token_data = group_data.samples[0].token_data
                if 'compressed_group' in token_data:
                    compressed_group = token_data['compressed_group']
                    group_stats[group_id] = compressed_group['stats']
        
        if group_stats:
            scripts.append(f'''
    <!-- Group Statistics (uncompressed for immediate access) -->
    <script type="application/json" id="groups-stats">
{json.dumps(group_stats, indent=2)}
    </script>''')
        
        # Generate compressed data scripts for each group
        for group_id, group_data in app_data.groups.items():
            if group_data.samples and group_data.samples[0].token_data:
                token_data = group_data.samples[0].token_data
                if 'compressed_group' in token_data:
                    compressed_group = token_data['compressed_group']
                    compressed_data = compressed_group['compressed_data']
                    
                    # Escape group_id for HTML id attribute
                    safe_group_id = html_escape.escape(group_id).replace(' ', '_').replace('-', '_')
                    
                    scripts.append(f'''
    <!-- Compressed data for group: {group_id} -->
    <script type="application/octet-stream" id="group-{safe_group_id}" data-encoding="base64+gzip">
{compressed_data}
    </script>''')
        
        return '\n'.join(scripts)


class ComponentRenderer:
    """Helper class for rendering individual components"""
    
    @staticmethod
    def render_message_bubble(message: Dict[str, Any], index: int = 0) -> str:
        """Render a single message bubble"""
        role_class = message["role"].lower()
        
        html = f'''
        <div class="message animate-fade-in-up" style="animation-delay: {index * 0.1}s">
            <div class="message-header">
                <span class="message-role-badge {role_class}">
                    {message["role"].upper()}
                </span>
                <span class="message-timestamp">{message.get("timestamp", "")}</span>
            </div>
            <div class="message-content {role_class}">
                <div class="message-text">{html_escape.escape(message["content"])}</div>
        '''
        
        # Add turn metrics if present
        if message.get("rewards"):
            html += '''
                <div class="turn-metrics">
                    <div class="turn-metrics-title">📊 Turn Metrics</div>
                    <div class="turn-metrics-grid">
            '''
            
            for key, value in message["rewards"].items():
                html += f'''
                    <div class="turn-metric-item">
                        <div class="turn-metric-label">{html_escape.escape(key)}</div>
                        <div class="turn-metric-value">{value:.3f}</div>
                    </div>
                '''
            
            html += '</div></div>'
        
        html += '</div></div>'
        return html
    
    @staticmethod
    def render_reward_item(key: str, value: float) -> str:
        """Render a single reward item"""
        return f'''
        <div class="reward-item animate-scale-in">
            <div class="reward-item-label">{html_escape.escape(key)}</div>
            <div class="reward-item-value">{value:.3f}</div>
        </div>
        '''
    
    @staticmethod
    def render_stats_card(icon_svg: str, label: str, value: str, element_id: str = "") -> str:
        """Render a statistics card"""
        return f'''
        <div class="stats-card">
            <div class="stats-card-header">
                {icon_svg}
                <span class="stats-card-label">{label}</span>
            </div>
            <span id="{element_id}" class="stats-card-value">{value}</span>
        </div>
        '''