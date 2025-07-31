"""
Template manager for loading CSS and JavaScript resources
"""
from pathlib import Path
from typing import Optional


class TemplateManager:
    """Manages HTML templates, CSS, and JavaScript resources"""
    
    def __init__(self, template_dir: Optional[Path] = None):
        if template_dir is None:
            # Default to templates directory next to this file
            template_dir = Path(__file__).parent / 'templates'
        self.template_dir = Path(template_dir)
    
    def load_css(self) -> str:
        """Load CSS styles"""
        css_path = self.template_dir / 'styles.css'
        if css_path.exists():
            return css_path.read_text(encoding='utf-8')
        return ""
    
    def load_js(self) -> str:
        """Load JavaScript"""
        js_path = self.template_dir / 'scripts.js'
        if js_path.exists():
            return js_path.read_text(encoding='utf-8')
        return ""
    
    def get_html_head(self, title: str) -> str:
        """Generate HTML head with embedded CSS"""
        css_content = self.load_css()
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css" rel="stylesheet">
    <style>
{css_content}
    </style>
</head>
<body>
    <div class="glow-orb"></div>
    <div class="glow-orb"></div>
    <div class="glow-orb"></div>

    <div class="container">'''

    def get_html_footer(self) -> str:
        """Generate HTML footer with embedded JavaScript"""
        js_content = self.load_js()
        return f'''
    </div>
    
    <script>
{js_content}
    </script>
</body>
</html>'''

    def get_header_section(self, title: str) -> str:
        """Generate header section"""
        return f'''
        <div class="header animate__animated animate__fadeInDown">
            <h1>🌸 {title} 🌸</h1>
        </div>

        <div class="chat-container">'''

    def get_selector_section(self, title: str, results) -> str:
        """Generate selector section for multiple results"""
        options = []
        
        for i, result in enumerate(results):
            result_id = result.id or f"Result {i+1}"
            
            # Calculate reward sum
            if result.weighted_reward is not None:
                reward_display = f"{result.weighted_reward:.2f}"
            else:
                reward_display = "None"
            
            # Get advantage value
            if result.advantage is not None:
                advantage_display = f"{result.advantage:.2f}"
            else:
                advantage_display = "None"
            
            # Calculate tool calls count
            tool_calls_count = 0
            for msg in result.messages or []:
                if msg.role == "tool":
                    tool_calls_count += 1
            
            # Generate option text with ID, reward sum, advantage, and tool calls count
            option_text = f"{result_id} | Reward: {reward_display} | Advantage: {advantage_display} | Tools: {tool_calls_count}"
            options.append(f'<option value="{i}">{option_text}</option>')
        
        return f'''
        <div class="selector-section animate__animated animate__fadeInDown">
            <span class="selector-label">🔄 Select Conversation:</span>
            <select class="result-selector" onchange="switchResult(this.value)">
                {''.join(options)}
            </select>
        </div>'''