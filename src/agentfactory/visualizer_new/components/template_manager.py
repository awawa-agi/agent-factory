"""
Template and asset manager for the unified visualizer
"""

import base64
from pathlib import Path
from typing import Optional, List


class TemplateManager:
    """Manages templates, CSS, and JavaScript resources for the unified visualizer"""
    
    def __init__(self, template_dir: Optional[Path] = None):
        if template_dir is None:
            # Default to assets directory relative to this file
            template_dir = Path(__file__).parent.parent / 'assets'
        self.template_dir = Path(template_dir)
    
    def load_css(self, filename: str) -> str:
        """Load a specific CSS file"""
        css_path = self.template_dir / 'styles' / filename
        if css_path.exists():
            return css_path.read_text(encoding='utf-8')
        return ""
    
    def load_all_css(self, background: str = "waves.webp") -> str:
        """Load all CSS files in the correct order
        
        Args:
            background: Background filename (e.g., "waves.webp", "spiral.webp", "flow.webp", "custom.jpg")
        """
        css_files = [
            'variables.css',
            'main.css', 
            'components.css',
            'animations.css',
            'token-viewer.css'  # Add token viewer styles
        ]
        
        css_content = []
        
        # Add background image as base64
        background_css = self._generate_background_css(background)
        if background_css:
            css_content.append("/* Background Image */")
            css_content.append(background_css)
            css_content.append("")
        
        for filename in css_files:
            content = self.load_css(filename)
            if content:
                css_content.append(f"/* {filename} */")
                css_content.append(content)
                css_content.append("")  # Add spacing between files
        
        return '\n'.join(css_content)
    
    def load_js(self, filename: str = 'main.js') -> str:
        """Load JavaScript file"""
        js_path = self.template_dir / 'scripts' / filename
        if js_path.exists():
            return js_path.read_text(encoding='utf-8')
        return ""
    
    def load_all_js(self) -> str:
        """Load all JavaScript files in the correct order"""
        js_content = []
        
        # First, add pako.js for decompression
        pako_js = self.load_js('pako.min.js')
        if pako_js:
            js_content.append("// pako.min.js - gzip decompression library")
            js_content.append(pako_js)
            js_content.append("")
        
        # Then add our custom JS files
        js_files = [
            'token_viewer.js',  # Load token viewer first
            'main.js'           # Main app logic
        ]
        
        for filename in js_files:
            content = self.load_js(filename)
            if content:
                js_content.append(f"// {filename}")
                js_content.append(content)
                js_content.append("")  # Add spacing between files
        
        return '\n'.join(js_content)
    
    def load_template(self, template_name: str) -> str:
        """Load an HTML template"""
        template_path = self.template_dir.parent / 'templates' / f'{template_name}.html'
        if template_path.exists():
            return template_path.read_text(encoding='utf-8')
        return ""
    
    def get_asset_path(self, asset_type: str, filename: str) -> Path:
        """Get the path to a specific asset"""
        return self.template_dir / asset_type / filename
    
    def asset_exists(self, asset_type: str, filename: str) -> bool:
        """Check if an asset exists"""
        return self.get_asset_path(asset_type, filename).exists()
    
    def list_available_backgrounds(self) -> List[str]:
        """List all available background files in the backgrounds directory"""
        backgrounds_dir = self.template_dir / 'backgrounds'
        if not backgrounds_dir.exists():
            return []
        
        # Get all image files
        image_extensions = {'.webp', '.jpg', '.jpeg', '.png'}
        background_files = []
        
        for file_path in backgrounds_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                background_files.append(file_path.name)
        
        return sorted(background_files)
    
    def _generate_background_css(self, background: str = "waves.webp") -> str:
        """Generate CSS with base64 encoded background image
        
        Args:
            background: Background filename (e.g., "waves.webp", "spiral.webp", "flow.webp", "custom.jpg")
        """
        # Direct filename usage only
        background_filename = background
        display_name = background.replace('.webp', '').replace('.jpg', '').replace('.png', '').replace('.jpeg', '')
        
        # Look for background in backgrounds directory
        background_path = self.template_dir / 'backgrounds' / background_filename
        
        if not background_path.exists():
            # Try fallback to waves background
            if background != "waves" and background != "waves.webp":
                return self._generate_background_css("waves.webp")
            return ""
        
        try:
            with open(background_path, 'rb') as f:
                base64_data = base64.b64encode(f.read()).decode()
            
            # Determine MIME type based on file extension
            if background_filename.lower().endswith('.webp'):
                mime_type = 'image/webp'
            elif background_filename.lower().endswith('.jpg') or background_filename.lower().endswith('.jpeg'):
                mime_type = 'image/jpeg'
            elif background_filename.lower().endswith('.png'):
                mime_type = 'image/png'
            else:
                mime_type = 'image/webp'  # Default fallback
            
            return f"""
/* Fixed Background Image - {display_name} */
.app {{
    background-image: url(data:{mime_type};base64,{base64_data});
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

/* Ensure background doesn't scroll with content */
.app-container {{
    background: none;
}}
"""
        except Exception as e:
            print(f"Warning: Could not load background image '{background}': {e}")
            return ""