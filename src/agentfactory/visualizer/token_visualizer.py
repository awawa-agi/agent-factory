"""Token Visualizer for AgentFactory - Refactored
Interactive visualization of token-level data including logprobs, entropies, and advantages
"""

import json
import html
from typing import List, Optional, Dict, Any, Tuple

def process_token_with_newlines(token: str) -> Tuple[str, int]:
    """Process tokens containing newlines for display"""
    original_token = token
    newline_count = 0
    
    while token.endswith('\n'):
        newline_count += 1
        token = token[:-1]
    
    if original_token.strip() == '' and '\n' in original_token:
        display_token = '\\n' * original_token.count('\n')
    else:
        display_token = token.replace('\n', '\\n')
        if newline_count > 0:
            display_token += '\\n' * newline_count
    
    return display_token, newline_count

def _identify_collapse_groups(assistant_masks: List[int], min_length: int = 3) -> List[Optional[int]]:
    """Identify consecutive groups of non-assistant tokens that should be collapsed"""
    collapse_groups = [None] * len(assistant_masks)
    group_id = 0
    i = 0
    
    while i < len(assistant_masks):
        if assistant_masks[i] == 0:
            start = i
            while i < len(assistant_masks) and assistant_masks[i] == 0:
                i += 1
            
            if i - start >= min_length:
                for j in range(start, i):
                    collapse_groups[j] = group_id
                group_id += 1
        else:
            i += 1
    
    return collapse_groups

def _process_sequence_data(tokens: List[str], logprobs: List[float], 
                          entropies: List[float], advantages: Optional[List[float]] = None,
                          assistant_masks: Optional[List[int]] = None,
                          collapse_min_length: int = 3) -> List[Dict]:
    """Process a single sequence into display format"""
    if advantages is None:
        avg_logprob = sum(logprobs) / len(logprobs) if logprobs else 0
        advantages = [lp - avg_logprob for lp in logprobs]
    
    if assistant_masks is None:
        assistant_masks = [1] * len(tokens)
    
    collapse_groups = _identify_collapse_groups(assistant_masks, collapse_min_length)
    
    sentence_data = []
    for i, token in enumerate(tokens):
        display_token, newline_count = process_token_with_newlines(token)
        sentence_data.append({
            "token": html.escape(display_token),
            "logprob": logprobs[i],
            "entropy": entropies[i],
            "advantage": advantages[i],
            "assistant_masks": assistant_masks[i],
            "newline_count": newline_count,
            "collapse_group": collapse_groups[i]
        })
    
    return sentence_data

def _generate_select_options(sequences_data: List[Dict]) -> str:
    """Generate HTML options for sequence selector"""
    options = []
    for i, seq_data in enumerate(sequences_data):
        display_id = seq_data.get("display_id") or seq_data.get("label") or f"Sequence {i+1}"
        options.append(f'<option value="{i}">{html.escape(display_id)}</option>')
    return '\n'.join(options)

def _build_html_document(title: str, select_options: str, js_data: str) -> str:
    """Build complete HTML document"""
    
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AgentFactory Token Visualizer</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #f0f8ff 0%, #e1f5fe 40%, #b3e5fc 80%, #e0f2f1 100%);
            min-height: 100vh;
            overflow-x: hidden;
            position: relative;
        }}
        .background-animation {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 0;
        }}
        .floating-shape {{
            position: absolute;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            animation: float 6s ease-in-out infinite;
        }}
        .floating-shape:nth-child(1) {{
            width: 80px;
            height: 80px;
            top: 20%;
            left: 10%;
            animation-delay: 0s;
        }}
        .floating-shape:nth-child(2) {{
            width: 120px;
            height: 120px;
            top: 60%;
            left: 80%;
            animation-delay: 2s;
        }}
        .floating-shape:nth-child(3) {{
            width: 60px;
            height: 60px;
            top: 80%;
            left: 20%;
            animation-delay: 4s;
        }}
        @keyframes float {{
            0%, 100% {{
                transform: translateY(0px) rotate(0deg);
                opacity: 0.5;
            }}
            50% {{
                transform: translateY(-20px) rotate(180deg);
                opacity: 0.8;
            }}
        }}
        @keyframes shimmer {{
            0%, 100% {{
                text-shadow: 0 0 20px rgba(255, 255, 255, 0.6), 0 0 40px rgba(174, 234, 255, 0.4);
                background: rgba(255, 255, 255, 0.15);
                transform: scale(1);
            }}
            50% {{
                text-shadow: 0 0 30px rgba(255, 255, 255, 0.9), 0 0 60px rgba(174, 234, 255, 0.6);
                background: rgba(255, 255, 255, 0.25);
                transform: scale(1.02);
            }}
        }}
        .container {{
            position: relative;
            z-index: 1;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        .glass-container {{
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }}
        .title {{
            font-size: 2.5rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 20px;
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(15px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 15px;
            padding: 20px;
            color: rgba(50, 50, 50, 0.9);
            text-shadow: 0 0 20px rgba(255, 255, 255, 0.6);
            animation: shimmer 4s ease-in-out infinite;
            box-shadow: 0 8px 32px rgba(174, 234, 255, 0.2);
        }}
        .controls {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            align-items: center;
            justify-content: center;
            margin-bottom: 20px;
        }}
        .select-wrapper {{
            position: relative;
        }}
        select {{
            background: rgba(255, 255, 255, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 10px;
            padding: 12px 40px 12px 16px;
            font-size: 16px;
            color: #1976d2;
            font-weight: 500;
            cursor: pointer;
            appearance: none;
            min-width: 200px;
        }}
        .button-group {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}
        .mode-btn {{
            background: rgba(255, 255, 255, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 10px;
            padding: 12px 20px;
            font-size: 14px;
            font-weight: 600;
            color: #1976d2;
            cursor: pointer;
            transition: all 0.15s ease;
        }}
        .mode-btn:hover {{
            background: rgba(255, 255, 255, 0.3);
            transform: translateY(-2px);
        }}
        .mode-btn.active {{
            background: linear-gradient(135deg, #1976d2, #42a5f5);
            color: white;
            box-shadow: 0 4px 15px rgba(25, 118, 210, 0.3);
        }}
        .tokens-container {{
            line-height: 1.4;
            font-size: 16px;
            word-wrap: break-word;
        }}
        .token {{
            display: inline-block;
            padding: 0px 1px;
            margin: 0px -1px;
            border-radius: 2px;
            cursor: pointer;
            transition: all 0.15s ease;
            position: relative;
            font-weight: 500;
        }}
        .token:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
            z-index: 10;
        }}
        .tooltip {{
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 8px 12px;
            border-radius: 8px;
            font-size: 12px;
            white-space: nowrap;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.2s ease;
            z-index: 100;
            margin-bottom: 5px;
        }}
        .token:hover .tooltip {{
            opacity: 1;
        }}
        .stats-container {{
            margin-bottom: 20px;
            padding: 15px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            font-size: 0.9em;
        }}
        .stat-item {{
            background: rgba(255, 255, 255, 0.2);
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-label {{
            display: block;
            color: #1976d2;
            font-weight: 600;
            margin-bottom: 5px;
        }}
        .stat-value {{
            display: block;
            color: #333;
            font-weight: 700;
            font-size: 1.1em;
        }}

        /* Modified collapsible button - removed reflection effect */
        .collapse-toggle {{
            display: inline-flex;
            align-items: center;
            background: linear-gradient(135deg, rgba(156, 39, 176, 0.15), rgba(103, 58, 183, 0.15));
            border: 1px solid rgba(156, 39, 176, 0.3);
            border-radius: 12px;
            padding: 8px 12px;
            margin: 2px 4px;
            cursor: pointer;
            transition: all 0.25s ease;
            font-size: 0.85em;
            font-weight: 500;
            color: #5e35b1;
            text-shadow: 0 1px 2px rgba(255, 255, 255, 0.3);
            box-shadow: 0 2px 8px rgba(156, 39, 176, 0.1);
            position: relative;
            overflow: hidden;
        }}
        .collapse-toggle:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 16px rgba(156, 39, 176, 0.2);
            border-color: rgba(156, 39, 176, 0.5);
        }}
        .collapse-icon {{
            margin-right: 6px;
            font-size: 0.9em;
            transition: transform 0.25s ease;
        }}
        .collapse-toggle.expanded .collapse-icon {{
            transform: rotate(90deg);
        }}
        .collapse-content {{
            display: block;
            margin-left: 16px;
            border-left: 3px solid rgba(156, 39, 176, 0.3);
            padding-left: 12px;
            margin-top: 8px;
            margin-bottom: 8px;
            border-radius: 0 8px 8px 0;
            background: rgba(156, 39, 176, 0.05);
        }}
        .collapse-content.hidden {{
            display: none;
        }}

        /* Modified legend - horizontally arranged color bars */
        .legend {{
            margin-top: 20px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.15);
        }}
        .legend-title {{
            font-size: 1.1em;
            font-weight: 600;
            color: #1976d2;
            text-align: center;
            margin-bottom: 20px;
            text-shadow: 0 1px 2px rgba(255, 255, 255, 0.3);
        }}
        .legend-section {{
            margin-bottom: 20px;
        }}
        .legend-section:last-child {{
            margin-bottom: 0;
        }}
        .legend-subtitle {{
            font-size: 0.9em;
            font-weight: 600;
            color: #424242;
            margin-bottom: 12px;
            text-align: center;
        }}
        .color-bars {{
            display: flex;
            flex-direction: row;
            gap: 32px;
            align-items: center;
            justify-content: center;
        }}
        .color-bar-item {{
            display: flex;
            align-items: center;
            gap: 16px;
            flex: 1;
            max-width: 300px;
        }}
        .color-bar {{
            position: relative;
            height: 20px;
            border-radius: 10px;
            border: 1px solid rgba(0,0,0,0.15);
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
            flex: 1;
            min-width: 150px;
        }}
        .color-bar-label {{
            font-size: 0.85em;
            font-weight: 500;
            color: #424242;
            min-width: 80px;
            text-align: right;
        }}
        .color-scale {{
            position: absolute;
            bottom: -20px;
            width: 100%;
            display: flex;
            justify-content: space-between;
            font-size: 0.7em;
            color: #666;
        }}
        .logprob-bar {{
            background: linear-gradient(to right, 
                rgba(76, 175, 80, 0.2) 0%, 
                rgba(76, 175, 80, 0.5) 25%, 
                rgba(76, 175, 80, 0.7) 50%, 
                rgba(76, 175, 80, 0.85) 75%, 
                rgba(76, 175, 80, 1) 100%);
        }}
        .entropy-bar {{
            background: linear-gradient(to right, 
                rgba(255, 87, 34, 0.2) 0%, 
                rgba(255, 87, 34, 0.5) 25%, 
                rgba(255, 87, 34, 0.7) 50%, 
                rgba(255, 87, 34, 0.85) 75%, 
                rgba(255, 87, 34, 1) 100%);
        }}

        @media (max-width: 768px) {{
            .container {{
                padding: 10px;
            }}
            .glass-container {{
                padding: 15px;
            }}
            .title {{
                font-size: 2rem;
            }}
            .controls {{
                flex-direction: column;
            }}
            .button-group {{
                justify-content: center;
            }}
            .tokens-container {{
                font-size: 14px;
            }}
            .color-bars {{
                flex-direction: column;
                gap: 16px;
            }}
            .color-bar-item {{
                max-width: 100%;
            }}
            .color-bar-label {{
                text-align: center;
                min-width: auto;
            }}
        }}
    </style>
</head>
<body>
    <div class="background-animation">
        <div class="floating-shape"></div>
        <div class="floating-shape"></div>
        <div class="floating-shape"></div>
    </div>
    <div class="container">
        <div class="glass-container">
            <h1 class="title">{html.escape(title)}</h1>
            <div class="controls">
                <div class="select-wrapper">
                    <select id="sentenceSelect">
                        {select_options}
                    </select>
                </div>
                <div class="button-group">
                    <button class="mode-btn" data-mode="none">No Color</button>
                    <button class="mode-btn" data-mode="logprob">LogProb</button>
                    <button class="mode-btn active" data-mode="entropy">Entropy</button>
                    <button class="mode-btn" data-mode="advantage">Advantage</button>
                    <button class="mode-btn" data-mode="combined">Combined</button>
                </div>
            </div>
        </div>
        <div class="glass-container">
            <div class="stats-container">
                <div class="stats-grid" id="statsGrid"></div>
                <div class="legend">
                    <div class="legend-section">
                        <div class="color-bars">
                            <div class="color-bar-item">
                                <div class="color-bar-label">LogProb</div>
                                <div class="color-bar logprob-bar">
                                    <div class="color-scale">
                                        <span>0</span>
                                        <span>-1</span>
                                        <span>-2</span>
                                        <span>-3</span>
                                        <span>-4</span>
                                    </div>
                                </div>
                            </div>
                            <div class="color-bar-item">
                                <div class="color-bar-label">Entropy</div>
                                <div class="color-bar entropy-bar">
                                    <div class="color-scale">
                                        <span>0</span>
                                        <span>1</span>
                                        <span>2</span>
                                        <span>3</span>
                                        <span>4</span>
                                        <span>5</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div class="glass-container">
            <div class="tokens-container" id="tokensContainer"></div>
        </div>
    </div>
    <script>
        const sentences = {js_data};
        let currentMode = 'entropy';
        let currentSentence = 0;

        document.addEventListener('DOMContentLoaded', function() {{
            renderTokens();
            setupEventListeners();
        }});

        function setupEventListeners() {{
            document.getElementById('sentenceSelect').addEventListener('change', function(e) {{
                currentSentence = parseInt(e.target.value);
                renderTokens();
            }});

            document.querySelectorAll('.mode-btn').forEach(btn => {{
                btn.addEventListener('click', function() {{
                    document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
                    this.classList.add('active');
                    currentMode = this.dataset.mode;
                    renderTokens();
                }});
            }});
        }}

        function renderTokens() {{
            const container = document.getElementById('tokensContainer');
            const tokens = sentences[currentSentence];
            
            container.innerHTML = generateTokensHTML(tokens);
            updateStats(tokens);
        }}

        function generateTokensHTML(tokens) {{
            let html = '';
            let i = 0;
            
            while (i < tokens.length) {{
                const token = tokens[i];
                
                if (token.collapse_group !== null && token.collapse_group !== undefined) {{
                    const groupId = token.collapse_group;
                    const groupTokens = [];
                    
                    while (i < tokens.length && tokens[i].collapse_group === groupId) {{
                        groupTokens.push(tokens[i]);
                        i++;
                    }}
                    
                    html += generateCollapseGroupHTML(groupTokens, groupId);
                }} else {{
                    html += generateTokenHTML(token);
                    i++;
                }}
            }}
            
            return html;
        }}

        function generateCollapseGroupHTML(groupTokens, groupId) {{
            const contentHTML = groupTokens.map(generateTokenHTML).join('');
            
            return `
                <span class="collapse-toggle" onclick="toggleCollapse(${{groupId}})" id="toggle-${{groupId}}">
                    <span class="collapse-icon">▶</span>
                    <span>Hidden: ${{groupTokens.length}} tokens</span>
                </span><br>
                <span class="collapse-content hidden" id="collapse-group-${{groupId}}">
                    ${{contentHTML}}
                </span>
            `;
        }}

        function generateTokenHTML(tokenData) {{
            const style = getTokenStyle(tokenData);
            const breakTags = '<br>'.repeat(tokenData.newline_count || 0);
            
            return `
                <span class="token" style="${{style}}">
                    ${{tokenData.token}}
                    <div class="tooltip">
                        LogProb: ${{tokenData.logprob.toFixed(3)}}<br>
                        Entropy: ${{tokenData.entropy.toFixed(3)}}<br>
                        Advantage: ${{tokenData.advantage.toFixed(3)}}<br>
                        Assistant: ${{tokenData.assistant_masks ? 'Yes' : 'No'}}
                    </div>
                </span>${{breakTags}}
            `;
        }}

        function toggleCollapse(groupId) {{
            const content = document.getElementById(`collapse-group-${{groupId}}`);
            const toggle = document.getElementById(`toggle-${{groupId}}`);
            const icon = toggle.querySelector('.collapse-icon');
            
            if (content.classList.contains('hidden')) {{
                content.classList.remove('hidden');
                toggle.classList.add('expanded');
                icon.textContent = '▼';
            }} else {{
                content.classList.add('hidden');
                toggle.classList.remove('expanded');
                icon.textContent = '▶';
            }}
        }}

        function updateStats(tokens) {{
            const statsGrid = document.getElementById('statsGrid');
            
            const assistantTokens = tokens.filter(t => t.assistant_masks === 1);
            const totalTokens = tokens.length;
            
            if (assistantTokens.length === 0) {{
                statsGrid.innerHTML = `
                    <div class="stat-item">
                        <span class="stat-label">Total Tokens</span>
                        <span class="stat-value">${{totalTokens}}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Assistant Tokens</span>
                        <span class="stat-value">0</span>
                    </div>
                `;
                return;
            }}
            
            const avgLogprob = assistantTokens.reduce((sum, t) => sum + t.logprob, 0) / assistantTokens.length;
            const avgEntropy = assistantTokens.reduce((sum, t) => sum + t.entropy, 0) / assistantTokens.length;
            const avgAdvantage = assistantTokens.reduce((sum, t) => sum + t.advantage, 0) / assistantTokens.length;
            
            statsGrid.innerHTML = `
                <div class="stat-item">
                    <span class="stat-label">Total Tokens</span>
                    <span class="stat-value">${{totalTokens}}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Assistant Tokens</span>
                    <span class="stat-value">${{assistantTokens.length}}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Avg LogProb</span>
                    <span class="stat-value">${{avgLogprob.toFixed(3)}}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Avg Entropy</span>
                    <span class="stat-value">${{avgEntropy.toFixed(3)}}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Avg Advantage</span>
                    <span class="stat-value">${{avgAdvantage.toFixed(3)}}</span>
                </div>
            `;
        }}

        function getTokenStyle(tokenData) {{
            if (tokenData.assistant_masks === 0) {{
                return 'background: transparent; border: 1px solid transparent; color: #9e9e9e;';
            }}

            switch (currentMode) {{
                case 'none':
                    return 'background: transparent; border: 1px solid transparent; color: inherit;';
                
                case 'logprob':
                    const logprobIntensity = Math.max(0, Math.min(1, (Math.abs(tokenData.logprob) / 2.5)));
                    return `background: rgba(76, 175, 80, ${{logprobIntensity * 0.7}}); border: 1px solid rgba(76, 175, 80, ${{logprobIntensity * 0.9}}); color: inherit;`;
                
                case 'entropy':
                    const entropyIntensity = Math.max(0, Math.min(1, tokenData.entropy / 2.5));
                    return `background: rgba(255, 87, 34, ${{entropyIntensity * 0.7}}); border: 1px solid rgba(255, 87, 34, ${{entropyIntensity * 0.9}}); color: inherit;`;
                
                case 'advantage':
                    const advantageIntensity = Math.max(0, Math.min(1, Math.abs(tokenData.advantage) / 2.0));
                    const color = tokenData.advantage >= 0 ? '76, 175, 80' : '156, 39, 176';
                    return `background: rgba(${{color}}, ${{advantageIntensity * 0.7}}); border: 1px solid rgba(${{color}}, ${{advantageIntensity * 0.9}}); color: inherit;`;
                
                case 'combined':
                    const logprobBg = Math.max(0, Math.min(1, (Math.abs(tokenData.logprob) / 2.5)));
                    const entropyBorder = Math.max(0, Math.min(3, tokenData.entropy * 1.2));
                    const borderWidth = entropyBorder > 0.3 ? Math.ceil(entropyBorder) : 0;
                    return `background: rgba(76, 175, 80, ${{logprobBg * 0.7}}); border: ${{borderWidth}}px solid rgba(255, 87, 34, 0.8); color: inherit;`;
                
                default:
                    return 'background: transparent; border: 1px solid transparent; color: inherit;';
            }}
        }}
    </script>
</body>
</html>'''

def generate_token_visualizer_html(
    sequences_data: List[Dict[str, Any]],
    title: str = "🧊 Token Visualizer 🧊",
    collapse_min_length: int = 3) -> str:
    """Generate interactive HTML visualization for token-level data"""
    if not sequences_data:
        sequences_data = [{
            "tokens": ["<empty>"],
            "logprobs": [0.0],
            "entropies": [0.0],
            "display_id": "Empty"
        }]
    
    # Support legacy format
    if isinstance(sequences_data, dict) or (
        isinstance(sequences_data, list) and 
        len(sequences_data) > 0 and 
        isinstance(sequences_data[0], str)
    ):
        if isinstance(sequences_data, dict):
            sequences_data = [sequences_data]
        else:
            sequences_data = [{
                "tokens": sequences_data[0] if len(sequences_data) > 0 else [],
                "logprobs": sequences_data[1] if len(sequences_data) > 1 else [],
                "entropies": sequences_data[2] if len(sequences_data) > 2 else [],
                "advantages": sequences_data[3] if len(sequences_data) > 3 else None,
                "assistant_masks": sequences_data[4] if len(sequences_data) > 4 else None,
            }]
    
    # Process all sequences
    all_sentences = []
    for seq_data in sequences_data:
        sentence_data = _process_sequence_data(
            seq_data["tokens"],
            seq_data["logprobs"],
            seq_data["entropies"],
            seq_data.get("advantages"),
            seq_data.get("assistant_masks"),
            collapse_min_length
        )
        all_sentences.append(sentence_data)
    
    # Generate components
    select_options = _generate_select_options(sequences_data)
    js_data = json.dumps(all_sentences, ensure_ascii=False, indent=8)
    
    return _build_html_document(title, select_options, js_data)

# Legacy compatibility
def generate_multi_sequence_visualizer_html(
    sequences_data: List[Dict[str, Any]],
    title: str = "🧊 Multi-Sequence Token Visualizer 🧊",
    collapse_min_length: int = 3) -> str:
    """Legacy function - now just calls the unified function"""
    return generate_token_visualizer_html(sequences_data, title, collapse_min_length)