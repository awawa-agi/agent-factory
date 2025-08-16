/**
 * Token Viewer with Gzip Compression Support
 * Modern, clean implementation focused on compressed token groups
 * Enhanced with React-style HSL color system and pre-calculated colors
 */

/**
 * Advanced HSL Color System (from React component)
 * Provides rich multi-dimensional color mapping for better data visualization
 */

/**
 * Calculate rich HSL color for log probability values
 * Maps logprob (-10 to 0) to a red-orange-yellow gradient
 */
function getLogprobColor(logprob) {
    const normalized = Math.max(0, Math.min(1, (logprob + 10) / 10));
    
    if (normalized < 0.5) {
        const t = normalized * 2;
        const hue = 0 + (t * 30);           // 0° (red) to 30° (orange)
        const saturation = 100 - (t * 20);  // 100% to 80%
        const lightness = 50 + (t * 20);    // 50% to 70%
        return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
    } else {
        const t = (normalized - 0.5) * 2;
        const hue = 30;                      // Keep orange hue
        const saturation = 80 - (t * 80);   // 80% to 0%
        const lightness = 70 + (t * 25);    // 70% to 95%
        return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
    }
}

/**
 * Calculate rich HSL color for entropy values
 * Maps entropy (0 to 5) to a blue-purple gradient
 */
function getEntropyColor(entropy) {
    const normalized = Math.max(0, Math.min(1, entropy / 5));
    
    if (normalized < 0.5) {
        const t = normalized * 2;
        const hue = 240;                     // Blue
        const saturation = t * 80;           // 0% to 80%
        const lightness = 95 - (t * 25);    // 95% to 70%
        return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
    } else {
        const t = (normalized - 0.5) * 2;
        const hue = 240 + (t * 60);         // Blue to purple (240° to 300°)
        const saturation = 80 + (t * 20);   // 80% to 100%
        const lightness = 70 - (t * 20);    // 70% to 50%
        return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
    }
}

/**
 * Calculate rich HSL color for advantage values
 * Maps advantage (-1 to 1) to a red-green gradient
 */
function getAdvantageColor(advantage) {
    const normalized = Math.max(-1, Math.min(1, advantage));
    
    if (advantage >= 0) {
        // Positive advantage - green gradient
        const t = normalized;
        const hue = 120;                     // Green
        const saturation = 70 + (t * 30);   // 70% to 100%
        const lightness = 70 - (t * 20);    // 70% to 50%
        return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
    } else {
        // Negative advantage - red gradient
        const t = Math.abs(normalized);
        const hue = 0;                       // Red
        const saturation = 70 + (t * 30);   // 70% to 100%
        const lightness = 70 - (t * 20);    // 70% to 50%
        return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
    }
}

class TokenDataManager {
    constructor() {
        this.groupsStats = this._loadGroupsStats();
        this.loadedGroups = new Map();    // groupId -> decompressed data
    }
    
    _loadGroupsStats() {
        const element = document.getElementById('groups-stats');
        if (element) {
            try {
                return JSON.parse(element.textContent.trim());
            } catch (e) {
                console.error('Failed to parse groups stats:', e);
            }
        }
        return {};
    }
    
    loadGroupData(groupId) {
        if (!this.loadedGroups.has(groupId)) {
            try {
                // Clean group ID for element lookup
                const safeGroupId = groupId.replace(/[^a-zA-Z0-9_-]/g, '_');
                const element = document.getElementById(`group-${safeGroupId}`);
                
                if (!element) {
                    console.warn(`No compressed data found for group: ${groupId}`);
                    return [];
                }
                
                const compressedBase64 = element.textContent.trim();
                
                // Decompress using pako
                const compressed = Uint8Array.from(atob(compressedBase64), c => c.charCodeAt(0));
                const decompressed = pako.inflate(compressed, {to: 'string'});
                const quantizedData = JSON.parse(decompressed);
                
                // Dequantize data
                const dequantizedData = this._dequantizeGroup(quantizedData);
                this.loadedGroups.set(groupId, dequantizedData);
                
                console.log(`Loaded group ${groupId}: ${dequantizedData.length} sequences`);
                
            } catch (e) {
                console.error(`Failed to load group ${groupId}:`, e);
                return [];
            }
        }
        
        return this.loadedGroups.get(groupId) || [];
    }
    
    _dequantizeGroup(quantizedGroup) {
        return quantizedGroup.map(seq => {
            // Dequantize arrays
            const logprobs = seq.logprobs_q.map(x => x / 1000.0);
            const entropies = seq.entropies_q.map(x => x / 1000.0);
            const advantages = seq.advantages_q.map(x => x / 1000.0);
            
            // Pre-process token data with PRE-CALCULATED COLORS - major performance boost!
            const tokenDataReady = seq.tokens.map((token, index) => ({
                token: token,
                logprob: logprobs[index] || 0,
                entropy: entropies[index] || 0,
                advantage: advantages[index] || 0,
                assistantMask: seq.assistant_masks[index],
                // ✨ NEW: Pre-calculated HSL colors for instant mode switching
                lpColor: getLogprobColor(logprobs[index] || 0),
                enColor: getEntropyColor(entropies[index] || 0),
                advColor: getAdvantageColor(advantages[index] || 0)
            }));
            
            return {
                ...seq,
                logprobs,
                entropies, 
                advantages,
                tokenDataReady  // Pre-processed data with colors ready for immediate use
            };
        });
    }
    
    clearGroup(groupId) {
        if (this.loadedGroups.has(groupId)) {
            this.loadedGroups.delete(groupId);
            console.log(`Cleared cached data for group: ${groupId}`);
        }
    }
    
    getGroupStats(groupId) {
        return this.groupsStats[groupId] || null;
    }
    
    getAllGroupIds() {
        return Object.keys(this.groupsStats);
    }
}

/**
 * Modern Token Viewer
 * Simple, efficient token visualization with direct DOM rendering
 */
class TokenViewer {
    constructor(container, config = {}) {
        this.container = container;
        this.config = {
            groupId: null,
            sequenceId: null,
            ...config
        };
        
        // State
        this.dataManager = window.tokenDataManager || new TokenDataManager();
        this.currentGroup = this.config.groupId;
        this.currentSequence = this.config.sequenceId;
        this.colorMode = 'entropy';
        this.sequences = [];
        this.currentSequenceData = null;
        
        // UI elements
        this.elements = {};
        this.init();
    }
    
    init() {
        this.createUI();
        this.setupEventListeners();
        
        // Load data if group is specified
        if (this.currentGroup) {
            this.loadGroupData();
        }
    }
    
    createUI() {
        this.container.innerHTML = `
            <div class="token-viewer">
                <!-- Controls -->
                <div class="token-controls glass-card">
                    <div class="control-group">
                        <label>Sequence:</label>
                        <select id="sequenceSelect" class="app-select">
                            <option>Loading...</option>
                        </select>
                        <span id="sequenceInfo" class="sequence-info"></span>
                    </div>
                    
                    <div class="control-group">
                        <label>Color Mode:</label>
                        <div class="mode-buttons">
                            <button class="mode-btn active" data-mode="entropy">Entropy</button>
                            <button class="mode-btn" data-mode="logprob">LogProb</button>
                            <button class="mode-btn" data-mode="advantage">Advantage</button>
                            <button class="mode-btn" data-mode="none">None</button>
                        </div>
                    </div>
                </div>
                
                <!-- Token Display -->
                <div class="token-display glass-card">
                    <div id="tokenContent" class="token-content">
                        <div class="loading-state">
                            <div class="loading-spinner"></div>
                            <p>Loading tokens...</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Cache DOM elements
        this.elements = {
            sequenceSelect: this.container.querySelector('#sequenceSelect'),
            tokenContent: this.container.querySelector('#tokenContent'),
            sequenceInfo: this.container.querySelector('#sequenceInfo')
        };
    }
    
    loadGroupData() {
        if (!this.currentGroup) {
            console.warn('No group specified to load');
            return;
        }
        
        try {
            this.sequences = this.dataManager.loadGroupData(this.currentGroup);
            this.populateSequenceSelector();
            
            // Auto-select the specified sequence or first sequence
            if (this.currentSequence) {
                const sequenceIndex = this.sequences.findIndex(seq => seq.sequence_id === this.currentSequence);
                if (sequenceIndex >= 0) {
                    this.loadSequence(sequenceIndex);
                }
            } else if (this.sequences.length > 0) {
                this.loadSequence(0);
            }
            
        } catch (e) {
            console.error('Failed to load group data:', e);
            this.showError('Failed to load token data');
        }
    }
    
    populateSequenceSelector() {
        const select = this.elements.sequenceSelect;
        select.innerHTML = '';
        
        if (!this.sequences || this.sequences.length === 0) {
            select.innerHTML = '<option>No sequences available</option>';
            return;
        }
        
        this.sequences.forEach((seq, index) => {
            const option = document.createElement('option');
            option.value = index;
            option.textContent = seq.display_id || `Sequence ${index + 1}`;
            select.appendChild(option);
        });
    }
    
    loadSequence(sequenceIndex) {
        if (!this.sequences || sequenceIndex >= this.sequences.length) {
            console.warn('Invalid sequence index:', sequenceIndex);
            return;
        }
        
        const sequence = this.sequences[sequenceIndex];
        this.currentSequenceData = sequence;
        this.currentSequence = sequence.sequence_id;
        
        // Update sequence selector
        this.elements.sequenceSelect.value = sequenceIndex;
        
        // Update sequence info
        this.elements.sequenceInfo.textContent = `${sequence.length} tokens`;
        
        // Render tokens directly
        this.renderTokens(sequence);
        
        console.log('Loaded sequence:', sequence.display_id, 'with', sequence.length, 'tokens');
    }
    
    renderTokens(sequence) {
        const content = this.elements.tokenContent;
        
        if (!sequence.tokens || sequence.tokens.length === 0) {
            content.innerHTML = '<div class="empty-state">No tokens in this sequence</div>';
            return;
        }
        
        // Use pre-processed data - no more runtime map() calls!
        this.currentTokenData = sequence.tokenDataReady;
        
        // Simple one-time rendering - much faster on iPad Safari
        this.renderTokensSimple(content);
    }
    
    generateTokensHTML() {
        let html = '';
        this.currentTokenData.forEach((tokenData, index) => {
            html += this.generateTokenHTML(tokenData, index);
        });
        return html;
    }
    
    renderTokensSimple(content) {
        // Create container with initial color mode
        const tokenList = document.createElement('div');
        tokenList.className = 'token-list';
        tokenList.setAttribute('data-mode', this.colorMode);
        
        // Generate HTML string - simple and fast like old version
        const tokensHTML = this.currentTokenData.map((tokenData, index) => {
            const assistantAttr = tokenData.assistantMask ? 'true' : 'false';
            const tooltip = `LogProb: ${tokenData.logprob.toFixed(3)}\\nEntropy: ${tokenData.entropy.toFixed(3)}\\nAdvantage: ${tokenData.advantage.toFixed(3)}`;
            
            // Use CSS variables for colors but set them via style attribute for simplicity
            const cssVars = `--lp-color: ${tokenData.lpColor}; --en-color: ${tokenData.enColor}; --adv-color: ${tokenData.advColor};`;
            
            return `<span class="token" data-assistant="${assistantAttr}" style="${cssVars}" title="${tooltip}">${this.escapeHtml(tokenData.token)}</span>`;
        }).join('');
        
        // One-time innerHTML - let browser optimize this
        tokenList.innerHTML = tokensHTML;
        
        // Replace content
        content.innerHTML = '';
        content.appendChild(tokenList);
    }
    
    generateTokenHTML(tokenData, index) {
        const style = this.getTokenStyle(tokenData.logprob, tokenData.entropy, tokenData.advantage, tokenData.assistantMask);
        const escapedToken = this.escapeHtml(tokenData.token);
        const tooltip = `LogProb: ${tokenData.logprob.toFixed(3)}&#10;Entropy: ${tokenData.entropy.toFixed(3)}&#10;Advantage: ${tokenData.advantage.toFixed(3)}`;
        
        return `<span class="token" style="${style}" title="${tooltip}">${escapedToken}</span>`;
    }
    
    
    // ❌ REMOVED: getTokenStyle() method - no longer needed!
    // Colors are now pre-calculated and stored in CSS variables
    // Mode switching handled by CSS selectors for instant performance
    
    setColorMode(mode) {
        this.colorMode = mode;
        
        // Update active button
        this.container.querySelectorAll('.mode-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });
        
        // Simple CSS attribute change - instant mode switching
        const tokenContainer = this.elements.tokenContent.querySelector('.token-list');
        if (tokenContainer) {
            tokenContainer.setAttribute('data-mode', mode);
            console.log(`🚀 Instant color mode switch to: ${mode}`);
        }
    }
    
    setupEventListeners() {
        // Sequence selection
        this.elements.sequenceSelect.addEventListener('change', (e) => {
            this.loadSequence(parseInt(e.target.value));
        });
        
        // Color mode buttons
        this.container.querySelectorAll('.mode-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.setColorMode(e.target.dataset.mode);
            });
        });
    }
    
    showError(message) {
        this.elements.tokenContent.innerHTML = `
            <div class="error-state">
                <div class="error-icon">⚠️</div>
                <p class="error-text">${message}</p>
            </div>
        `;
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Export classes to global scope
window.TokenDataManager = TokenDataManager;
window.TokenViewer = TokenViewer;

// Backward compatibility alias
window.ChunkedTokenViewer = TokenViewer;