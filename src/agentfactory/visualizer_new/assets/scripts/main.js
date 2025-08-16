/**
 * Main JavaScript for AgentFactory VisualizerNew
 * Handles navigation, state management, and UI interactions
 */

class AgentFactoryApp {
    constructor(appData) {
        this.appData = appData;
        this.state = {
            currentGroup: null,
            currentSample: null,
            showSystemPrompt: {},  // Track per result
            viewType: appData.view_type || 'conversation'
        };
        
        this.init();
    }
    
    init() {
        // Initialize first group and sample
        const groupNames = Object.keys(this.appData.groups);
        if (groupNames.length > 0) {
            this.state.currentGroup = groupNames[0];
            const firstGroup = this.appData.groups[this.state.currentGroup];
            if (firstGroup.samples.length > 0) {
                this.state.currentSample = firstGroup.samples[0].sample_id;
            }
        }
        
        this.setupEventListeners();
        this.updateUI();
    }
    
    setupEventListeners() {
        // Group navigation
        const groupSelect = document.getElementById('groupSelect');
        if (groupSelect) {
            groupSelect.addEventListener('change', (e) => {
                this.setCurrentGroup(e.target.value);
            });
        }
        
        const groupPrevBtn = document.getElementById('groupPrevBtn');
        const groupNextBtn = document.getElementById('groupNextBtn');
        if (groupPrevBtn) {
            groupPrevBtn.addEventListener('click', () => {
                this.navigateGroup('prev');
            });
        }
        if (groupNextBtn) {
            groupNextBtn.addEventListener('click', () => {
                this.navigateGroup('next');
            });
        }
        
        // Sample navigation
        const sampleSelect = document.getElementById('sampleSelect');
        if (sampleSelect) {
            sampleSelect.addEventListener('change', (e) => {
                this.setCurrentSample(e.target.value);
            });
        }
        
        const samplePrevBtn = document.getElementById('samplePrevBtn');
        const sampleNextBtn = document.getElementById('sampleNextBtn');
        if (samplePrevBtn) {
            samplePrevBtn.addEventListener('click', () => {
                this.navigateSample('prev');
            });
        }
        if (sampleNextBtn) {
            sampleNextBtn.addEventListener('click', () => {
                this.navigateSample('next');
            });
        }
        
        // System prompt toggles (handled dynamically)
        document.addEventListener('click', (e) => {
            if (e.target.matches('.system-prompt-header') || e.target.closest('.system-prompt-header')) {
                const header = e.target.matches('.system-prompt-header') ? 
                              e.target : e.target.closest('.system-prompt-header');
                this.toggleSystemPrompt(header);
            }
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            this.handleKeydown(e);
        });
    }
    
    setCurrentGroup(groupId) {
        if (groupId in this.appData.groups) {
            const previousGroup = this.state.currentGroup;
            this.state.currentGroup = groupId;
            
            // Clear previous group's compressed data if it's a token view
            if (this.state.viewType === 'token' && previousGroup && window.tokenDataManager) {
                window.tokenDataManager.clearGroup(previousGroup);
            }
            
            // Auto-select first sample in new group
            const group = this.appData.groups[groupId];
            if (group.samples.length > 0) {
                this.state.currentSample = group.samples[0].sample_id;
            } else {
                this.state.currentSample = null;
            }
            
            this.updateUI();
        }
    }
    
    setCurrentSample(sampleId) {
        const currentGroup = this.appData.groups[this.state.currentGroup];
        if (currentGroup && currentGroup.samples.find(s => s.sample_id === sampleId)) {
            this.state.currentSample = sampleId;
            this.updateSampleSelector(); // Update sample indicator
            this.updateContent();
        }
    }
    
    navigateGroup(direction) {
        const groupNames = Object.keys(this.appData.groups);
        const currentIndex = groupNames.indexOf(this.state.currentGroup);
        
        let newIndex;
        if (direction === 'next') {
            newIndex = currentIndex < groupNames.length - 1 ? currentIndex + 1 : 0;
        } else {
            newIndex = currentIndex > 0 ? currentIndex - 1 : groupNames.length - 1;
        }
        
        this.setCurrentGroup(groupNames[newIndex]);
    }
    
    navigateSample(direction) {
        const currentGroup = this.appData.groups[this.state.currentGroup];
        if (!currentGroup || currentGroup.samples.length === 0) return;
        
        const currentIndex = currentGroup.samples.findIndex(s => s.sample_id === this.state.currentSample);
        
        let newIndex;
        if (direction === 'next') {
            newIndex = currentIndex < currentGroup.samples.length - 1 ? currentIndex + 1 : 0;
        } else {
            newIndex = currentIndex > 0 ? currentIndex - 1 : currentGroup.samples.length - 1;
        }
        
        this.setCurrentSample(currentGroup.samples[newIndex].sample_id);
    }
    
    toggleSystemPrompt(headerElement) {
        const contentId = headerElement.getAttribute('data-content-id') || 'systemPromptContent';
        const content = document.getElementById(contentId);
        const icon = headerElement.querySelector('.system-prompt-icon');
        
        if (content && icon) {
            const isVisible = !content.classList.contains('hidden');
            
            if (isVisible) {
                content.classList.add('hidden');
                icon.classList.remove('expanded');
                this.state.showSystemPrompt[contentId] = false;
            } else {
                content.classList.remove('hidden');
                icon.classList.add('expanded');
                this.state.showSystemPrompt[contentId] = true;
            }
        }
    }
    
    updateUI() {
        this.updateNavigation();
        this.updateStats();
        this.updateSampleSelector();
        this.updateContent();
    }
    
    updateNavigation() {
        const groupNames = Object.keys(this.appData.groups);
        const currentIndex = groupNames.indexOf(this.state.currentGroup);
        
        // Update group selector
        const groupSelect = document.getElementById('groupSelect');
        if (groupSelect) {
            groupSelect.value = this.state.currentGroup;
        }
        
        // Update group indicator
        const groupIndicator = document.getElementById('groupIndicator');
        if (groupIndicator) {
            groupIndicator.textContent = `${currentIndex + 1} of ${groupNames.length}`;
        }
        
        // Update navigation buttons
        const groupPrevBtn = document.getElementById('groupPrevBtn');
        const groupNextBtn = document.getElementById('groupNextBtn');
        if (groupPrevBtn) groupPrevBtn.disabled = groupNames.length <= 1;
        if (groupNextBtn) groupNextBtn.disabled = groupNames.length <= 1;
    }
    
    updateStats() {
        const currentGroup = this.appData.groups[this.state.currentGroup];
        if (!currentGroup) return;
        
        const stats = currentGroup.stats;
        
        // Update stats display
        const updateStatElement = (id, value) => {
            const element = document.getElementById(id);
            if (element) element.textContent = value;
        };
        
        updateStatElement('successRate', stats.success_rate);
        updateStatElement('avgReward', stats.avg_reward.toFixed(3));
        updateStatElement('totalResults', stats.total_results);
        updateStatElement('executionTime', stats.execution_time || 'N/A');
    }
    
    updateSampleSelector() {
        const currentGroup = this.appData.groups[this.state.currentGroup];
        if (!currentGroup) return;
        
        const sampleSelect = document.getElementById('sampleSelect');
        if (sampleSelect) {
            // Clear existing options
            sampleSelect.innerHTML = '';
            
            // Add new options
            currentGroup.samples.forEach(sample => {
                const option = document.createElement('option');
                option.value = sample.sample_id;
                option.textContent = `${sample.name} - Reward: ${sample.weighted_reward.toFixed(3)}`;
                sampleSelect.appendChild(option);
            });
            
            sampleSelect.value = this.state.currentSample;
        }
        
        // Update sample indicator
        const currentIndex = currentGroup.samples.findIndex(s => s.sample_id === this.state.currentSample);
        const sampleIndicator = document.getElementById('sampleIndicator');
        if (sampleIndicator) {
            sampleIndicator.textContent = currentIndex >= 0 ? 
                `${currentIndex + 1} of ${currentGroup.samples.length}` : '';
        }
        
        // Update navigation buttons
        const samplePrevBtn = document.getElementById('samplePrevBtn');
        const sampleNextBtn = document.getElementById('sampleNextBtn');
        if (samplePrevBtn) samplePrevBtn.disabled = currentGroup.samples.length <= 1;
        if (sampleNextBtn) sampleNextBtn.disabled = currentGroup.samples.length <= 1;
    }
    
    updateContent() {
        const currentGroup = this.appData.groups[this.state.currentGroup];
        const currentSample = currentGroup ? 
            currentGroup.samples.find(s => s.sample_id === this.state.currentSample) : null;
        
        if (!currentSample) {
            this.showEmptyState();
            return;
        }
        
        // Update reward scores
        this.updateRewardScores(currentSample.rewards);
        
        // Update main content
        const contentContainer = document.getElementById('mainContent');
        if (contentContainer) {
            if (this.state.viewType === 'conversation') {
                this.renderConversationContent(contentContainer, currentSample);
            } else if (this.state.viewType === 'token') {
                this.renderTokenContent(contentContainer, currentSample);
            }
        }
    }
    
    updateRewardScores(rewards) {
        const rewardGrid = document.getElementById('rewardScoresGrid');
        if (!rewardGrid || !rewards) return;
        
        rewardGrid.innerHTML = '';
        
        Object.entries(rewards).forEach(([key, value]) => {
            const rewardItem = document.createElement('div');
            rewardItem.className = 'reward-item animate-scale-in';
            rewardItem.innerHTML = `
                <div class="reward-item-label">${this.escapeHtml(key)}</div>
                <div class="reward-item-value">${value.toFixed(3)}</div>
            `;
            rewardGrid.appendChild(rewardItem);
        });
    }
    
    renderConversationContent(container, sample) {
        let html = '';
        
        // System prompt
        if (sample.system_prompt) {
            const contentId = `systemPrompt_${sample.sample_id}`;
            const isExpanded = this.state.showSystemPrompt[contentId] || false;
            
            html += `
                <div class="system-prompt animate-fade-in-up">
                    <button class="system-prompt-header" data-content-id="${contentId}">
                        <span class="system-prompt-icon ${isExpanded ? 'expanded' : ''}">▶</span>
                        🔧 System Prompt
                    </button>
                    <div class="system-prompt-content ${isExpanded ? '' : 'hidden'}" id="${contentId}">
                        <pre class="system-prompt-text">${this.escapeHtml(sample.system_prompt)}</pre>
                    </div>
                </div>
            `;
        }
        
        // Conversation
        if (sample.conversation && sample.conversation.length > 0) {
            html += '<div class="conversation animate-fade-in-up">';
            html += '<h3 class="conversation-title">💬 Conversation</h3>';
            
            // Filter out system messages since they're shown separately above
            const nonSystemMessages = sample.conversation.filter(msg => msg.role.toLowerCase() !== 'system');
            
            nonSystemMessages.forEach((message, index) => {
                html += this.renderMessage(message, index);
            });
            
            html += '</div>';
        }
        
        // Metadata
        if (sample.metadata) {
            html += this.renderMetadata(sample.metadata);
        }
        
        container.innerHTML = html;
    }
    
    renderMessage(message, index) {
        const roleClass = message.role.toLowerCase();
        let html = `
            <div class="message animate-fade-in-up" style="animation-delay: ${index * 0.1}s">
                <div class="message-header">
                    <span class="message-role-badge ${roleClass}">
                        ${message.role.toUpperCase()}
                    </span>
                    <span class="message-timestamp">${message.timestamp}</span>
                </div>
                <div class="message-content ${roleClass}">
        `;
        
        // Special handling for tool messages
        if (roleClass === 'tool') {
            // Tool messages might have tool_name
            if (message.tool_name) {
                html += `<div class="message-text"><strong>🔧 ${this.escapeHtml(message.tool_name)}()</strong><br><div style="margin-top: 0.5em;">${this.escapeHtml(message.content)}</div></div>`;
            } else {
                html += `<div class="message-text">${this.escapeHtml(message.content)}</div>`;
            }
            
            // Add structured output if present
            if (message.structured_output) {
                const outputJson = this.formatCompactJson(message.structured_output);
                html += `<div class="tool-output">${this.escapeHtml(outputJson)}</div>`;
            }
        } else {
            html += `<div class="message-text">${this.escapeHtml(message.content)}</div>`;
        }
        
        // Add turn metrics if present
        if (message.rewards) {
            html += `
                <div class="turn-metrics">
                    <div class="turn-metrics-title">📊 Turn Metrics</div>
                    <div class="turn-metrics-grid">
            `;
            
            Object.entries(message.rewards).forEach(([key, value]) => {
                html += `
                    <div class="turn-metric-item">
                        <div class="turn-metric-label">${this.escapeHtml(key)}</div>
                        <div class="turn-metric-value">${value.toFixed(3)}</div>
                    </div>
                `;
            });
            
            html += '</div></div>';
        }
        
        html += '</div></div>';
        return html;
    }
    
    renderMetadata(metadata) {
        const currentGroup = this.appData.groups[this.state.currentGroup];
        const currentSample = currentGroup ? 
            currentGroup.samples.find(s => s.sample_id === this.state.currentSample) : null;
            
        if (!currentSample) return '';
        
        let html = `
            <div class="metadata-section animate-fade-in-up">
                <h3 class="metadata-title">
                    <span>📊</span>
                    <span>Execution Statistics</span>
                </h3>
                <div class="metadata-grid">
        `;
        
        // Token usage card
        if (metadata && (metadata.inputTokens !== undefined || metadata.totalTokens !== undefined)) {
            html += `
                <div class="metadata-card">
                    <h4 class="metadata-card-title">
                        <span>🪙</span>
                        <span>Token Usage</span>
                    </h4>
                    <ul class="metadata-list">
            `;
            
            if (metadata.inputTokens !== undefined) {
                html += `
                        <li class="metadata-item">
                            <span class="metadata-item-label">input_tokens</span>
                            <span class="metadata-item-value">${metadata.inputTokens.toLocaleString()}</span>
                        </li>
                `;
            }
            
            if (metadata.outputTokens !== undefined) {
                html += `
                        <li class="metadata-item">
                            <span class="metadata-item-label">output_tokens</span>
                            <span class="metadata-item-value">${metadata.outputTokens.toLocaleString()}</span>
                        </li>
                `;
            }
            
            if (metadata.totalTokens !== undefined) {
                html += `
                        <li class="metadata-item">
                            <span class="metadata-item-label">total_tokens</span>
                            <span class="metadata-item-value">${metadata.totalTokens.toLocaleString()}</span>
                        </li>
                `;
            }
            
            html += `
                    </ul>
                </div>
            `;
        }
        
        // Conversation stats card
        const conversationStats = this.getConversationStats(currentSample.conversation);
        html += `
            <div class="metadata-card">
                <h4 class="metadata-card-title">
                    <span>💬</span>
                    <span>Conversation Stats</span>
                </h4>
                <ul class="metadata-list">
                    <li class="metadata-item">
                        <span class="metadata-item-label">user_messages</span>
                        <span class="metadata-item-value">${conversationStats.user}</span>
                    </li>
                    <li class="metadata-item">
                        <span class="metadata-item-label">ai_responses</span>
                        <span class="metadata-item-value">${conversationStats.assistant}</span>
                    </li>
                    <li class="metadata-item">
                        <span class="metadata-item-label">tool_calls</span>
                        <span class="metadata-item-value">${conversationStats.tool}</span>
                    </li>
                    ${metadata && metadata.executionTime ? `
                    <li class="metadata-item">
                        <span class="metadata-item-label">execution_time</span>
                        <span class="metadata-item-value">${metadata.executionTime}</span>
                    </li>
                    ` : ''}
                    ${metadata && metadata.endReason ? `
                    <li class="metadata-item">
                        <span class="metadata-item-label">end_reason</span>
                        <span class="metadata-item-value">${metadata.endReason}</span>
                    </li>
                    ` : ''}
                </ul>
            </div>
        `;
        
        // Tool Call Details card
        if (conversationStats.toolCalls.length > 0) {
            html += `
                <div class="metadata-card">
                    <h4 class="metadata-card-title">
                        <span>🛠️</span>
                        <span>Tool Call Details</span>
                    </h4>
                    <div class="tool-calls-list">
            `;
            
            conversationStats.toolCalls.forEach(toolCall => {
                const argsDisplay = toolCall.arguments ? 
                    `<div class="tool-args">${this.escapeHtml(JSON.stringify(toolCall.arguments))}</div>` : '';
                html += `
                    <div class="tool-call-item">
                        <div class="tool-name">🔧 ${this.escapeHtml(toolCall.name)}()</div>
                        ${argsDisplay}
                    </div>
                `;
            });
            
            html += `
                    </div>
                </div>
            `;
        }
        
        html += '</div></div>';
        
        return html;
    }
    
    renderTokenContent(container, sample) {
        if (!sample.token_data) {
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">🪙</div>
                    <p class="empty-state-text">No token data available for this sample</p>
                </div>
            `;
            return;
        }
        
        // Initialize global token data manager if not exists
        if (!window.tokenDataManager && window.TokenDataManager) {
            window.tokenDataManager = new window.TokenDataManager();
        }
        
        // Check if this is compressed token data
        if (sample.token_data.compressed_group) {
            this.renderCompressedTokenContent(container, sample);
        } else {
            // Legacy token data format
            this.renderLegacyTokenContent(container, sample);
        }
    }
    
    renderCompressedTokenContent(container, sample) {
        const compressedGroup = sample.token_data.compressed_group;
        const sequenceInfo = sample.token_data.sequence_info;
        
        // Create token viewer container
        container.innerHTML = `
            <div id="tokenViewerContainer" class="token-viewer-container animate-fade-in-up">
                <div class="token-controls glass-card">
                    <div class="control-group">
                        <label>Sequence:</label>
                        <select id="sequenceSelect" class="app-select">
                            <!-- Populated by JavaScript -->
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
        
        // Initialize token viewer with compressed data
        if (window.TokenViewer && window.tokenDataManager) {
            console.log('Loading compressed token data for group:', compressedGroup.group_id);
            
            this.tokenViewer = new window.TokenViewer(
                container.querySelector('#tokenViewerContainer'), 
                {
                    groupId: compressedGroup.group_id,
                    sequenceId: sequenceInfo.sequence_id
                }
            );
        } else {
            console.warn('TokenViewer or TokenDataManager not available');
            this.renderTokenContentFallback(container, sample);
        }
    }
    
    renderLegacyTokenContent(container, sample) {
        // Create token viewer container
        container.innerHTML = `
            <div id="tokenViewerContainer" class="token-viewer-container animate-fade-in-up">
                <!-- Legacy token viewer will be initialized here -->
            </div>
        `;
        
        // Legacy token data - not supported in new simplified version
        console.warn('Legacy token data format detected, using fallback rendering');
        this.renderTokenContentFallback(container, sample);
    }
    
    renderTokenContentFallback(container, sample) {
        // Simple fallback renderer for token data
        let html = `
            <div class="token-content-fallback animate-fade-in-up">
                <h3 class="token-title">🪙 Token Analysis</h3>
                <div class="token-info">
                    <p>Advanced token viewer is loading...</p>
                    <p>Token data format: ${sample.token_data.format_version || 'legacy'}</p>
        `;
        
        if (sample.token_data.sequences) {
            html += `<p>Sequences: ${sample.token_data.sequences.length}</p>`;
        } else if (sample.token_data.tokens) {
            html += `<p>Tokens: ${sample.token_data.tokens.length}</p>`;
        }
        
        html += `
                </div>
            </div>
        `;
        
        container.innerHTML = html;
    }
    
    showEmptyState() {
        const container = document.getElementById('mainContent');
        if (container) {
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">📊</div>
                    <p class="empty-state-text">Select a generation to view details</p>
                </div>
            `;
        }
    }
    
    handleKeydown(e) {
        // Navigation shortcuts
        if (e.ctrlKey || e.metaKey) {
            switch (e.key) {
                case 'ArrowLeft':
                    e.preventDefault();
                    this.navigateGroup('prev');
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    this.navigateGroup('next');
                    break;
                case 'ArrowUp':
                    e.preventDefault();
                    this.navigateSample('prev');
                    break;
                case 'ArrowDown':
                    e.preventDefault();
                    this.navigateSample('next');
                    break;
            }
        }
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    formatCompactJson(obj, indent = 2, maxLen = 8) {
        /**
         * Compact JSON serialization - single line for structured_output
         */
        // For structured output, use compact single-line format
        return JSON.stringify(obj);
    }
    
    getConversationStats(conversation) {
        const stats = { user: 0, assistant: 0, tool: 0, toolCalls: [] };
        
        if (!conversation) return stats;
        
        conversation.forEach(message => {
            if (message.role in stats) {
                stats[message.role]++;
            }
            
            if (message.role === 'tool' && message.tool_name) {
                stats.toolCalls.push({
                    name: message.tool_name,
                    arguments: message.tool_arguments || null
                });
            }
        });
        
        return stats;
    }
    
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // App data will be injected by the Python backend
    if (typeof window.appData !== 'undefined') {
        window.agentFactoryApp = new AgentFactoryApp(window.appData);
    }
});