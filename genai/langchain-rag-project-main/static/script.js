// RAG Chatbot JavaScript
class RAGChatbot {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8000';
        this.conversationId = this.generateConversationId();
        this.isProcessing = false;
        this.showPerformanceMetrics = false; // Toggle for showing performance data
        
        this.initializeElements();
        this.setupEventListeners();
        this.checkApiStatus();
        this.autoResizeTextarea();
    }

    generateConversationId() {
        return 'web-' + Math.random().toString(36).substring(2, 15);
    }

    initializeElements() {
        this.elements = {
            chatMessages: document.getElementById('chat-messages'),
            userInput: document.getElementById('user-input'),
            sendButton: document.getElementById('send-button'),
            statusIndicator: document.getElementById('status-indicator'),
            statusText: document.getElementById('status-text'),
            typingIndicator: document.getElementById('typing-indicator'),
            loadingOverlay: document.getElementById('loading-overlay'),
            errorToast: document.getElementById('error-toast'),
            errorMessage: document.getElementById('error-message'),
            charCount: document.getElementById('char-count'),
            performanceToggle: document.getElementById('performance-toggle')
        };
    }

    setupEventListeners() {
        // Send button click
        this.elements.sendButton.addEventListener('click', () => this.handleSendMessage());
        
        // Enter key press (with Shift+Enter for new lines)
        this.elements.userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.handleSendMessage();
            }
        });

        // Input changes
        this.elements.userInput.addEventListener('input', () => {
            this.updateCharCount();
            this.updateSendButtonState();
            this.autoResizeTextarea();
        });

        // Performance toggle
        this.elements.performanceToggle.addEventListener('click', () => this.togglePerformanceMetrics());

        // Auto-focus input
        this.elements.userInput.focus();
    }

    updateCharCount() {
        const length = this.elements.userInput.value.length;
        this.elements.charCount.textContent = `${length}/2000`;
        
        if (length > 1800) {
            this.elements.charCount.style.color = 'var(--error-color)';
        } else if (length > 1500) {
            this.elements.charCount.style.color = 'var(--warning-color)';
        } else {
            this.elements.charCount.style.color = 'var(--text-muted)';
        }
    }

    updateSendButtonState() {
        const hasText = this.elements.userInput.value.trim().length > 0;
        this.elements.sendButton.disabled = !hasText || this.isProcessing;
    }

    autoResizeTextarea() {
        const textarea = this.elements.userInput;
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    }

    async checkApiStatus() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            if (response.ok) {
                const data = await response.json();
                this.updateStatus('online', `Online (${data.documents_count} docs)`);
            } else {
                throw new Error('API not responding');
            }
        } catch (error) {
            this.updateStatus('offline', 'Offline');
            this.showError('Unable to connect to the RAG service. Please ensure the FastAPI server is running.');
        }
    }

    updateStatus(status, text) {
        this.elements.statusIndicator.className = `status-indicator ${status}`;
        this.elements.statusText.textContent = text;
    }

    async handleSendMessage() {
        const message = this.elements.userInput.value.trim();
        if (!message || this.isProcessing) return;

        this.isProcessing = true;
        this.updateSendButtonState();

        // Add user message to chat
        this.addMessage(message, 'user');
        
        // Clear input
        this.elements.userInput.value = '';
        this.updateCharCount();
        this.autoResizeTextarea();

        // Show typing indicator
        this.showTypingIndicator();

        // Track API response time
        const apiStartTime = Date.now();

        try {
            const response = await this.sendToAPI(message);
            const apiResponseTime = (Date.now() - apiStartTime) / 1000; // Convert to seconds
            
            this.hideTypingIndicator();
            
            if (response.success) {
                // Add API response time to the response object
                response.api_response_time_seconds = apiResponseTime;
                this.addAssistantMessage(response);
            } else {
                throw new Error(response.error || 'Failed to get response');
            }
        } catch (error) {
            this.hideTypingIndicator();
            this.showError(`Error: ${error.message}`);
            this.addMessage('Sorry, I encountered an error while processing your question. Please try again.', 'assistant');
        } finally {
            this.isProcessing = false;
            this.updateSendButtonState();
            this.elements.userInput.focus();
        }
    }

    async sendToAPI(message) {
        const response = await fetch(`${this.apiBaseUrl}/api/v1/enhanced/ultimate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: message,
                domain_context: "LangChain",
                num_hypothetical: 3,
                semantic_weight: 0.7,
                keyword_weight: 0.3,
                enable_reranking: true,
                top_k: 8,
                use_memory: true,
                conversation_id: this.conversationId,
                use_cache: true
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
    }

    addMessage(text, sender, timestamp = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;

        const time = timestamp || new Date().toLocaleTimeString([], { 
            hour: '2-digit', 
            minute: '2-digit' 
        });

        messageDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-${sender === 'user' ? 'user' : 'robot'}"></i>
            </div>
            <div class="message-content">
                <div class="message-text">${this.formatMessage(text)}</div>
                <div class="message-time">${time}</div>
            </div>
        `;

        this.elements.chatMessages.appendChild(messageDiv);
        
        // Apply syntax highlighting to any code blocks
        this.applySyntaxHighlighting(messageDiv);
        
        this.scrollToBottom();
    }

    addAssistantMessage(response) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant-message enhanced-response';

        const time = new Date().toLocaleTimeString([], { 
            hour: '2-digit', 
            minute: '2-digit' 
        });

        // Enhanced features indicator
        let enhancementHtml = '';
        if (response.enhancement_details) {
            const details = response.enhancement_details;
            const features = [];
            if (details.hyde_applied) features.push('🧠 HyDE Enhanced');
            if (details.hybrid_search_applied) features.push('🔍 Hybrid Search');
            if (details.reranking_applied) features.push('📊 Re-ranked');
            
            if (features.length > 0) {
                enhancementHtml = `
                    <div class="enhancement-indicators">
                        <div class="enhancement-title">⚡ Enhanced Processing:</div>
                        <div class="enhancement-badges">
                            ${features.map(feature => `<span class="enhancement-badge">${feature}</span>`).join('')}
                        </div>
                    </div>
                `;
            }
        }

        // Sources from Ultimate RAG response with URLs and context
        let sourcesHtml = '';
        if (response.source_documents && response.source_documents.length > 0) {
            sourcesHtml = `
                <div class="message-sources">
                    <div class="sources-title">📚 Sources & Retrieved Context</div>
                    ${response.source_documents.map((doc, index) => {
                        const metadata = doc.metadata || {};
                        const fileName = metadata.file_name || 'Document';
                        const docType = metadata.doc_type || '';
                        const url = metadata.url || null;
                        const content = doc.page_content || doc.content || '';
                        const score = (doc.score || 0).toFixed(3);
                        const sourceId = `source-${index}-${Date.now()}`;
                        
                        return `
                            <div class="source-item">
                                <div class="source-header">
                                    <div class="source-info">
                                        <i class="fas fa-file-text"></i>
                                        <span class="source-filename">${fileName} ${docType ? `(${docType})` : ''}</span>
                                        <span class="source-score">Score: ${score}</span>
                                    </div>
                                    ${url ? `<a href="${url}" target="_blank" class="source-url" title="Open source document">
                                        <i class="fas fa-external-link-alt"></i>
                                        View Source
                                    </a>` : ''}
                                </div>
                                <div class="source-content">
                                    <div class="content-label">
                                        <i class="fas fa-quote-left"></i>
                                        Retrieved Context:
                                        <button class="toggle-content" onclick="toggleSourceContent('${sourceId}')">
                                            <span class="toggle-text">Show More</span>
                                            <i class="fas fa-chevron-down"></i>
                                        </button>
                                    </div>
                                    <div id="${sourceId}" class="source-context collapsed">
                                        ${this.formatSourceContent(content)}
                                    </div>
                                </div>
                            </div>
                        `;
                    }).join('')}
                </div>
            `;
        }

        // Keep existing follow-up questions handling
        let followupHtml = '';
        if (response.follow_up_questions && response.follow_up_questions.length > 0) {
            followupHtml = `
                <div class="followup-questions">
                    <div class="followup-title">💡 Related Questions</div>
                    ${response.follow_up_questions.map(question => `
                        <div class="followup-question" onclick="sendQuickQuery('${this.escapeHtml(question)}')">
                            ${question}
                        </div>
                    `).join('')}
                </div>
            `;
        }

        // Performance metrics - always show API response time
        let performanceHtml = '';
        if (response.performance || response.api_response_time_seconds) {
            const perf = response.performance || {};
            const showDetailed = window.ragChatbot.showPerformanceMetrics;
            
            performanceHtml = `
                <div class="performance-metrics">
                    <div class="performance-title">⏱️ Performance</div>
                    <div class="performance-stats">
                        ${response.api_response_time_seconds ? `<span class="api-response-time">API Response: ${response.api_response_time_seconds.toFixed(2)}s</span>` : ''}
                        ${showDetailed && perf.total_time_seconds ? `<span>Server Total: ${(perf.total_time_seconds || 0).toFixed(2)}s</span>` : ''}
                        ${showDetailed && perf.hyde_enhancement_time_seconds ? `<span>HyDE: ${perf.hyde_enhancement_time_seconds.toFixed(2)}s</span>` : ''}
                        ${showDetailed && perf.hybrid_search_time_seconds ? `<span>Search: ${perf.hybrid_search_time_seconds.toFixed(2)}s</span>` : ''}
                        ${showDetailed && perf.response_generation_time_seconds ? `<span>Generation: ${perf.response_generation_time_seconds.toFixed(2)}s</span>` : ''}
                        ${showDetailed && perf.documents_retrieved ? `<span>Documents: ${perf.documents_retrieved}</span>` : ''}
                    </div>
                </div>
            `;
        }

        messageDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="message-content">
                <div class="message-text">
                    ${this.formatMessage(response.answer)}
                    ${enhancementHtml}
                    ${sourcesHtml}
                    ${followupHtml}
                    ${performanceHtml}
                </div>
                <div class="message-time">${time}</div>
            </div>
        `;

        this.elements.chatMessages.appendChild(messageDiv);
        
        // Apply syntax highlighting to any code blocks
        this.applySyntaxHighlighting(messageDiv);
        
        this.scrollToBottom();
    }

    formatMessage(text) {
        // Use Marked.js for proper markdown parsing
        // Configure marked with custom renderer for code blocks
        const renderer = new marked.Renderer();
        
        // Custom code block renderer to add copy functionality
        let codeBlockId = 0;
        renderer.code = (code, language) => {
            const blockId = `code-block-${++codeBlockId}`;
            const lang = language || 'text';
            const languageIcon = this.getLanguageIcon(lang);
            const displayName = lang.charAt(0).toUpperCase() + lang.slice(1);
            
            return `
                <div class="code-block-container">
                    <div class="code-block-header">
                        <div class="code-language">
                            <i class="${languageIcon}"></i>
                            ${displayName}
                        </div>
                        <button class="copy-code-btn" onclick="window.ragChatbot.copyCodeToClipboard('${blockId}')">
                            <i class="fas fa-copy"></i>
                            Copy Code
                        </button>
                    </div>
                    <div class="code-block-content">
                        <pre><code id="${blockId}" class="language-${lang}">${this.escapeHtml(code)}</code></pre>
                    </div>
                </div>
            `;
        };
        
        // Configure marked options
        marked.setOptions({
            renderer: renderer,
            breaks: true, // Convert line breaks to <br>
            gfm: true,    // GitHub Flavored Markdown
            sanitize: false // We handle escaping ourselves
        });
        
        // Parse markdown to HTML
        return marked.parse(text);
    }
    
    getLanguageIcon(language) {
        const icons = {
            'python': 'fab fa-python',
            'javascript': 'fab fa-js-square',
            'js': 'fab fa-js-square',
            'bash': 'fas fa-terminal',
            'shell': 'fas fa-terminal',
            'json': 'fas fa-code',
            'html': 'fab fa-html5',
            'css': 'fab fa-css3-alt',
            'sql': 'fas fa-database',
            'text': 'fas fa-file-alt',
            'plaintext': 'fas fa-file-alt'
        };
        return icons[language] || 'fas fa-code';
    }
    
    async copyCodeToClipboard(blockId) {
        const codeElement = document.getElementById(blockId);
        const copyBtn = codeElement.closest('.code-block-container').querySelector('.copy-code-btn');
        
        if (!codeElement) return;
        
        const code = codeElement.textContent;
        
        try {
            await navigator.clipboard.writeText(code);
            
            // Visual feedback
            const originalContent = copyBtn.innerHTML;
            copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
            copyBtn.classList.add('copied');
            
            setTimeout(() => {
                copyBtn.innerHTML = originalContent;
                copyBtn.classList.remove('copied');
            }, 2000);
            
        } catch (err) {
            console.error('Failed to copy code: ', err);
            
            // Fallback for older browsers
            this.fallbackCopyTextToClipboard(code, copyBtn);
        }
    }
    
    fallbackCopyTextToClipboard(text, button) {
        const textArea = document.createElement("textarea");
        textArea.value = text;
        textArea.style.top = "0";
        textArea.style.left = "0";
        textArea.style.position = "fixed";
        
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        
        try {
            document.execCommand('copy');
            
            const originalContent = button.innerHTML;
            button.innerHTML = '<i class="fas fa-check"></i> Copied!';
            button.classList.add('copied');
            
            setTimeout(() => {
                button.innerHTML = originalContent;
                button.classList.remove('copied');
            }, 2000);
        } catch (err) {
            console.error('Fallback copy failed: ', err);
        }
        
        document.body.removeChild(textArea);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    formatSourceContent(content) {
        if (!content) return '<em>No content available</em>';
        
        // Truncate very long content for initial display
        const maxLength = 300;
        let displayContent = content;
        
        if (content.length > maxLength) {
            displayContent = content.substring(0, maxLength) + '...';
        }
        
        // Escape HTML and preserve line breaks
        return this.escapeHtml(displayContent).replace(/\n/g, '<br>');
    }

    showTypingIndicator() {
        this.elements.typingIndicator.classList.remove('hidden');
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        this.elements.typingIndicator.classList.add('hidden');
    }

    scrollToBottom() {
        setTimeout(() => {
            this.elements.chatMessages.scrollTop = this.elements.chatMessages.scrollHeight;
        }, 100);
    }

    showError(message) {
        this.elements.errorMessage.textContent = message;
        this.elements.errorToast.classList.remove('hidden');
        
        // Auto-hide after 5 seconds
        setTimeout(() => this.hideError(), 5000);
    }

    hideError() {
        this.elements.errorToast.classList.add('hidden');
    }

    clearChat() {
        // Keep only the welcome message
        const messages = this.elements.chatMessages.querySelectorAll('.message');
        messages.forEach((message, index) => {
            if (index > 0) { // Keep first message (welcome)
                message.remove();
            }
        });
        
        // Generate new conversation ID
        this.conversationId = this.generateConversationId();
        
        // Focus input
        this.elements.userInput.focus();
    }

    togglePerformanceMetrics() {
        this.showPerformanceMetrics = !this.showPerformanceMetrics;
        
        // Update button appearance
        if (this.showPerformanceMetrics) {
            this.elements.performanceToggle.classList.add('active');
            this.elements.performanceToggle.title = 'Hide detailed performance metrics';
        } else {
            this.elements.performanceToggle.classList.remove('active');
            this.elements.performanceToggle.title = 'Show detailed performance metrics';
        }

        // Update existing messages to show/hide detailed metrics
        const existingMessages = this.elements.chatMessages.querySelectorAll('.assistant-message .performance-metrics');
        existingMessages.forEach(metricsDiv => {
            this.refreshPerformanceDisplay(metricsDiv);
        });
    }

    refreshPerformanceDisplay(metricsDiv) {
        // This could be implemented to refresh existing performance displays
        // For now, we'll rely on new messages showing the updated format
    }
    
    applySyntaxHighlighting(container) {
        // Apply syntax highlighting to code blocks using Prism.js
        const codeBlocks = container.querySelectorAll('pre code[class*="language-"]');
        codeBlocks.forEach((block) => {
            if (typeof Prism !== 'undefined') {
                Prism.highlightElement(block);
            }
        });
    }
}

// Global functions for quick actions
function sendQuickQuery(query) {
    const chatbot = window.ragChatbot;
    chatbot.elements.userInput.value = query;
    chatbot.updateCharCount();
    chatbot.updateSendButtonState();
    chatbot.handleSendMessage();
}

function clearChat() {
    window.ragChatbot.clearChat();
}

function hideError() {
    window.ragChatbot.hideError();
}

function toggleSourceContent(sourceId) {
    const contentDiv = document.getElementById(sourceId);
    const toggleBtn = contentDiv.previousElementSibling.querySelector('.toggle-content');
    const toggleText = toggleBtn.querySelector('.toggle-text');
    const toggleIcon = toggleBtn.querySelector('.fas');
    
    if (contentDiv.classList.contains('collapsed')) {
        contentDiv.classList.remove('collapsed');
        contentDiv.classList.add('expanded');
        toggleText.textContent = 'Show Less';
        toggleIcon.className = 'fas fa-chevron-up';
    } else {
        contentDiv.classList.remove('expanded');
        contentDiv.classList.add('collapsed');
        toggleText.textContent = 'Show More';
        toggleIcon.className = 'fas fa-chevron-down';
    }
}

// Initialize chatbot when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.ragChatbot = new RAGChatbot();
    
    // Periodic status check
    setInterval(() => {
        window.ragChatbot.checkApiStatus();
    }, 30000); // Check every 30 seconds
});