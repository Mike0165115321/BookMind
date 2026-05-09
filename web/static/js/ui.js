/**
 * UI Module - Handles DOM manipulation and rendering
 */
export const UI = {
    elements: {
        chatMessages: document.getElementById('chatMessages'),
        queryInput: document.getElementById('queryInput'),
        sendBtn: document.getElementById('sendBtn'),
        sourcesList: document.getElementById('sourcesList'),
        modelSelector: document.getElementById('modelSelector'),
        hydeToggle: document.getElementById('hydeToggle'),
        agenticToggle: document.getElementById('agenticToggle')
    },

    getProviderEmoji(provider) {
        const emojis = {
            'gemini': '🤖',
            'groq': '⚡',
            'ollama': '🏠',
            'openrouter': '🌉'
        };
        return emojis[provider] || '🧠';
    },

    scrollToBottom() {
        this.elements.chatMessages.scrollTop = this.elements.chatMessages.scrollHeight;
    },

    escapeHtml(text) {
        const el = document.createElement('div');
        el.textContent = text;
        return el.innerHTML;
    },

    addUserMessage(text) {
        const el = document.createElement('div');
        el.className = 'message message-user';
        el.innerHTML = `<div class="message-content">${this.escapeHtml(text)}</div>`;
        this.elements.chatMessages.appendChild(el);
        this.scrollToBottom();
    },

    addAIMessage(isAgentic = false) {
        const messageEl = document.createElement('div');
        messageEl.className = 'message message-ai';
        
        const avatar = `<div class="ai-avatar">🤖</div>`;
        const body = document.createElement('div');
        body.style.cssText = 'flex: 1; min-width: 0;';
        
        const statusEl = document.createElement('div');
        statusEl.className = 'status-bar';
        statusEl.innerHTML = '<div class="status-dot"></div><span class="status-text">⏳ กำลังวิเคราะห์...</span>';

        const contentEl = document.createElement('div');
        contentEl.className = 'message-content markdown-body';

        body.appendChild(statusEl);
        if (isAgentic) {
            const stepsEl = document.createElement('div');
            stepsEl.className = 'agentic-steps';
            body.appendChild(stepsEl);
        }
        body.appendChild(contentEl);
        
        messageEl.innerHTML = avatar;
        messageEl.appendChild(body);
        
        this.elements.chatMessages.appendChild(messageEl);
        this.scrollToBottom();

        return { messageEl: body, contentEl, statusEl };
    },

    renderSources(sources) {
        this.elements.sourcesList.innerHTML = '';
        sources.forEach((src) => {
            const card = document.createElement('div');
            card.className = 'source-card';
            card.innerHTML = `
                <div class="source-title">📖 ${this.escapeHtml(src.title)}</div>
                <div class="source-text">${this.escapeHtml(src.text)}</div>
            `;
            this.elements.sourcesList.appendChild(card);
        });
    }
};
