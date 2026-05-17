/**
 * UI Module - Handles DOM manipulation and rendering
 */
export const UI = {
    elements: {
        chatMessages: document.getElementById('chatMessages'),
        queryInput: document.getElementById('queryInput'),
        sendBtn: document.getElementById('sendBtn'),
        modelSelector: document.getElementById('modelSelector'),
        hydeToggle: document.getElementById('hydeToggle'),
        agenticToggle: document.getElementById('agenticToggle'),
        sidebar: document.getElementById('sidebar'),
        sidebarToggle: document.getElementById('sidebarToggle'),
        sourcesPanel: document.getElementById('sourcesPanel'),
        sourcesToggle: document.getElementById('sourcesToggle'),
        welcomeHero: document.getElementById('welcomeHero'),
        sourcesList: document.getElementById('sourcesList'),
        historyList: document.getElementById('historyList'),
        newChatBtn: document.getElementById('newChatBtn'),
        railNewChatBtn: document.getElementById('railNewChatBtn'),
        settingsBtn: document.getElementById('settingsBtn'),
        railSettingsBtn: document.getElementById('railSettingsBtn'),
        settingsModal: document.getElementById('settingsModal'),
        closeSettings: document.getElementById('closeSettings'),
        saveSettings: document.getElementById('saveSettings'),
        hydeModelSelect: document.getElementById('hydeModelSelect'),
        genModelSelect: document.getElementById('genModelSelect'),
        webSearchBtn: document.getElementById('webSearchBtn')
    },

    init() {
        if (this.elements.sidebarToggle) {
            this.elements.sidebarToggle.addEventListener('click', () => {
                this.elements.sidebar.classList.toggle('collapsed');
            });
        }
        
        // Link rail buttons to existing logic
        if (this.elements.railNewChatBtn && this.elements.newChatBtn) {
            this.elements.railNewChatBtn.addEventListener('click', () => {
                this.elements.newChatBtn.click();
            });
        }
        
        if (this.elements.railSettingsBtn) {
            this.elements.railSettingsBtn.addEventListener('click', () => {
                this.showSettings();
            });
        }

        if (this.elements.sourcesToggle) {
            this.elements.sourcesToggle.addEventListener('click', () => {
                this.elements.sourcesPanel.classList.toggle('collapsed');
            });
        }
        if (this.elements.closeSettings) {
            this.elements.closeSettings.addEventListener('click', () => this.hideSettings());
        }
        
        if (this.elements.webSearchBtn) {
            this.elements.webSearchBtn.addEventListener('click', () => {
                this.elements.webSearchBtn.classList.toggle('active');
            });
        }
    },

    isWebSearchActive() {
        return this.elements.webSearchBtn && this.elements.webSearchBtn.classList.contains('active');
    },

    showSettings() {
        this.elements.settingsModal.classList.add('active');
    },

    hideSettings() {
        this.elements.settingsModal.classList.remove('active');
    },

    renderSettingsOptions(data) {
        const hydeSelect = this.elements.hydeModelSelect;
        const genSelect = this.elements.genModelSelect;
        [hydeSelect, genSelect].forEach(sel => {
            sel.innerHTML = '';
            for (const [provider, models] of Object.entries(data)) {
                const group = document.createElement('optgroup');
                group.label = provider.toUpperCase();
                models.forEach(model => {
                    const opt = document.createElement('option');
                    opt.value = `${provider}:${model}`;
                    opt.textContent = `${this.getProviderEmoji(provider)} ${model}`;
                    group.appendChild(opt);
                });
                sel.appendChild(group);
            }
        });
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

    renderHistory(chats, activeChatId, onSelect, onDelete) {
        this.elements.historyList.innerHTML = '';
        if (!chats || chats.length === 0) {
            this.elements.historyList.innerHTML = '<div style="padding: 20px; color: var(--text-muted); font-size: 13px; text-align: center;">ไม่มีประวัติการแชท</div>';
            return;
        }

        chats.forEach(chat => {
            const item = document.createElement('div');
            item.className = `history-item ${chat.id === activeChatId ? 'active' : ''}`;
            item.innerHTML = `
                <i class="far fa-comment"></i>
                <span class="history-title">${this.escapeHtml(chat.title)}</span>
                <button class="delete-chat-btn" title="Delete Chat">
                    <i class="fas fa-trash-alt"></i>
                </button>
            `;
            
            item.addEventListener('click', (e) => {
                if (e.target.closest('.delete-chat-btn')) {
                    e.stopPropagation();
                    onDelete(chat.id);
                } else {
                    onSelect(chat.id);
                }
            });
            
            this.elements.historyList.appendChild(item);
        });
    },

    clearChat() {
        this.elements.chatMessages.innerHTML = `
            <div class="welcome-hero" id="welcomeHero">
                <h2 style="font-size: 32px; font-weight: 700; margin-bottom: 12px; color: var(--text-muted);">BookMind</h2>
                <p style="color: var(--text-muted); opacity: 0.6;">วันนี้มีอะไรให้ช่วยไหมครับ?</p>
            </div>
        `;
        this.elements.welcomeHero = document.getElementById('welcomeHero');
        this.elements.sourcesList.innerHTML = '';
    },

    escapeHtml(text) {
        const el = document.createElement('div');
        el.textContent = text;
        return el.innerHTML;
    },

    addUserMessage(text, fileInfo = null) {
        if (this.elements.welcomeHero) this.elements.welcomeHero.style.display = 'none';
        
        const el = document.createElement('div');
        el.className = 'message message-user';
        
        let contentHtml = this.escapeHtml(text);
        
        if (fileInfo && fileInfo.name) {
            let fileHtml = `<div class="file-attachment-badge"><i class="fas fa-paperclip"></i> ${this.escapeHtml(fileInfo.name)}</div>`;
            
            if (fileInfo.type && fileInfo.type.startsWith('image/') && fileInfo.dataUrl) {
                fileHtml += `<div class="file-preview-image"><img src="${fileInfo.dataUrl}" alt="${this.escapeHtml(fileInfo.name)}" /></div>`;
            }
            
            contentHtml = fileHtml + contentHtml;
        }
        
        el.innerHTML = `<div class="message-content">${contentHtml}</div>`;
        this.elements.chatMessages.appendChild(el);
        this.scrollToBottom();
    },

    addAIMessage(isAgentic = false) {
        const messageEl = document.createElement('div');
        messageEl.className = 'message message-ai';
        
        const body = document.createElement('div');
        body.style.cssText = 'flex: 1; min-width: 0;';
        
        const thoughtEl = document.createElement('div');
        thoughtEl.className = 'thought-trace';
        thoughtEl.style.display = 'none';
        thoughtEl.innerHTML = `<i class="far fa-lightbulb"></i> <span class="thought-text">Thinking...</span>`;

        const contentEl = document.createElement('div');
        contentEl.className = 'message-content';

        body.appendChild(thoughtEl);
        if (isAgentic) {
            const stepsEl = document.createElement('div');
            stepsEl.className = 'agentic-steps';
            body.appendChild(stepsEl);
        }
        body.appendChild(contentEl);
        messageEl.appendChild(body);
        
        this.elements.chatMessages.appendChild(messageEl);
        this.scrollToBottom();

        return { messageEl: body, contentEl, thoughtEl };
    },

    renderSources(sources) {
        this.elements.sourcesList.innerHTML = '';
        if (!sources || sources.length === 0) return;

        // Auto-open sources panel if closed and we have sources
        if (this.elements.sourcesPanel.classList.contains('collapsed')) {
            this.elements.sourcesPanel.classList.remove('collapsed');
        }

        sources.forEach((src) => {
            const card = document.createElement('div');
            card.className = 'source-card';
            card.id = `source-card-${src.rank || (sources.indexOf(src) + 1)}`;
            card.innerHTML = `
                <div class="source-title">📖 [${src.rank || (sources.indexOf(src) + 1)}] ${this.escapeHtml(src.title)}</div>
                <div class="source-text">${this.escapeHtml(src.text)}</div>
            `;
            this.elements.sourcesList.appendChild(card);
        });
    }
};
