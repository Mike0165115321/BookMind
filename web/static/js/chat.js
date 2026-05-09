/**
 * Chat Module - Handles streaming logic and event processing
 */
import { UI } from './ui.js';

export const Chat = {
    async handleStream(reader, contentEl, thoughtEl, messageEl, isAgentic, onChatId) {
        const decoder = new TextDecoder();
        let fullText = '';
        let buffer = '';
        let currentEvent = null;

        // Show thinking status initially
        if (thoughtEl) {
            thoughtEl.style.display = 'flex';
            thoughtEl.querySelector('.thought-text').textContent = 'Thinking...';
        }

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (line.startsWith('event:')) {
                    currentEvent = line.slice(6).trim();
                } else if (line.startsWith('data:') && currentEvent) {
                    const data = JSON.parse(line.slice(5).trim());
                    this.processEvent(currentEvent, data, contentEl, thoughtEl, messageEl, isAgentic);

                    if (currentEvent === 'chat_id' && onChatId) {
                        onChatId(data.chat_id);
                    }

                    if (currentEvent === 'token') {
                        // Hide thinking status on first token
                        if (thoughtEl && thoughtEl.style.display !== 'none') {
                            thoughtEl.style.display = 'none';
                        }
                        fullText += data.text;
                        contentEl.innerHTML = marked.parse(fullText);
                        UI.scrollToBottom();
                    }
                    currentEvent = null;
                }
            }
        }
    },

    processEvent(event, data, contentEl, thoughtEl, messageEl, isAgentic) {
        switch (event) {
            case 'status':
                if (thoughtEl) {
                    thoughtEl.style.display = 'flex';
                    thoughtEl.querySelector('.thought-text').textContent = data.message;
                }
                break;
            case 'sources':
                UI.renderSources(data.sources);
                break;
            case 'done':
                if (thoughtEl) thoughtEl.remove();
                this.renderTiming(data, messageEl);
                break;
        }
    },

    renderTiming(data, messageEl) {
        const timingEl = document.createElement('div');
        timingEl.className = 'timing-bar';
        let parts = [];
        if (data.provider) {
            parts.push(`<span class="metadata-badge">${UI.getProviderEmoji(data.provider)} ${data.provider.toUpperCase()}</span>`);
        }
        
        // Use a safer check for total_time
        const time = data.total_time || data.total_duration || 0;
        parts.push(`<span>⏱️ ${parseFloat(time).toFixed(1)}s</span>`);
        
        timingEl.innerHTML = parts.join('');
        messageEl.appendChild(timingEl);
        UI.scrollToBottom();
    }
};
