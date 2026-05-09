/**
 * Chat Module - Handles streaming logic and event processing
 */
import { UI } from './ui.js';

export const Chat = {
    async handleStream(reader, contentEl, statusEl, messageEl, isAgentic) {
        const decoder = new TextDecoder();
        let fullText = '';
        let buffer = '';
        let currentEvent = null;

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
                    this.processEvent(currentEvent, data, contentEl, statusEl, messageEl, isAgentic);

                    if (currentEvent === 'token') {
                        fullText += data.text;
                        contentEl.innerHTML = marked.parse(fullText);
                        UI.scrollToBottom();
                    }
                    currentEvent = null;
                }
            }
        }
    },

    processEvent(event, data, contentEl, statusEl, messageEl, isAgentic) {
        // Handle various event types (status, sources, done, etc.)
        // This is a simplified version, we can expand it for agentic steps
        switch (event) {
            case 'status':
                statusEl.querySelector('.status-text').textContent = data.message;
                break;
            case 'sources':
                UI.renderSources(data.sources);
                break;
            case 'done':
                statusEl.remove();
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
        parts.push(`<span>⏱️ ${data.total_time}s</span>`);
        timingEl.innerHTML = parts.join('');
        messageEl.appendChild(timingEl);
        UI.scrollToBottom();
    }
};
