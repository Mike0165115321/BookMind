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
                    try {
                        const data = JSON.parse(line.slice(5).trim());
                        this.processEvent(currentEvent, data, contentEl, thoughtEl, messageEl, isAgentic);

                        if (currentEvent === 'chat_id' && onChatId) {
                            onChatId(data.chat_id);
                        }

                        if (currentEvent === 'token') {
                            if (thoughtEl && thoughtEl.style.display !== 'none') {
                                thoughtEl.style.display = 'none';
                            }
                            fullText += data.text;
                            contentEl.innerHTML = marked.parse(fullText);
                            UI.scrollToBottom();
                        }
                    } catch (e) {
                        console.warn("SSE Parse Error:", e, line);
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
                this.renderTiming(data, messageEl, contentEl.innerText);
                break;
        }
    },

    renderTiming(data, messageEl, textToCopy) {
        const timingContainer = document.createElement('div');
        timingContainer.className = 'timing-container';
        timingContainer.style.display = 'flex';
        timingContainer.style.alignItems = 'center';
        timingContainer.style.gap = '8px';
        timingContainer.style.marginTop = '12px';
        timingContainer.style.width = '100%';

        const pillContainer = document.createElement('div');
        pillContainer.className = 'timing-pill-wrapper';
        pillContainer.style.display = 'flex';
        pillContainer.style.alignItems = 'center';
        pillContainer.style.gap = '12px';
        pillContainer.style.padding = '6px 16px';
        pillContainer.style.background = 'rgba(255, 255, 255, 0.05)';
        pillContainer.style.borderRadius = '20px';
        pillContainer.style.border = '1px solid rgba(255, 255, 255, 0.08)';

        // 1. Search Time Badge
        const searchTime = (data.stage_times?.search || data.search_time || 0);
        pillContainer.innerHTML += `
            <div class="timing-badge" style="display: flex; alignItems: center; gap: 6px;">
                <span style="opacity: 0.8;">🔍</span>
                <span style="font-family: var(--font-mono); font-size: 13px; color: #82aaff;">${parseFloat(searchTime).toFixed(2)}s</span>
            </div>
        `;

        // 2. HyDE Time
        const hydeTime = data.stage_times?.hyde || data.hyde_time || 0;
        if (hydeTime > 0) {
            pillContainer.innerHTML += `
                <div class="timing-badge" style="display: flex; alignItems: center; gap: 6px;" title="HyDE Query Transform Time">
                    <span style="opacity: 0.8;">🪄</span>
                    <span style="font-family: var(--font-mono); font-size: 13px; color: var(--orange);">${parseFloat(hydeTime).toFixed(2)}s</span>
                </div>
            `;
        }

        // 3. Agentic Time (Decompose + Evaluate)
        const agenticTime = (data.stage_times?.decompose || 0) + (data.stage_times?.evaluate || 0);
        if (agenticTime > 0) {
            pillContainer.innerHTML += `
                <div class="timing-badge" style="display: flex; alignItems: center; gap: 6px;" title="Agentic Overhead (Decompose & Evaluate)">
                    <span style="opacity: 0.8;">🧠</span>
                    <span style="font-family: var(--font-mono); font-size: 13px; color: #ff7eb6;">${parseFloat(agenticTime).toFixed(2)}s</span>
                </div>
            `;
        }

        // 4. AI Generation/Synthesis Time
        const aiTime = (data.stage_times?.synthesize || 0);
        if (aiTime > 0) {
            pillContainer.innerHTML += `
                <div class="timing-badge" style="display: flex; alignItems: center; gap: 6px;">
                    <span style="opacity: 0.8;">🤖</span>
                    <span style="font-family: var(--font-mono); font-size: 13px; color: var(--text-main);">${parseFloat(aiTime).toFixed(1)}s</span>
                </div>
            `;
        } else if (data.provider) {
             pillContainer.innerHTML += `
                <div class="timing-badge" style="display: flex; alignItems: center; gap: 6px;">
                    <span style="opacity: 0.8;">🤖</span>
                    <span style="font-family: var(--font-mono); font-size: 13px; color: var(--text-main);">${data.provider.toUpperCase()}</span>
                </div>
            `;
        }

        // 3. Total Time Badge
        const totalTime = data.total_time || 0;
        pillContainer.innerHTML += `
            <div class="timing-badge" style="display: flex; alignItems: center; gap: 6px;">
                <span style="opacity: 0.8;">⏱️</span>
                <span style="font-family: var(--font-mono); font-size: 13px; color: #ececec; font-weight: 500;">${parseFloat(totalTime).toFixed(1)}s</span>
            </div>
        `;

        // 4. Action Buttons (Copy)
        const actionsEl = document.createElement('div');
        actionsEl.className = 'message-actions';
        actionsEl.style.marginLeft = 'auto';
        
        const copyBtn = document.createElement('button');
        copyBtn.className = 'action-btn';
        copyBtn.title = 'Copy response';
        copyBtn.innerHTML = '<i class="far fa-copy"></i>';
        copyBtn.onclick = () => {
            navigator.clipboard.writeText(textToCopy);
            copyBtn.innerHTML = '<i class="fas fa-check" style="color: var(--green);"></i>';
            setTimeout(() => { copyBtn.innerHTML = '<i class="far fa-copy"></i>'; }, 2000);
        };
        
        actionsEl.appendChild(copyBtn);

        // Assemble everything
        timingContainer.appendChild(pillContainer);
        timingContainer.appendChild(actionsEl);
        
        messageEl.appendChild(timingContainer);
        UI.scrollToBottom();
    }
};
