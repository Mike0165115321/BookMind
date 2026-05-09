/**
 * Main Application Entry Point
 */
import { API } from './api.js';
import { UI } from './ui.js';
import { Chat } from './chat.js';

document.addEventListener('DOMContentLoaded', async () => {
    // 1. Initial Load
    const models = await API.fetchModels();
    renderModelOptions(models);

    // 2. Setup Event Listeners
    UI.elements.sendBtn.addEventListener('click', handleSend);
    UI.elements.queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });

    // Auto-resize textarea
    UI.elements.queryInput.addEventListener('input', () => {
        UI.elements.queryInput.style.height = 'auto';
        UI.elements.queryInput.style.height = Math.min(UI.elements.queryInput.scrollHeight, 150) + 'px';
    });
});

async function handleSend() {
    const query = UI.elements.queryInput.value.trim();
    if (!query) return;

    // UI Feedback
    UI.addUserMessage(query);
    UI.elements.queryInput.value = '';
    UI.elements.queryInput.style.height = 'auto';
    
    const isAgentic = UI.elements.agenticToggle.checked;
    const { contentEl, statusEl, messageEl } = UI.addAIMessage(isAgentic);

    // Get selected model
    const [provider, model] = UI.elements.modelSelector.value.split(':');

    try {
        const response = await API.ask({
            query,
            use_hyde: UI.elements.hydeToggle.checked,
            mode: isAgentic ? 'agentic' : 'classic',
            provider,
            model
        });

        await Chat.handleStream(response.body.getReader(), contentEl, statusEl, messageEl, isAgentic);
    } catch (err) {
        contentEl.innerHTML = `<p style="color: var(--orange);">❌ Error: ${err.message}</p>`;
    }
}

function renderModelOptions(data) {
    const selector = UI.elements.modelSelector;
    if (!selector) return;

    selector.innerHTML = '';
    for (const [provider, models] of Object.entries(data)) {
        const group = document.createElement('optgroup');
        group.label = provider.toUpperCase();
        
        models.forEach(model => {
            const opt = document.createElement('option');
            opt.value = `${provider}:${model}`;
            opt.textContent = `${UI.getProviderEmoji(provider)} ${model}`;
            group.appendChild(opt);
        });
        selector.appendChild(group);
    }
}
