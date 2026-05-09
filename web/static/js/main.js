/**
 * Main Application Entry Point
 */
import { API } from './api.js';
import { UI } from './ui.js';
import { Chat } from './chat.js';

let currentChatId = null;

document.addEventListener('DOMContentLoaded', async () => {
    // 1. Initial Load
    UI.init();
    loadModels();
    loadHistory();

    // 2. Setup Event Listeners
    UI.elements.sendBtn.addEventListener('click', handleSend);
    UI.elements.queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });

    UI.elements.newChatBtn.addEventListener('click', () => {
        currentChatId = null;
        UI.clearChat();
        loadHistory(); // Refresh to clear active state
    });

    UI.elements.settingsBtn.addEventListener('click', async () => {
        const models = await API.fetchModels();
        UI.renderSettingsOptions(models);
        
        const settings = await API.fetchSettings();
        if (settings.hyde_model) UI.elements.hydeModelSelect.value = settings.hyde_model;
        if (settings.gen_model) UI.elements.genModelSelect.value = settings.gen_model;
        
        UI.showSettings();
    });

    UI.elements.saveSettings.addEventListener('click', async () => {
        const settings = {
            hyde_model: UI.elements.hydeModelSelect.value,
            gen_model: UI.elements.genModelSelect.value
        };
        await API.saveSettings(settings);
        UI.hideSettings();
        alert("บันทึกการตั้งค่าเรียบร้อยแล้ว");
        // Update inline selector to match gen_model
        UI.elements.modelSelector.value = settings.gen_model;
    });

    // Auto-resize textarea
    UI.elements.queryInput.addEventListener('input', () => {
        UI.elements.queryInput.style.height = 'auto';
        UI.elements.queryInput.style.height = Math.min(UI.elements.queryInput.scrollHeight, 150) + 'px';
    });
});

async function loadModels() {
    try {
        const [models, settings] = await Promise.all([
            API.fetchModels(),
            API.fetchSettings()
        ]);
        renderModelOptions(models, settings.gen_provider);
    } catch (err) {
        console.error("Failed to load models:", err);
    }
}

async function loadHistory() {
    try {
        const chats = await API.fetchChats();
        UI.renderHistory(
            chats, 
            currentChatId, 
            (id) => switchChat(id),
            (id) => deleteChat(id)
        );
    } catch (err) {
        console.error("Failed to load history:", err);
    }
}

async function switchChat(chatId) {
    if (currentChatId === chatId) return;
    currentChatId = chatId;
    
    UI.clearChat();
    loadHistory(); // Update active state in sidebar

    try {
        const messages = await API.fetchMessages(chatId);
        messages.forEach(msg => {
            if (msg.role === 'user') {
                UI.addUserMessage(msg.content);
            } else {
                // For AI messages, we might have metadata in the DB
                const meta = msg.metadata ? JSON.parse(msg.metadata) : null;
                const { contentEl, thoughtEl, messageEl } = UI.addAIMessage(false);
                if (thoughtEl) thoughtEl.remove();
                contentEl.innerHTML = marked.parse(msg.content);
                if (meta) Chat.renderTiming(meta, messageEl);
            }
        });
    } catch (err) {
        console.error("Failed to load messages:", err);
    }
}

async function deleteChat(chatId) {
    if (!confirm("คุณแน่ใจหรือไม่ว่าต้องการลบแชทนี้?")) return;
    try {
        await API.deleteChat(chatId);
        if (currentChatId === chatId) {
            currentChatId = null;
            UI.clearChat();
        }
        loadHistory();
    } catch (err) {
        alert("ลบแชทไม่สำเร็จ");
    }
}

async function handleSend() {
    const query = UI.elements.queryInput.value.trim();
    if (!query) return;

    // UI Feedback
    UI.addUserMessage(query);
    UI.elements.queryInput.value = '';
    UI.elements.queryInput.style.height = 'auto';
    
    const isAgentic = UI.elements.agenticToggle ? UI.elements.agenticToggle.checked : false;
    const { contentEl, thoughtEl, messageEl } = UI.addAIMessage(isAgentic);

    // Get selected model
    const [provider, model] = UI.elements.modelSelector.value.split(':');

    try {
        const response = await API.ask({
            query,
            use_hyde: UI.elements.hydeToggle ? UI.elements.hydeToggle.checked : false,
            mode: isAgentic ? 'agentic' : 'classic',
            provider,
            model,
            chat_id: currentChatId
        });

        await Chat.handleStream(
            response.body.getReader(), 
            contentEl, 
            thoughtEl, 
            messageEl, 
            isAgentic,
            (newId) => {
                if (!currentChatId) {
                    currentChatId = newId;
                    loadHistory();
                }
            }
        );
        
        // Final history refresh to update titles if needed
        loadHistory();
        
    } catch (err) {
        contentEl.innerHTML = `<p style="color: var(--orange);">❌ Error: ${err.message}</p>`;
    }
}

function renderModelOptions(data, targetProvider = null) {
    const selector = UI.elements.modelSelector;
    if (!selector) return;

    selector.innerHTML = '';
    const sortedProviders = Object.keys(data).sort();

    for (const provider of sortedProviders) {
        if (targetProvider && provider !== targetProvider) continue; // Filter by provider

        const models = data[provider];
        if (!models || models.length === 0) continue;

        const group = document.createElement('optgroup');
        group.label = provider.toUpperCase();
        
        models.forEach(model => {
            const opt = document.createElement('option');
            opt.value = `${provider}:${model}`;
            opt.textContent = `${UI.getProviderEmoji(provider)} ${model}`; // Cleaner look
            group.appendChild(opt);
        });
        selector.appendChild(group);
    }
}
