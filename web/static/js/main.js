/**
 * Main Application Entry Point
 */
import { API } from './api.js';
import { UI } from './ui.js';
import { Chat } from './chat.js';

let currentChatId = null;
let currentPersonaId = 'general_assistant';

document.addEventListener('DOMContentLoaded', async () => {
    // 1. Initial Load
    UI.init();
    loadHistory();
    loadPersonasForMenu();

    // Toggle Persona Menu
    const menuBtn = document.getElementById('personaMenuBtn');
    const menuEl = document.getElementById('personaFloatingMenu');
    if (menuBtn && menuEl) {
        menuBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const isVisible = menuEl.style.display === 'block';
            menuEl.style.display = isVisible ? 'none' : 'block';
        });
        
        document.addEventListener('click', () => {
            menuEl.style.display = 'none';
        });
    }

    // 2. Setup Event Listeners
    if (UI.elements.sendBtn) UI.elements.sendBtn.addEventListener('click', handleSend);
    if (UI.elements.queryInput) {
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
    }

    if (UI.elements.newChatBtn) {
        UI.elements.newChatBtn.addEventListener('click', () => {
            currentChatId = null;
            UI.clearChat();
            loadHistory(); // Refresh to clear active state
        });
    }

    // Citation Click Handler
    document.addEventListener('click', (e) => {
        const badge = e.target.closest('.citation-badge');
        if (badge) {
            const id = badge.getAttribute('data-id');
            scrollToCitation(id);
        }
    });
});

function scrollToCitation(id) {
    const targetElId = `source-card-${id}`;
    
    // Open right sidebar if collapsed
    if (UI.elements.sourcesPanel && UI.elements.sourcesPanel.classList.contains('collapsed')) {
        UI.elements.sourcesPanel.classList.remove('collapsed');
    }

    const tryHighlight = () => {
        const targetEl = document.getElementById(targetElId);
        if (targetEl) {
            targetEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
            targetEl.classList.remove('highlight-animation');
            // Trigger reflow to restart animation
            void targetEl.offsetWidth;
            targetEl.classList.add('highlight-animation');
            return true;
        }
        return false;
    };

    if (!tryHighlight()) {
        // Source card might not be rendered yet, use MutationObserver
        const observer = new MutationObserver((mutations, obs) => {
            if (tryHighlight()) {
                obs.disconnect();
            }
        });
        if (UI.elements.sourcesList) {
            observer.observe(UI.elements.sourcesList, { childList: true, subtree: true });
            // Timeout to prevent memory leak if source never appears
            setTimeout(() => observer.disconnect(), 5000);
        }
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

    try {
        const response = await API.ask({
            query,
            use_hyde: UI.elements.hydeToggle ? UI.elements.hydeToggle.checked : false,
            mode: isAgentic ? 'agentic' : 'classic',
            chat_id: currentChatId,
            persona_id: currentPersonaId
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

async function loadPersonasForMenu() {
    const menuEl = document.getElementById('personaFloatingMenu');
    if (!menuEl) return;
    
    try {
        const data = await API.fetchPersonas();
        const personas = data.personas || {};
        
        menuEl.innerHTML = '';
        
        Object.keys(personas).forEach(id => {
            const p = personas[id];
            const item = document.createElement('div');
            item.style.padding = '8px 12px';
            item.style.cursor = 'pointer';
            item.style.fontSize = '12px';
            item.style.color = 'var(--text-main)';
            item.style.borderRadius = '4px';
            item.style.display = 'flex';
            item.style.alignItems = 'center';
            item.style.gap = '8px';
            
            item.addEventListener('mouseenter', () => item.style.background = 'rgba(255,255,255,0.05)');
            item.addEventListener('mouseleave', () => item.style.background = 'transparent');
            
            item.innerHTML = `
                <i class="${p.meta.icon || 'fas fa-robot'}" style="color: ${p.meta.color || 'var(--blue-primary)'}; width: 14px;"></i>
                <span>${p.meta.label}</span>
            `;
            
            item.addEventListener('click', () => {
                currentPersonaId = id;
                const labelEl = document.getElementById('currentPersonaLabel');
                if (labelEl) labelEl.textContent = `บทบาท: ${p.meta.label}`;
                menuEl.style.display = 'none';
            });
            
            menuEl.appendChild(item);
        });
    } catch (err) {
        console.error("Failed to load personas for menu:", err);
    }
}
