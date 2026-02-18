/**
 * RAG Knowledge Base — Frontend Application
 *
 * Handles:
 *   - SSE streaming from /api/ask
 *   - Real-time markdown rendering
 *   - Source panel updates
 *   - Chat history management
 */

// ──────────────────────────────────────────────
// DOM Elements
// ──────────────────────────────────────────────
const chatMessages = document.getElementById('chatMessages');
const queryInput = document.getElementById('queryInput');
const sendBtn = document.getElementById('sendBtn');
const sourcesList = document.getElementById('sourcesList');
const hydeToggle = document.getElementById('hydeToggle');

let isGenerating = false;

// ──────────────────────────────────────────────
// Configure marked.js
// ──────────────────────────────────────────────
marked.setOptions({
    breaks: true,
    gfm: true,
});

// ──────────────────────────────────────────────
// Auto-resize textarea
// ──────────────────────────────────────────────
queryInput.addEventListener('input', () => {
    queryInput.style.height = 'auto';
    queryInput.style.height = Math.min(queryInput.scrollHeight, 120) + 'px';
});

queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// ──────────────────────────────────────────────
// Suggestion chips
// ──────────────────────────────────────────────
function askSuggestion(btn) {
    queryInput.value = btn.textContent;
    sendMessage();
}

// ──────────────────────────────────────────────
// Send message & handle SSE stream
// ──────────────────────────────────────────────
async function sendMessage() {
    const query = queryInput.value.trim();
    if (!query || isGenerating) return;

    isGenerating = true;
    sendBtn.disabled = true;

    // Remove welcome card
    const welcome = document.querySelector('.welcome-card');
    if (welcome) welcome.remove();

    // Add user bubble
    addUserMessage(query);

    // Clear input
    queryInput.value = '';
    queryInput.style.height = 'auto';

    // Add AI response area
    const { messageEl, contentEl, statusEl } = addAIMessage();

    // Start SSE
    try {
        const response = await fetch('/api/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                use_hyde: hydeToggle.checked,
            }),
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullText = '';
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // Parse SSE events from buffer
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep incomplete line in buffer

            for (const line of lines) {
                if (line.startsWith('event:')) {
                    var currentEvent = line.slice(6).trim();
                } else if (line.startsWith('data:') && currentEvent) {
                    const data = JSON.parse(line.slice(5).trim());
                    handleEvent(currentEvent, data, contentEl, statusEl, messageEl);

                    if (currentEvent === 'token') {
                        fullText += data.text;
                        contentEl.innerHTML = marked.parse(fullText);
                        scrollToBottom();
                    }

                    currentEvent = null;
                }
            }
        }

    } catch (err) {
        contentEl.innerHTML = `<p style="color: var(--orange);">❌ เกิดข้อผิดพลาด: ${err.message}</p>`;
    }

    isGenerating = false;
    sendBtn.disabled = false;
    queryInput.focus();
}

// ──────────────────────────────────────────────
// Handle SSE events
// ──────────────────────────────────────────────
function handleEvent(event, data, contentEl, statusEl, messageEl) {
    switch (event) {
        case 'status':
            statusEl.querySelector('.status-text').textContent = data.message;
            break;

        case 'hyde':
            statusEl.querySelector('.status-text').textContent =
                `🪄 HyDE สำเร็จ (${data.time}s)`;
            break;

        case 'sources':
            renderSources(data.sources, data.search_time);
            statusEl.querySelector('.status-text').textContent =
                `🔍 พบ ${data.sources.length} แหล่ง (${data.search_time}s)`;
            break;

        case 'done':
            // Remove status bar
            statusEl.remove();

            // Add timing bar
            const timingEl = document.createElement('div');
            timingEl.className = 'timing-bar';
            let timingParts = [];
            if (data.hyde_time > 0) timingParts.push(`<span>🪄 ${data.hyde_time}s</span>`);
            timingParts.push(`<span>🔍 ${data.search_time}s</span>`);
            timingParts.push(`<span>🤖 ${data.gen_time}s</span>`);
            timingParts.push(`<span>⏱️ ${data.total_time}s</span>`);
            timingEl.innerHTML = timingParts.join('');
            messageEl.appendChild(timingEl);
            scrollToBottom();
            break;
    }
}

// ──────────────────────────────────────────────
// Render sources in side panel
// ──────────────────────────────────────────────
function renderSources(sources, searchTime) {
    sourcesList.innerHTML = '';

    sources.forEach((src) => {
        const scoreClass = src.score >= 0.5 ? 'high' : src.score >= 0.25 ? 'medium' : 'low';

        const card = document.createElement('div');
        card.className = 'source-card';
        card.innerHTML = `
            <div class="source-card-header">
                <span class="source-rank">${src.rank}</span>
                <span class="source-score ${scoreClass}">${(src.score * 100).toFixed(0)}%</span>
            </div>
            <div class="source-title">📖 ${escapeHtml(src.title)}</div>
            <div class="source-text">${escapeHtml(src.text)}</div>
        `;
        sourcesList.appendChild(card);
    });
}

// ──────────────────────────────────────────────
// DOM Helpers
// ──────────────────────────────────────────────
function addUserMessage(text) {
    const el = document.createElement('div');
    el.className = 'message message-user';
    el.innerHTML = `<div class="message-content">${escapeHtml(text)}</div>`;
    chatMessages.appendChild(el);
    scrollToBottom();
}

function addAIMessage() {
    const messageEl = document.createElement('div');
    messageEl.className = 'message message-ai';

    const statusEl = document.createElement('div');
    statusEl.className = 'status-bar';
    statusEl.innerHTML = '<div class="status-dot"></div><span class="status-text">⏳ กำลังประมวลผล...</span>';

    const contentEl = document.createElement('div');
    contentEl.className = 'message-content';

    messageEl.innerHTML = '<div class="ai-avatar">🤖</div>';
    const body = document.createElement('div');
    body.style.cssText = 'flex: 1; min-width: 0;';
    body.appendChild(statusEl);
    body.appendChild(contentEl);
    messageEl.appendChild(body);

    chatMessages.appendChild(messageEl);
    scrollToBottom();

    return { messageEl: body, contentEl, statusEl };
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function escapeHtml(text) {
    const el = document.createElement('div');
    el.textContent = text;
    return el.innerHTML;
}
