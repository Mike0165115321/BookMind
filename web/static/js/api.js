/**
 * API Module - Handles all communication with the backend
 */
export const API = {
    async fetchModels() {
        try {
            const res = await fetch('/api/llm-models');
            return await res.json();
        } catch (err) {
            console.error("API Error: Failed to fetch models", err);
            return {};
        }
    },

    async ask({ query, use_hyde, mode, provider, model, chat_id }) {
        const response = await fetch('/api/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, use_hyde, mode, provider, model, chat_id })
        });
        if (!response.ok) throw new Error('API request failed');
        return response;
    },

    async fetchChats() {
        const response = await fetch('/api/chats');
        return await response.json();
    },

    async fetchMessages(chatId) {
        const response = await fetch(`/api/chats/${chatId}/messages`);
        return await response.json();
    },

    async deleteChat(chatId) {
        const response = await fetch(`/api/chats/${chatId}`, { method: 'DELETE' });
        return await response.json();
    },

    async fetchSettings() {
        const response = await fetch('/api/settings');
        return await response.json();
    },

    async saveSettings(settings) {
        const response = await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(settings)
        });
        return await response.json();
    }
};
