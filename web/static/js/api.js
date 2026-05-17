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

    async fetchPersonas() {
        try {
            const res = await fetch('/api/personas');
            return await res.json();
        } catch (err) {
            console.error("API Error: Failed to fetch personas", err);
            return {};
        }
    },

    async createPersona(data) {
        const response = await fetch('/api/personas', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        if (!response.ok) throw new Error('API request failed');
        return await response.json();
    },

    async ask({ query, use_hyde, mode, provider, model, chat_id, persona_id, temp_file_path, temp_file_name, use_web_search }) {
        const response = await fetch('/api/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, use_hyde, mode, provider, model, chat_id, persona_id, temp_file_path, temp_file_name, use_web_search })
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
