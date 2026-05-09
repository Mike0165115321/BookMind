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

    async ask(payload) {
        return fetch('/api/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
    }
};
