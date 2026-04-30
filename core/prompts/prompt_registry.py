"""
Prompt Registry — Manages loading and caching of prompt templates from text files.
"""
import os
from typing import Dict

class PromptRegistry:
    def __init__(self, prompt_dir: str = None):
        if prompt_dir is None:
            # Default to the directory where this file is located
            prompt_dir = os.path.dirname(os.path.abspath(__file__))
        self.prompt_dir = prompt_dir
        self.cache: Dict[str, str] = {}

    def get(self, name: str, use_cache: bool = True) -> str:
        """
        Load a prompt by name (e.g., 'rag_system').
        Looks for 'name.txt' in the prompt directory.
        """
        if use_cache and name in self.cache:
            return self.cache[name]

        file_path = os.path.join(self.prompt_dir, f"{name}.txt")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Prompt file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            self.cache[name] = content
            return content

    def reload_all(self):
        """Clear the cache to force reloading from disk."""
        self.cache.clear()

# Global instance
registry = PromptRegistry()
