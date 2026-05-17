import json
import os
import config

class PersonaService:
    def __init__(self, config_path: str = None):
        if not config_path:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, "core", "prompts", "personas.json")
        self.config_path = config_path
        self.registry = self._load_personas()
        
    def _load_personas(self) -> dict:
        if not os.path.exists(self.config_path):
            return {"default": "general_assistant", "personas": {}}
            
        with open(self.config_path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {"default": "general_assistant", "personas": {}}
                
    def get_persona(self, persona_id: str) -> dict:
        """Get persona config by ID or return default."""
        self.registry = self._load_personas() # Reload from disk to catch manual edits
        personas = self.registry.get("personas", {})
        
        # Fallback to default if not found
        if not persona_id or persona_id not in personas:
            persona_id = self.registry.get("default", "general_assistant")
            
        # Return empty dict if even default doesn't exist (failsafe)
        return personas.get(persona_id, {
            "meta": {"label": "Default", "icon": "fas fa-robot"},
            "prompt": {"system_role": ""},
            "model_config": {}
        })

    def get_all_personas(self) -> dict:
        """Get all available personas for UI."""
        self.registry = self._load_personas() # Reload from disk to catch manual edits
        return {
            "default": self.registry.get("default", "general_assistant"),
            "personas": self.registry.get("personas", {})
        }

    def add_persona(self, label: str, description: str, system_role: str, tone: str = "neutral", language: str = "th", temperature: float = 0.5) -> str:
        """Add a custom persona and save it to the registry."""
        import time
        persona_id = f"custom_{int(time.time())}"
        
        # Ensure 'personas' dict exists
        if "personas" not in self.registry:
            self.registry["personas"] = {}
            
        self.registry["personas"][persona_id] = {
            "meta": {
                "label": label,
                "description": description,
                "icon": "fas fa-user-astronaut",
                "color": "#14b8a6" # Teal-500
            },
            "prompt": {
                "system_role": system_role,
                "tone": tone,
                "language": language
            },
            "model_config": {
                "temperature": temperature
            }
        }
        
        # Save to file
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, ensure_ascii=False, indent=2)
            
        return persona_id

persona_service = PersonaService()
