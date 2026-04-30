"""
Gemini Provider — Handles low-level interaction with Google Gemini API.

Responsibilities:
  - Key rotation (via KeyManager)
  - Retry logic for API calls
  - Streaming and non-streaming generation
"""
import time
from google import genai
from google.genai import types, errors
from core.config import settings
from core.key_manager import gemini_key_manager

class GeminiProvider:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.GEMINI_MODEL

    def _get_client(self) -> genai.Client:
        """Create a Gemini client with the next API key from rotation."""
        api_key = gemini_key_manager.get_key()
        if not api_key:
            raise RuntimeError("❌ ไม่มี API key สำหรับ Gemini — กรุณาตั้งค่าใน .env")
        return genai.Client(api_key=api_key)

    def generate(self, prompt: str, system_instruction: str = None, temperature: float = None):
        """Non-streaming generation with retry/rotation."""
        temperature = temperature if temperature is not None else settings.GEMINI_TEMPERATURE
        gen_config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        )

        max_retries = len(gemini_key_manager.keys) or 1
        last_error = None

        for attempt in range(max_retries):
            try:
                client = self._get_client()
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=gen_config,
                )
                return response.text
            except (errors.ServerError, errors.ClientError) as e:
                last_error = e
                status_code = getattr(e, 'status_code', 'Unknown')
                print(f"⚠️  Gemini Attempt {attempt+1} failed ({status_code}). Rotating key...")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    raise last_error
            except Exception as e:
                print(f"❌ Unexpected error in Gemini provider: {e}")
                raise e

    def generate_stream(self, prompt: str, system_instruction: str = None, temperature: float = None):
        """Streaming generation with retry/rotation."""
        temperature = temperature if temperature is not None else settings.GEMINI_TEMPERATURE
        gen_config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        )

        max_retries = len(gemini_key_manager.keys) or 1
        
        for attempt in range(max_retries):
            try:
                client = self._get_client()
                for chunk in client.models.generate_content_stream(
                    model=self.model_name,
                    contents=prompt,
                    config=gen_config,
                ):
                    if chunk.text:
                        yield chunk.text
                return # Success!
            except (errors.ServerError, errors.ClientError) as e:
                status_code = getattr(e, 'status_code', 'Unknown')
                print(f"⚠️  Gemini Stream Attempt {attempt+1} failed ({status_code}). Rotating key...")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    yield f"\n\n❌ [AI Error {status_code}]: {str(e)}\nกรุณาลองใหม่อีกครั้ง"
                    return
            except Exception as e:
                print(f"❌ Stream error in Gemini provider: {e}")
                yield f"\n\n❌ [Unexpected Error]: {str(e)}"
                return
