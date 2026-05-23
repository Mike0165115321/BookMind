"""
ChatService — Orchestrator for RAG and LLM Generation.

This service acts as a bridge between the API layer and the core components.
It manages the RAGSearcher and coordinates the search/generation flow.
"""
import asyncio
import time
import config
from rag_searcher import RAGSearcher
from core.llm.generator import generate
from core.query_transformer import hyde_transform
from core.agentic.engine import AgenticEngine
from core.agentic.formatter import AgenticFormatter

class ChatService:
    def __init__(self):
        self.searcher = RAGSearcher()
        self.searcher.load_index()
        self.agentic_engine = None

    def get_searcher(self):
        return self.searcher

    async def run_classic_pipeline(self, query: str, use_hyde: bool = True, provider: str = "gemini", model_name: str = None, persona_id: str = "default", temp_file_path: str = None, temp_file_name: str = None, use_web_search: bool = False):
        """
        Executes the classic RAG pipeline.
        Yields status updates and final results (non-SSE).
        """
        from core.llm.shared.types import ProviderName
        p_name = ProviderName(provider)
        t_total = time.time()
        
        # 1. HyDE
        search_query = query
        hyde_time = 0
        
        from core.database import db
        enable_web_hyde = db.get_setting("enable_web_hyde")
        is_web_hyde_enabled = enable_web_hyde in ["true", True, 1, "True", "1"]
        
        if use_web_search and is_web_hyde_enabled:
            w_p = db.get_setting("web_hyde_provider", provider)
            w_m = db.get_setting("web_hyde_model")
            
            yield {"type": "status", "stage": "hyde", "message": f"🪄 กำลังวิเคราะห์คำค้นหาเว็บ ({w_p})..."}
            t_hyde = time.time()
            from core.query_transformer import web_hyde_transform
            search_query = await asyncio.to_thread(web_hyde_transform, query, provider=ProviderName(w_p), model_name=w_m)
            hyde_time = time.time() - t_hyde
            yield {"type": "hyde", "hyde_query": search_query[:200], "time": round(hyde_time, 2)}
        elif use_hyde and config.ENABLE_HYDE and not temp_file_path: # Skip HyDE if file uploaded
            h_p = db.get_setting("hyde_provider", provider)
            h_m = db.get_setting("hyde_model", model_name)
            
            yield {"type": "status", "stage": "hyde", "message": f"🪄 กำลังสร้าง HyDE ({h_p})..."}
            t_hyde = time.time()
            search_query = await asyncio.to_thread(hyde_transform, query, provider=ProviderName(h_p), model_name=h_m)
            hyde_time = time.time() - t_hyde
            yield {"type": "hyde", "hyde_query": search_query[:200], "time": round(hyde_time, 2)}
 
        # 2. Search
        results = []
        search_time = 0
        temp_file_content = None
        
        if use_web_search:
            yield {"type": "status", "stage": "search", "message": "🌐 กำลังค้นหาบนเว็บ..."}
            from core.web_searcher import WebSearcher
            t_web = time.time()
            results = await asyncio.to_thread(WebSearcher.search, search_query[:180], 5)
            search_time = time.time() - t_web
        elif temp_file_path:
            yield {"type": "status", "stage": "search", "message": "📄 กำลังอ่านไฟล์แนบ..."}
            from core.document_loader import DocumentLoader
            import os
            try:
                docs = DocumentLoader.load(temp_file_path)
                if docs:
                    temp_file_content = "\n".join([doc["content"] for doc in docs])
                    # Use provided name or fallback to path basename
                    display_name = temp_file_name or os.path.basename(temp_file_path)
                    # Add to results for UI display with correct format
                    results.append((f"[{display_name}] เนื้อหาจากไฟล์แนบ", 1.0))
            except Exception as e:
                print(f"❌ Error loading temp file: {e}")
        else:
            yield {"type": "status", "stage": "search", "message": "🔍 กำลังค้นหา..."}
            t_search = time.time()
            results = await asyncio.to_thread(self.searcher.search, search_query, config.TOP_K_RETRIEVAL)
            search_time = time.time() - t_search
        
        yield {"type": "sources", "results": results, "search_time": search_time}

        # 3. Generate (Streaming)
        yield {"type": "status", "stage": "generate", "message": f"🤖 กำลังสร้างคำตอบ..."}
        t_gen = time.time()
        
        # Capture response metadata from the generator loop if possible, 
        # but for streaming, the metadata usually comes at the end or we track it here.
        # Note: generator.generate returns a generator for streaming
        for chunk in generate(query, results[:config.TOP_K_DISPLAY], stream=True, provider=p_name, model_name=model_name, persona_id=persona_id, temp_file_content=temp_file_content):
            # Check if chunk is LLMStreamChunk
            text = chunk.text if hasattr(chunk, 'text') else str(chunk)
            yield {"type": "token", "text": text}
            await asyncio.sleep(0)

        gen_time = time.time() - t_gen
        total_time = time.time() - t_total
        
        yield {
            "type": "done",
            "mode": "classic",
            "provider": provider,
            "model": model_name,
            "hyde_time": round(hyde_time, 2),
            "search_time": round(search_time, 3),
            "gen_time": round(gen_time, 2),
            "total_time": round(total_time, 2),
        }

    async def run_agentic_pipeline(self, query: str, use_hyde: bool = True, provider: str = "gemini", model_name: str = None, persona_id: str = "default", temp_file_path: str = None, temp_file_name: str = None, use_web_search: bool = False):
        """
        Executes the agentic RAG pipeline.
        """
        from core.llm.shared.types import ProviderName
        p_name = ProviderName(provider)
        
        # Fetch HyDE & Agentic settings from DB
        from core.database import db
        h_p = db.get_setting("hyde_provider", provider)
        h_m = db.get_setting("hyde_model", model_name)
        
        # Fetch Agentic brain settings (Decomposer/Evaluator) from DB
        a_p = db.get_setting("agentic_provider")
        a_m = db.get_setting("agentic_model")
        
        if not a_p or not a_m:
            # Fallback to main selection if agentic brain is not specifically configured
            a_p = a_p or provider
            a_m = a_m or model_name

        # Re-initialize engine with current selection
        self.agentic_engine = AgenticEngine(
            searcher=self.searcher, 
            use_hyde=use_hyde,
            provider=p_name,
            model_name=model_name,
            hyde_provider=h_p,
            hyde_model=h_m,
            agentic_provider=a_p,
            agentic_model=a_m,
            persona_id=persona_id,
            use_web_search=use_web_search
        )
        
        # We start with an initial status
        yield {"type": "status", "stage": "decompose", "message": "🧠 กำลังวิเคราะห์คำถาม..."}
        
        # Run engine in a thread and collect events
        def run_engine():
            for engine_event in self.agentic_engine.execute(query):
                for ui_event in AgenticFormatter.format(engine_event):
                    yield ui_event

        events = await asyncio.to_thread(lambda: list(run_engine()))
        
        for event in events:
            yield {"type": "agentic_event", "event": event}
            await asyncio.sleep(0)

# Global instance
chat_service = ChatService()
