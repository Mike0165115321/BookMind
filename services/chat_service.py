"""
ChatService — Orchestrator for RAG and LLM Generation.

This service acts as a bridge between the API layer and the core components.
It manages the RAGSearcher and coordinates the search/generation flow.
"""
import asyncio
import time
import config
from rag_searcher import RAGSearcher
from core.llm_generator import generate
from core.query_transformer import hyde_transform
from core.agentic_controller import AgenticController

class ChatService:
    def __init__(self):
        self.searcher = RAGSearcher()
        self.searcher.load_index()
        self.agentic_controller = None

    def get_searcher(self):
        return self.searcher

    async def run_classic_pipeline(self, query: str, use_hyde: bool = True):
        """
        Executes the classic RAG pipeline.
        Yields status updates and final results (non-SSE).
        """
        t_total = time.time()
        
        # 1. HyDE
        search_query = query
        hyde_time = 0
        if use_hyde and config.ENABLE_HYDE:
            yield {"type": "status", "stage": "hyde", "message": "🪄 กำลังสร้าง HyDE..."}
            t_hyde = time.time()
            search_query = await asyncio.to_thread(hyde_transform, query)
            hyde_time = time.time() - t_hyde
            yield {"type": "hyde", "hyde_query": search_query[:200], "time": round(hyde_time, 2)}

        # 2. Search
        yield {"type": "status", "stage": "search", "message": "🔍 กำลังค้นหา..."}
        t_search = time.time()
        results = await asyncio.to_thread(self.searcher.search, search_query, config.TOP_K_RETRIEVAL)
        search_time = time.time() - t_search
        
        yield {"type": "sources", "results": results, "search_time": search_time}

        # 3. Generate (Streaming)
        yield {"type": "status", "stage": "generate", "message": f"🤖 กำลังสร้างคำตอบ ({config.GEMINI_MODEL})..."}
        t_gen = time.time()
        
        for chunk in generate(query, results[:config.TOP_K_DISPLAY], stream=True):
            yield {"type": "token", "text": chunk}
            await asyncio.sleep(0)

        gen_time = time.time() - t_gen
        total_time = time.time() - t_total
        
        yield {
            "type": "done",
            "mode": "classic",
            "hyde_time": round(hyde_time, 2),
            "search_time": round(search_time, 3),
            "gen_time": round(gen_time, 2),
            "total_time": round(total_time, 2),
        }

    async def run_agentic_pipeline(self, query: str, use_hyde: bool = True):
        """
        Executes the agentic RAG pipeline.
        Yields status updates and events from AgenticController.
        """
        if not self.agentic_controller:
            self.agentic_controller = AgenticController(searcher=self.searcher, use_hyde=use_hyde)
        
        # The controller already has a stream method, we just wrap it
        yield {"type": "status", "stage": "decompose", "message": "🧠 กำลังวิเคราะห์คำถาม..."}
        
        events = await asyncio.to_thread(
            lambda: list(self.agentic_controller.run_stream_with_answer(query))
        )
        
        for event in events:
            yield {"type": "agentic_event", "event": event}
            await asyncio.sleep(0)

# Global instance
chat_service = ChatService()
