import uuid
import json
import time
import config
from core.database import db
from services.chat_service import chat_service

async def classic_event_generator(query: str, use_hyde: bool, provider: str = "gemini", model: str = None, chat_id: str = None, persona_id: str = "default", temp_file_path: str = None, temp_file_name: str = None):
    """Wraps ChatService classic pipeline with SSE formatting and DB persistence."""
    t_start = time.time()
    
    if not chat_id:
        chat_id = str(uuid.uuid4())
        db.create_chat(chat_id, title=query[:50]) # Use query as initial title
    
    # 1. Save User Message
    db.add_message(chat_id, "user", query)
    
    # Send chat_id to frontend first
    yield {"event": "chat_id", "data": json.dumps({"chat_id": chat_id})}

    # Emit session_init event with persona metadata
    from services.persona_service import persona_service
    p_config = persona_service.get_persona(persona_id)
    yield {"event": "session_init", "data": json.dumps({"persona": p_config.get("meta", {})})}

    full_ai_response = ""
    async for event in chat_service.run_classic_pipeline(query, use_hyde, provider, model, persona_id=persona_id, temp_file_path=temp_file_path, temp_file_name=temp_file_name):
        e_type = event.get("type")
        
        if e_type == "status":
            yield {"event": "status", "data": json.dumps({"stage": event["stage"], "message": event["message"]})}
        
        elif e_type == "hyde":
            yield {"event": "hyde", "data": json.dumps({"hyde_query": event["hyde_query"], "time": event["time"]})}
            
        elif e_type == "sources":
            results = event["results"]
            sources = []
            for i, (text, score) in enumerate(results[:config.TOP_K_DISPLAY]):
                title = text.split("]")[0].lstrip("[") if "[" in text else "ไม่ระบุ"
                sources.append({
                    "rank": i + 1,
                    "title": title,
                    "text": text[:300],
                    "score": round(float(score), 3),
                })
            yield {"event": "sources", "data": json.dumps({"sources": sources, "search_time": round(event["search_time"], 3)}, ensure_ascii=False)}
            
        elif e_type == "token":
            token = event["text"]
            full_ai_response += token
            yield {"event": "token", "data": json.dumps({"text": token})}
            
        elif e_type == "done":
            event["total_time"] = round(time.time() - t_start, 2)
            db.add_message(chat_id, "ai", full_ai_response, metadata=event)
            yield {"event": "done", "data": json.dumps(event)}

async def agentic_event_generator(query: str, use_hyde: bool, provider: str = "gemini", model: str = None, chat_id: str = None, persona_id: str = "default", temp_file_path: str = None, temp_file_name: str = None):
    """Wraps ChatService agentic pipeline with SSE formatting and DB persistence."""
    t_start = time.time()
    stage_times = {
        "decompose": 0,
        "search": 0,
        "evaluate": 0,
        "synthesize": 0
    }
    
    if not chat_id:
        chat_id = str(uuid.uuid4())
        db.create_chat(chat_id, title=query[:50])
    
    db.add_message(chat_id, "user", query)
    yield {"event": "chat_id", "data": json.dumps({"chat_id": chat_id})}

    # Emit session_init event with persona metadata
    from services.persona_service import persona_service
    p_config = persona_service.get_persona(persona_id)
    yield {"event": "session_init", "data": json.dumps({"persona": p_config.get("meta", {})})}

    full_ai_response = ""
    last_stage_time = time.time()
    
    async for event_wrapper in chat_service.run_agentic_pipeline(query, use_hyde, provider, model, persona_id=persona_id, temp_file_path=temp_file_path, temp_file_name=temp_file_name):
        now = time.time()
        
        if event_wrapper["type"] == "status":
            yield {"event": "status", "data": json.dumps({"stage": event_wrapper["stage"], "message": event_wrapper["message"]})}
            continue
            
        event = event_wrapper["event"]
        
        if event.event_type in ["decomposed", "decompose"]:
            duration = now - last_stage_time
            stage_times["decompose"] = round(duration, 2)
            last_stage_time = now
            
            d = event.data
            msg = f"🔀 แยกเป็น {len(d['sub_queries'])} sub-queries"
            yield {
                "event": "decompose",
                "data": json.dumps({
                    "query_type": d["query_type"],
                    "sub_queries": d["sub_queries"],
                    "reasoning": d["reasoning"],
                    "message": msg,
                    "time": stage_times["decompose"]
                }, ensure_ascii=False),
            }

        elif event.event_type in ["search_started", "search_start"]:
            last_stage_time = now # Reset for search duration
            d = event.data
            msg = f"🔍 ค้นหารอบ {d['iteration']}/{d['total_iterations']}: {d['query'][:80]}"
            yield {
                "event": "status", "data": json.dumps({"stage": "search", "message": msg})
            }

        elif event.event_type in ["search_completed", "search_done"]:
            duration = now - last_stage_time
            stage_times["search"] += duration
            last_stage_time = now
            
            d = event.data
            yield {
                "event": "search_done",
                "data": json.dumps({
                    "iteration": d["iteration"],
                    "query": d["query"],
                    "num_results": d["num_results"],
                    "time": round(duration, 2)
                }, ensure_ascii=False),
            }

        elif event.event_type in ["evaluation_completed", "evaluate"]:
            duration = now - last_stage_time
            stage_times["evaluate"] += duration
            last_stage_time = now
            
            d = event.data
            status = "✅ ข้อมูลเพียงพอ" if d["is_sufficient"] else f"🔄 ยังไม่ครบ (confidence={d['confidence']:.0%})"
            yield {
                "event": "evaluate",
                "data": json.dumps({
                    "is_sufficient": d["is_sufficient"],
                    "message": status,
                    "time": round(duration, 2)
                }, ensure_ascii=False),
            }

        elif event.event_type == "synthesis_started":
            last_stage_time = now
            d = event.data
            sources = []
            for i, chunk in enumerate(d.get("display_chunks", [])):
                title = chunk.metadata.get("title", "ไม่ระบุ")
                sources.append({
                    "rank": i + 1,
                    "title": title,
                    "text": chunk.text[:300],
                    "score": round(float(chunk.score), 3) if hasattr(chunk, 'score') else 0,
                })

            yield {
                "event": "sources",
                "data": json.dumps({"sources": sources, "total_chunks": d.get("total_chunks", 0)}),
            }
            yield {
                "event": "status",
                "data": json.dumps({"stage": "generate", "message": "🤖 กำลังสรุปคำตอบ..."}),
            }

        elif event.event_type == "token":
            chunk = event.data["text"]
            text = chunk.text if hasattr(chunk, 'text') else str(chunk)
            full_ai_response += text
            yield {"event": "token", "data": json.dumps({"text": text})}

        elif event.event_type in ["completed", "done"]:
            total_time = now - t_start
            stage_times["synthesize"] = round(now - last_stage_time, 2)
            
            result_obj = event.data.get("result")
            meta = {
                "mode": "agentic",
                "provider": provider,
                "model": model,
                "total_time": round(total_time, 2),
                "stage_times": {k: round(v, 2) for k, v in stage_times.items()},
                "iterations": getattr(result_obj, 'iterations', 1),
                "total_chunks": getattr(result_obj, 'total_chunks', 0),
            }
            
            db.add_message(chat_id, "ai", full_ai_response, metadata=meta)
            yield {"event": "done", "data": json.dumps(meta)}
