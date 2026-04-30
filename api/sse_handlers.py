"""
SSE Handlers — Format service output into Server-Sent Events (SSE).
"""
import json
from services.chat_service import chat_service
import config

async def classic_event_generator(query: str, use_hyde: bool):
    """Wraps ChatService classic pipeline with SSE formatting."""
    async for event in chat_service.run_classic_pipeline(query, use_hyde):
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
            yield {"event": "token", "data": json.dumps({"text": event["text"]})}
            
        elif e_type == "done":
            yield {"event": "done", "data": json.dumps(event)}

async def agentic_event_generator(query: str, use_hyde: bool):
    """Wraps ChatService agentic pipeline with SSE formatting."""
    async for event_wrapper in chat_service.run_agentic_pipeline(query, use_hyde):
        if event_wrapper["type"] == "status":
            yield {"event": "status", "data": json.dumps({"stage": event_wrapper["stage"], "message": event_wrapper["message"]})}
            continue
            
        event = event_wrapper["event"]
        
        if event.event_type == "decompose":
            d = event.data
            msg = f"🔀 แยกเป็น {len(d['sub_queries'])} sub-queries"
            yield {
                "event": "decompose",
                "data": json.dumps({
                    "query_type": d["query_type"],
                    "sub_queries": d["sub_queries"],
                    "reasoning": d["reasoning"],
                    "message": msg,
                }, ensure_ascii=False),
            }

        elif event.event_type == "search_start":
            d = event.data
            msg = f"🔍 ค้นหารรอบ {d['iteration']}/{d['total_iterations']}: {d['query'][:80]}"
            yield {
                "event": "search_iteration",
                "data": json.dumps({
                    "iteration": d["iteration"],
                    "query": d["query"],
                    "message": msg,
                }, ensure_ascii=False),
            }
            yield {
                "event": "status",
                "data": json.dumps({
                    "stage": "search",
                    "message": msg,
                }, ensure_ascii=False),
            }

        elif event.event_type == "search_done":
            d = event.data
            yield {
                "event": "search_done",
                "data": json.dumps({
                    "iteration": d["iteration"],
                    "query": d["query"],
                    "num_results": d["num_results"],
                    "new_chunks": d["new_chunks"],
                    "total_chunks": d["total_chunks"],
                }, ensure_ascii=False),
            }

        elif event.event_type == "evaluate":
            d = event.data
            status = "✅ ข้อมูลเพียงพอ" if d["is_sufficient"] else f"🔄 ยังไม่ครบ (confidence={d['confidence']:.0%})"
            yield {
                "event": "evaluate",
                "data": json.dumps({
                    "is_sufficient": d["is_sufficient"],
                    "confidence": d["confidence"],
                    "missing_aspects": d["missing_aspects"],
                    "message": status,
                }, ensure_ascii=False),
            }
            yield {
                "event": "status",
                "data": json.dumps({
                    "stage": "evaluate",
                    "message": status,
                }, ensure_ascii=False),
            }

        elif event.event_type == "sources":
            d = event.data
            yield {
                "event": "sources",
                "data": json.dumps({
                    "sources": d["sources"],
                    "search_time": 0,
                    "iterations": d.get("iterations", 1),
                    "total_chunks": d.get("total_chunks", 0),
                }, ensure_ascii=False),
            }

        elif event.event_type == "synthesize":
            d = event.data
            yield {
                "event": "status",
                "data": json.dumps({
                    "stage": "generate",
                    "message": f"🤖 สังเคราะห์คำตอบจาก {d['total_chunks']} chunks ({d['iterations']} iterations)...",
                }),
            }

        elif event.event_type == "token":
            yield {
                "event": "token",
                "data": json.dumps({"text": event.data["text"]}),
            }

        elif event.event_type == "done":
            d = event.data
            yield {
                "event": "done",
                "data": json.dumps({
                    "mode": "agentic",
                    "iterations": d.get("iterations", 1),
                    "query_type": d.get("query_type", "simple"),
                    "sub_queries": d.get("sub_queries", []),
                    "total_chunks": d.get("total_chunks", 0),
                }),
            }
