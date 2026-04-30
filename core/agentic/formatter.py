"""
Agentic Formatter — Maps internal engine events to UI-specific events (Thai).
"""
import json
from typing import Generator
from core.agentic.types import InternalEngineEvent, AgenticEvent

class AgenticFormatter:
    """
    Translates raw engine events into Thai-localized events for the UI.
    """
    @staticmethod
    def format(engine_event: InternalEngineEvent) -> Generator[AgenticEvent, None, None]:
        etype = engine_event.event_type
        data = engine_event.data

        if etype == "decomposed":
            msg = f"🔀 แยกเป็น {len(data['sub_queries'])} sub-queries"
            yield AgenticEvent(
                event_type="decompose",
                data={
                    **data,
                    "message": msg
                }
            )

        elif etype == "search_started":
            msg = f"🔍 ค้นหารอบ {data['iteration']}/{data['total_iterations']}: {data['query'][:80]}"
            # Special case: return both a specific event and a generic status event
            yield AgenticEvent(
                event_type="search_start",
                data={**data, "message": msg}
            )
            yield AgenticEvent(
                event_type="status",
                data={"stage": "search", "message": msg}
            )

        elif etype == "search_completed":
            yield AgenticEvent(
                event_type="search_done",
                data=data
            )

        elif etype == "evaluation_completed":
            status = "✅ ข้อมูลเพียงพอ" if data["is_sufficient"] else f"🔄 ยังไม่ครบ (confidence={data['confidence']:.0%})"
            yield AgenticEvent(
                event_type="evaluate",
                data={**data, "message": status}
            )
            yield AgenticEvent(
                event_type="status",
                data={"stage": "evaluate", "message": status}
            )

        elif etype == "synthesis_started":
            # Map balanced chunks to sources structure for UI
            display_chunks = data["display_chunks"]
            sources_data = []
            for i, (text, score) in enumerate(display_chunks):
                title = text.split("]")[0].lstrip("[") if "[" in text else "ไม่ระบุ"
                sources_data.append({
                    "rank": i + 1,
                    "title": title,
                    "text": text[:300],
                    "score": round(float(score), 3),
                })
            
            yield AgenticEvent(
                event_type="sources",
                data={
                    "sources": sources_data,
                    "total_chunks": data["total_chunks"],
                    "iterations": data["iterations"],
                }
            )
            
            yield AgenticEvent(
                event_type="status",
                data={
                    "stage": "generate",
                    "message": f"🤖 สังเคราะห์คำตอบจาก {data['total_chunks']} chunks ({data['iterations']} iterations)...",
                }
            )

        elif etype == "token":
            yield AgenticEvent(
                event_type="token",
                data=data
            )

        elif etype == "completed":
            res = data["result"]
            yield AgenticEvent(
                event_type="done",
                data={
                    "mode": "agentic",
                    "iterations": res.iterations,
                    "query_type": res.query_type,
                    "sub_queries": res.sub_queries,
                    "total_chunks": res.total_chunks,
                    "search_history": res.search_history
                }
            )
