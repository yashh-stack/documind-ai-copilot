import json
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.config import logger
from app.services.rag import LLMError, LLMQuotaError, run_pipeline, stream_pipeline

router = APIRouter(prefix="/api/v1", tags=["chat"])


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)
    session_id: str | None = "default"
    history: list[dict[str, str]] | None = []


class CitationItem(BaseModel):
    document_id: str
    page: int
    snippet: str
    relevance_score: float | None = 0.0


class QueryResponse(BaseModel):
    answer: str | None
    citations: list[CitationItem]
    status: str
    message: str | None = None
    confidence_score: float | None = 0.0
    latency_ms: float | None = 0.0
    eval_metrics: dict[str, Any] | None = {}


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    logger.info(f"Query: {req.question[:80]}...")
    try:
        answer, citations, extras = await run_pipeline(
            query=req.question, history=req.history, session_id=req.session_id,
        )
        return QueryResponse(
            answer=answer,
            citations=[CitationItem(**c) for c in citations],
            status="success",
            confidence_score=extras.get("confidence_score", 0.0),
            latency_ms=extras.get("latency_ms", 0.0),
            eval_metrics=extras.get("eval_metrics", {}),
        )
    except LLMQuotaError as e:
        return JSONResponse(status_code=429, content={
            "answer": None, "citations": [], "status": "error",
            "message": str(e), "error_type": "quota_exceeded",
            "confidence_score": 0.0, "latency_ms": 0.0,
        })
    except LLMError as e:
        return JSONResponse(status_code=503, content={
            "answer": None, "citations": [], "status": "error",
            "message": str(e), "error_type": "llm_error",
            "confidence_score": 0.0, "latency_ms": 0.0,
        })
    except Exception as e:  # noqa: BLE001
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={
            "answer": None, "citations": [], "status": "error",
            "message": "Pipeline execution failed.", "confidence_score": 0.0, "latency_ms": 0.0,
        })


@router.post("/chat/stream")
async def stream_chat_endpoint(req: QueryRequest):
    logger.info(f"Stream: {req.question[:80]}...")

    async def event_generator():
        try:
            async for chunk in stream_pipeline(
                query=req.question, history=req.history, session_id=req.session_id,
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:  # noqa: BLE001
            logger.error(f"Stream failed: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
