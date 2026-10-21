from fastapi import APIRouter

from app.config import get_settings
from app.services import embeddings

router = APIRouter(tags=["health"])
s = get_settings()


@router.get("/health")
async def health():
    return {"status": "ok", "service": "DocuMind AI Copilot", "version": s.api_version}


@router.get("/api/v1/health")
async def api_health():
    stats = embeddings.get_stats()
    return {
        "status": "ok",
        "service": "DocuMind AI Copilot",
        "version": s.api_version,
        "model": s.llm_model,
        "embedding_model": s.embedding_model,
        "indexed_documents": stats["documents"],
        "total_chunks": stats["total_chunks"],
    }
