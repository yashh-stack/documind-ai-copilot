import json
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from app.config import get_settings, logger
from app.routes import chat, health, upload
from app.services import embeddings

s = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"DocuMind v{s.api_version} starting...")
    logger.info(f"LLM: {s.llm_model} | Embeddings: {s.embedding_model} | Data: {s.data_dir}")
    os.makedirs(s.data_dir, exist_ok=True)
    embeddings.initialize()
    yield


app = FastAPI(
    title=s.api_title,
    version=s.api_version,
    docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT") != "production" else None,
    lifespan=lifespan,
)

# ── CORS: locked to explicit origins in production ─────────────────────────────
_raw_origins = s.cors_origins.strip()
if _raw_origins == "*":
    # Wildcard only permitted locally; prod must set CORS_ORIGINS explicitly
    origins = ["*"]
    if os.getenv("ENVIRONMENT") == "production":
        logger.warning("CORS_ORIGINS is '*' in production — lock this to your Vercel domain!")
else:
    origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
)


_request_count = 0
_error_count = 0


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
    request.state.request_id = rid
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    response.headers["X-Request-ID"] = rid
    response.headers["X-Response-Time-MS"] = str(round(elapsed * 1000, 1))
    logger.info(f"{request.method} {request.url.path} {response.status_code} {round(elapsed * 1000, 1)}ms")
    global _request_count
    _request_count += 1
    if response.status_code >= 500:
        global _error_count
        _error_count += 1
    return response


@app.get("/env.js")
async def env_js():
    api_url = s.frontend_api_url or ""
    return Response(
        content=f"window.API_URL={json.dumps(api_url)};",
        media_type="application/javascript",
        headers={"Cache-Control": "no-cache"},
    )


app.include_router(health.router)
app.include_router(upload.router)
app.include_router(chat.router)


@app.get("/metrics")
async def metrics():
    stats = embeddings.get_stats()
    return {
        "documind_requests_total": _request_count,
        "documind_errors_total": _error_count,
        "documind_chunks_indexed": stats["total_chunks"],
        "documind_documents_indexed": stats["documents"],
    }


@app.get("/")
async def serve_index():
    return FileResponse("app/templates/index.html")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={
        "answer": None, "citations": [], "status": "error",
        "message": "An unexpected system error occurred.",
    })
