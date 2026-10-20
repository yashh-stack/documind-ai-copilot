
from fastapi import APIRouter, File, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import get_settings, logger

router = APIRouter(prefix="/api/v1", tags=["documents"])
s = get_settings()


class UploadResponse(BaseModel):
    """Represents the result of a multipart PDF ingestion request."""

    status: str
    message: str
    document_ids: list[str] = Field(default_factory=list)
    chunk_count: int = 0
    warnings: list[str] | None = None


class DocumentItem(BaseModel):
    """Represents an indexed document summary."""

    document_id: str
    chunks: int


class DocumentListResponse(BaseModel):
    """Represents the current indexed document collection."""

    status: str
    documents: list[DocumentItem]
    count: int


class DeleteDocumentResponse(BaseModel):
    """Represents the result of deleting one indexed document."""

    status: str
    message: str


@router.post("/upload", response_model=UploadResponse)
async def upload_documents(files: list[UploadFile] = File(...)) -> UploadResponse | JSONResponse:  # noqa: B008
    """Ingests uploaded PDF files from multipart/form-data into the vector index."""
    doc_ids = []
    total_chunks = 0
    errors = []

    for f in files:
        filename = f.filename or "unnamed"
        if not filename.lower().endswith(".pdf"):
            errors.append(f"Skipped {filename}: not a PDF")
            continue

        try:
            content = await f.read()
            max_bytes = s.max_upload_size_mb * 1024 * 1024
            if len(content) > max_bytes:
                errors.append(f"{filename}: exceeds {s.max_upload_size_mb}MB limit")
                continue

            from app.services.rag import ingest_pdf

            doc_id, chunk_count = await ingest_pdf(content, filename)
            doc_ids.append(doc_id)
            total_chunks += chunk_count
            logger.info(f"Ingested {filename} -> {doc_id} ({chunk_count} chunks)")

        except ValueError as e:
            errors.append(f"{filename}: {e!s}")
            logger.error(f"Ingestion failed for {filename}: {e}")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{filename}: Processing error")
            logger.error(f"Unexpected error processing {filename}: {e}", exc_info=True)

    if not doc_ids and errors:
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={
            "status": "error", "message": "; ".join(errors),
            "document_ids": [], "chunk_count": 0, "warnings": errors,
        })

    return UploadResponse(
        status="success",
        document_ids=doc_ids,
        chunk_count=total_chunks,
        message=f"Indexed {len(doc_ids)} document(s), {total_chunks} chunks created.",
        warnings=errors if errors else None,
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
    """Lists indexed documents without mutating persisted state."""
    from app.services import embeddings

    docs = embeddings.list_documents()
    return DocumentListResponse(status="ok", documents=docs, count=len(docs))


@router.delete("/documents/{doc_id}", response_model=DeleteDocumentResponse)
async def delete_document(doc_id: str) -> DeleteDocumentResponse | JSONResponse:
    """Deletes all chunks for one indexed document from the vector index."""
    from app.services import embeddings

    removed = embeddings.remove_document(doc_id)
    if removed == 0:
        return JSONResponse(status_code=404, content={"status": "error", "message": "Document not found."})
    return DeleteDocumentResponse(status="ok", message=f"Document {doc_id} removed ({removed} chunks).")
