import sys
import types

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routes import upload


def test_upload_documents_accepts_multipart_pdf(monkeypatch) -> None:
    """Verifies multipart PDF uploads are accepted and routed to ingestion."""

    async def fake_ingest_pdf(file_bytes: bytes, filename: str) -> tuple[str, int]:
        """Returns a stable ingestion result without mutating the vector index."""
        assert file_bytes == b"%PDF-1.4\nfake pdf bytes"
        assert filename == "sample.pdf"
        return "doc_sample", 3

    fake_rag = types.SimpleNamespace(ingest_pdf=fake_ingest_pdf)
    monkeypatch.setitem(sys.modules, "app.services.rag", fake_rag)
    app = FastAPI()
    app.include_router(upload.router)
    client = TestClient(app)

    response = client.post(
        "/api/v1/upload",
        files={"files": ("sample.pdf", b"%PDF-1.4\nfake pdf bytes", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "message": "Indexed 1 document(s), 3 chunks created.",
        "document_ids": ["doc_sample"],
        "chunk_count": 3,
        "warnings": None,
    }


def test_upload_documents_rejects_non_pdf() -> None:
    """Verifies non-PDF multipart uploads fail with a visible client error."""

    app = FastAPI()
    app.include_router(upload.router)
    client = TestClient(app)

    response = client.post(
        "/api/v1/upload",
        files={"files": ("notes.txt", b"plain text", "text/plain")},
    )

    assert response.status_code == 422
    assert response.json() == {
        "status": "error",
        "message": "Skipped notes.txt: not a PDF",
        "document_ids": [],
        "chunk_count": 0,
        "warnings": ["Skipped notes.txt: not a PDF"],
    }
