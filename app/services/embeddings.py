import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import faiss
from app.config import get_settings, logger

s = get_settings()

_model: Optional[SentenceTransformer] = None
_index: Optional[faiss.IndexIDMap] = None
_chunks: List[Dict[str, Any]] = []
_next_id: int = 0
EMBEDDING_DIM = 384


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {s.embedding_model}")
        _model = SentenceTransformer(s.embedding_model)
    return _model


def _get_index_path() -> str:
    return os.path.join(s.data_dir, "faiss.index")


def _get_chunks_path() -> str:
    return os.path.join(s.data_dir, "chunks.pkl")


def _get_meta_path() -> str:
    return os.path.join(s.data_dir, "meta.json")


def initialize():
    global _index, _chunks, _next_id
    os.makedirs(s.data_dir, exist_ok=True)

    if os.path.exists(_get_index_path()) and os.path.exists(_get_chunks_path()):
        try:
            _index = faiss.read_index(_get_index_path())
            with open(_get_chunks_path(), "rb") as f:
                _chunks = pickle.load(f)
            if os.path.exists(_get_meta_path()):
                with open(_get_meta_path()) as f:
                    meta = json.load(f)
                _next_id = meta.get("next_id", len(_chunks))
            else:
                _next_id = len(_chunks)
            logger.info(f"Loaded FAISS index with {len(_chunks)} chunks from {s.data_dir}")
            return
        except Exception as e:
            logger.warning(f"Failed to load persisted index: {e}")

    _index = faiss.IndexIDMap(faiss.IndexFlatIP(EMBEDDING_DIM))
    _chunks = []
    _next_id = 0
    logger.info("Initialized new in-memory FAISS index")


def embed_texts(texts: List[str]) -> np.ndarray:
    model = _get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(embeddings, dtype=np.float32)


def embed_text(text: str) -> np.ndarray:
    return embed_texts([text])[0]


def add_chunks(chunks: List[Dict[str, Any]]):
    global _next_id
    if not chunks:
        return
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    ids = np.arange(_next_id, _next_id + len(chunks), dtype=np.int64)
    _index.add_with_ids(embeddings, ids)
    for i, chunk in enumerate(chunks):
        chunk["id"] = int(ids[i])
        _chunks.append(chunk)
    _next_id += len(chunks)
    _persist()
    logger.info(f"Added {len(chunks)} chunks (total: {len(_chunks)})")


def search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    if not _chunks or _index.ntotal == 0:
        return []
    query_vec = embed_text(query).reshape(1, -1)
    distances, indices = _index.search(query_vec, min(top_k, _index.ntotal))
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        chunk = _chunks[idx]
        results.append({**chunk, "score": float(dist)})
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def remove_document(doc_id: str) -> int:
    global _chunks, _next_id, _index
    ids_to_remove = [c["id"] for c in _chunks if c.get("document_id") == doc_id]
    if not ids_to_remove:
        return 0
    _chunks = [c for c in _chunks if c.get("document_id") != doc_id]
    _index = faiss.IndexIDMap(faiss.IndexFlatIP(EMBEDDING_DIM))
    _next_id = 0
    if _chunks:
        texts = [c["text"] for c in _chunks]
        embs = embed_texts(texts)
        ids = np.arange(len(_chunks), dtype=np.int64)
        _index.add_with_ids(embs, ids)
        for i, c in enumerate(_chunks):
            c["id"] = int(ids[i])
        _next_id = len(_chunks)
    _persist()
    logger.info(f"Removed doc {doc_id} ({len(ids_to_remove)} chunks)")
    return len(ids_to_remove)


def list_documents() -> List[Dict]:
    seen = {}
    for c in _chunks:
        did = c.get("document_id", "")
        if did not in seen:
            seen[did] = {"document_id": did, "chunks": 0}
        seen[did]["chunks"] += 1
    return list(seen.values())


def get_stats() -> dict:
    return {
        "total_chunks": len(_chunks),
        "index_size": _index.ntotal if _index else 0,
        "documents": len(list_documents()),
    }


def _persist():
    try:
        os.makedirs(s.data_dir, exist_ok=True)
        faiss.write_index(_index, _get_index_path())
        with open(_get_chunks_path(), "wb") as f:
            pickle.dump(_chunks, f)
        with open(_get_meta_path(), "w") as f:
            json.dump({"next_id": _next_id}, f)
    except Exception as e:
        logger.error(f"Persist failed: {e}")
