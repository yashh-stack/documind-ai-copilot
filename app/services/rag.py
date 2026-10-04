import re
import json
import time
import uuid
import asyncio
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import httpx
from app.config import get_settings, logger
from app.services import embeddings

s = get_settings()

# ═══════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════

SYSTEM_PROMPT = """You are DocuMind, an elite AI Customer Support Copilot built for enterprise teams.

## Your Core Principles
- You solve problems DIRECTLY. Never deflect with "contact support" or "I cannot help."
- You are precise, structured, and human. Never robotic.
- You cite specific document sections when available.
- You anticipate what the user needs next.

## Response Structure
1. **Direct Answer** -- Lead with the solution. No preamble.
2. **Explanation** -- Provide context only when it adds value.
3. **Action Steps** -- Numbered steps if the solution requires actions.
4. **Edge Cases** -- Mention relevant caveats or exceptions.

## Tone
Professional but warm. Think senior support engineer at a top SaaS company."""

RAG_PROMPT_TEMPLATE = """{system_prompt}

## Retrieved Knowledge Base Context
The following excerpts were retrieved from the company's documentation. Use them to ground your answer.
If the context doesn't contain relevant information, say so honestly rather than fabricating.
---
{context}
---

## Conversation History (Recent)
{history}

## Current Question
{query}

## Instructions
- Answer using ONLY the provided context when possible
- If context is insufficient, clearly state what you know and what you don't
- Be specific -- reference document sections, page numbers, or policy details
- Do NOT repeat the question back"""

NO_CONTEXT_PROMPT = """{system_prompt}

The knowledge base is currently empty -- no documents have been uploaded yet.

The user asked: {query}

Respond helpfully:
1. Acknowledge you don't have company-specific documents to reference
2. Provide any general guidance you can based on the question
3. Suggest the user upload relevant documents for accurate, source-grounded answers"""

# ═══════════════════════════════════════════════════
# Memory
# ═══════════════════════════════════════════════════

_session_memory: Dict[str, List[Dict]] = defaultdict(list)


class MemoryManager:
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.window_size = s.memory_window_size

    @property
    def turns(self):
        return _session_memory[self.session_id]

    def add_turn(self, user_msg: str, assistant_msg: str) -> None:
        _session_memory[self.session_id].append({
            "user": user_msg, "assistant": assistant_msg, "timestamp": time.time(),
        })
        if len(_session_memory[self.session_id]) > self.window_size * 2:
            _session_memory[self.session_id] = _session_memory[self.session_id][-self.window_size:]

    def get_short_term(self):
        return self.turns[-self.window_size:]

    def get_context_string(self) -> str:
        recent = self.get_short_term()
        if not recent:
            return ""
        lines = []
        for turn in recent:
            lines.append(f"User: {turn['user']}")
            a = turn['assistant']
            lines.append(f"Assistant: {a[:300] + '...' if len(a) > 300 else a}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════
# OpenRouter LLM
# ═══════════════════════════════════════════════════

class LLMError(Exception):
    pass


class LLMQuotaError(LLMError):
    pass


async def _call_openrouter_sync(
    messages: List[Dict], max_tokens: int = 1000, temperature: float = None,
) -> dict:
    api_key = s.openrouter_api_key
    if not api_key:
        raise LLMError("OPENROUTER_API_KEY not configured")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://documind.app",
        "X-Title": "DocuMind AI Copilot",
    }
    body = {
        "model": s.llm_model, "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature if temperature is not None else s.llm_temperature,
        "stream": False,
    }

    last_error = None
    for attempt in range(s.llm_max_retries):
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{s.openrouter_base_url}/chat/completions",
                    headers=headers, json=body,
                )
                if resp.status_code == 401:
                    raise LLMError("Invalid OpenRouter API key")
                if resp.status_code in (402, 429):
                    raise LLMQuotaError("OpenRouter quota/rate limit exceeded")
                if resp.status_code != 200:
                    raise LLMError(f"OpenRouter error {resp.status_code}: {resp.text[:200]}")
                return resp.json()
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            last_error = e
            logger.warning(f"OpenRouter attempt {attempt + 1} failed: {e}")
            if attempt < s.llm_max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            continue
        except (LLMError, LLMQuotaError):
            raise
        except Exception as e:
            raise LLMError(f"LLM call failed: {e}")

    raise LLMError(f"LLM call failed after {s.llm_max_retries} retries: {last_error}")


async def _call_openrouter_stream(
    messages: List[Dict], max_tokens: int = 1000, temperature: float = None,
):
    api_key = s.openrouter_api_key
    if not api_key:
        raise LLMError("OPENROUTER_API_KEY not configured")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://documind.app",
        "X-Title": "DocuMind AI Copilot",
    }
    body = {
        "model": s.llm_model, "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature if temperature is not None else s.llm_temperature,
        "stream": True,
    }

    last_error = None
    for attempt in range(s.llm_max_retries):
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST", f"{s.openrouter_base_url}/chat/completions",
                    headers=headers, json=body,
                ) as resp:
                    if resp.status_code == 401:
                        raise LLMError("Invalid OpenRouter API key")
                    if resp.status_code in (402, 429):
                        raise LLMQuotaError("OpenRouter quota/rate limit exceeded")
                    if resp.status_code != 200:
                        text = await resp.aread()
                        raise LLMError(f"OpenRouter error {resp.status_code}: {text.decode()[:200]}")
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data.strip() == "[DONE]":
                                break
                            try:
                                yield json.loads(data)
                            except json.JSONDecodeError:
                                continue
            return
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            last_error = e
            logger.warning(f"OpenRouter attempt {attempt + 1} failed: {e}")
            if attempt < s.llm_max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            continue
        except (LLMError, LLMQuotaError):
            raise
        except Exception as e:
            raise LLMError(f"LLM call failed: {e}")

    raise LLMError(f"LLM call failed after {s.llm_max_retries} retries: {last_error}")


async def generate_answer(query: str, context: str = "",
                          history: List[Dict] = None, no_context: bool = False) -> str:
    history_str = ""
    if history:
        lines = []
        for turn in history[-5:]:
            r = turn.get("role", "user")
            c = turn.get("content", "")
            lines.append(f"{r.capitalize()}: {c}")
        history_str = "\n".join(lines)

    if no_context:
        prompt = NO_CONTEXT_PROMPT.format(system_prompt=SYSTEM_PROMPT, query=query)
    else:
        prompt = RAG_PROMPT_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT, context=context, history=history_str, query=query)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    data = await _call_openrouter_sync(messages)
    return data["choices"][0]["message"]["content"].strip()


async def stream_answer(query: str, context: str = "",
                        history: List[Dict] = None, no_context: bool = False):
    history_str = ""
    if history:
        lines = []
        for turn in history[-5:]:
            r = turn.get("role", "user")
            c = turn.get("content", "")
            lines.append(f"{r.capitalize()}: {c}")
        history_str = "\n".join(lines)

    if no_context:
        prompt = NO_CONTEXT_PROMPT.format(system_prompt=SYSTEM_PROMPT, query=query)
    else:
        prompt = RAG_PROMPT_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT, context=context, history=history_str, query=query)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    try:
        async for chunk in _call_openrouter_stream(messages):
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content")
            if content:
                yield content
    except LLMQuotaError as e:
        yield f"\n\n> WARNING: {str(e)}"
    except LLMError as e:
        yield f"\n\n> ERROR: {str(e)}"


# ═══════════════════════════════════════════════════
# Chunking
# ═══════════════════════════════════════════════════

def chunk_text(text: str, document_id: str) -> List[Dict[str, Any]]:
    paragraphs = re.split(r'\n{2,}', text)
    chunks = []
    current_paras = []
    current_len = 0
    current_page = 1

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if "PAGE_" in para:
            try:
                current_page = int(para.split("PAGE_")[1].split()[0])
                para = re.sub(r'PAGE_\d+', '', para).strip()
                if not para:
                    continue
            except (ValueError, IndexError):
                pass
        current_paras.append(para)
        current_len += len(para)
        if current_len >= s.chunk_size:
            chunks.append({"document_id": document_id, "page": current_page, "text": " ".join(current_paras)})
            overlap = []
            ol = 0
            for p in reversed(current_paras):
                if ol + len(p) <= s.chunk_overlap:
                    overlap.insert(0, p)
                    ol += len(p)
                else:
                    break
            current_paras = overlap
            current_len = ol

    if current_paras:
        chunks.append({"document_id": document_id, "page": current_page, "text": " ".join(current_paras)})
    logger.info(f"Chunked {document_id}: {len(chunks)} chunks")
    return chunks


# ═══════════════════════════════════════════════════
# Ingestion
# ═══════════════════════════════════════════════════

async def ingest_pdf(file_bytes: bytes, filename: str) -> Tuple[str, int]:
    import PyPDF2
    from io import BytesIO

    doc_id = str(uuid.uuid4())[:8] + "_" + filename.replace(" ", "_").lower().replace(".pdf", "")

    try:
        reader = PyPDF2.PdfReader(BytesIO(file_bytes))
        num_pages = len(reader.pages)
        if num_pages > s.max_pages:
            raise ValueError(f"PDF has {num_pages} pages; max allowed is {s.max_pages}")
        full_text_parts = []
        for i in range(num_pages):
            text = reader.pages[i].extract_text()
            if text and text.strip():
                full_text_parts.append(f"\nPAGE_{i + 1}\n{text.strip()}")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Invalid PDF structure: {e}")

    raw_text = "\n".join(full_text_parts)
    if not raw_text.strip():
        raise ValueError("No extractable text found in PDF")

    chunks = chunk_text(raw_text, doc_id)
    if not chunks:
        raise ValueError("Chunking produced zero chunks")

    embeddings.add_chunks(chunks)
    logger.info(f"Ingested '{filename}' -> {doc_id} ({len(chunks)} chunks, {num_pages} pages)")
    return doc_id, len(chunks)


# ═══════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════

async def run_pipeline(query: str, history: List[Dict] = None,
                       session_id: str = "default") -> Tuple[str, List[Dict], Dict]:
    pipeline_start = time.time()
    memory = MemoryManager(session_id)
    retrieved = embeddings.search(query, top_k=s.top_k_retrieval)

    if not retrieved:
        answer = await generate_answer(query, "", history, no_context=True)
        latency = round((time.time() - pipeline_start) * 1000, 1)
        return answer, [], {"confidence_score": 0.0, "latency_ms": latency, "eval_metrics": {"retrieved_chunks": 0}}

    context_text = "\n\n---\n\n".join([
        f"[Source: {c['document_id']} | Page {c['page']}]\n{c['text']}" for c in retrieved
    ])
    mem_ctx = memory.get_context_string()
    if mem_ctx:
        context_text = f"## Previous Conversation Context\n{mem_ctx}\n\n## Document Context\n{context_text}"

    answer = await generate_answer(query, context_text, history)

    citations = []
    for c in retrieved:
        if c.get("score", 0) > 0.25:
            citations.append({
                "document_id": c["document_id"], "page": c["page"],
                "snippet": c["text"][:200] + "...", "relevance_score": round(c["score"], 3),
            })

    memory.add_turn(query, answer)
    total_latency = round((time.time() - pipeline_start) * 1000, 1)

    extras = {
        "confidence_score": round(citations[0]["relevance_score"] * 100, 1) if citations else 0.0,
        "latency_ms": total_latency,
        "eval_metrics": {"retrieved_chunks": len(retrieved), "citation_count": len(citations)},
    }
    return answer, citations, extras


async def stream_pipeline(query: str, history: List[Dict] = None, session_id: str = "default"):
    pipeline_start = time.time()
    memory = MemoryManager(session_id)
    retrieved = embeddings.search(query, top_k=s.top_k_retrieval)

    if not retrieved:
        context_text = ""
        no_context = True
    else:
        context_text = "\n\n---\n\n".join([
            f"[Source: {c['document_id']} | Page {c['page']}]\n{c['text']}" for c in retrieved
        ])
        mem_ctx = memory.get_context_string()
        if mem_ctx:
            context_text = f"## Previous Conversation Context\n{mem_ctx}\n\n## Document Context\n{context_text}"
        no_context = False

    full_answer = ""
    async for token in stream_answer(query, context_text, history, no_context=no_context):
        full_answer += token
        yield {"type": "token", "content": token}

    citations = []
    if retrieved:
        for c in retrieved:
            if c.get("score", 0) > 0.25:
                citations.append({
                    "document_id": c["document_id"], "page": c["page"],
                    "snippet": c["text"][:200] + "...", "relevance_score": round(c["score"], 3),
                })

    memory.add_turn(query, full_answer)
    total_latency = round((time.time() - pipeline_start) * 1000, 1)

    yield {"type": "metadata", "data": {
        "citations": citations,
        "confidence_score": round(citations[0]["relevance_score"] * 100, 1) if citations else 0.0,
        "latency_ms": total_latency,
    }}
    yield {"type": "done"}
