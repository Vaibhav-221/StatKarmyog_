"""
Uploaded-document RAG helpers for quiz generation.

This module keeps the quiz endpoint focused on request validation while reusing
the existing Sentence-Transformer dependency for document chunk embeddings.

Pipeline:
    PDF text → clean → chunk → embed → ChromaDB (ephemeral) → top-k retrieval
    → context string passed to Gemini.

If the embedding model or ChromaDB are unavailable (e.g. cold-start on a
memory-limited host), this module gracefully falls back to returning the raw
cleaned text so Gemini still receives PDF content rather than raising an error
that blocks quiz generation entirely.
"""

from __future__ import annotations

import re
import uuid
import logging

from app.services.semantic_search import _get_embedding_model

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1600
CHUNK_OVERLAP = 250
TOP_K_CHUNKS = 6

# Maximum characters of raw PDF text kept before chunking.
# Raised from 12 000 so that longer PDFs are fully indexed by the RAG step.
# The post-retrieval context sent to Gemini is naturally bounded by TOP_K_CHUNKS × CHUNK_SIZE.
RAW_TEXT_LIMIT = 60_000


def _as_plain_list(vectors):
    """Convert numpy/torch encoder output or test doubles into plain lists."""
    return vectors.tolist() if hasattr(vectors, "tolist") else vectors


def preload_embedding_model() -> None:
    """
    Warm up the sentence-transformer embedding model at server startup.

    Calling this once at startup means the first PDF quiz request does not
    have to download / load the 90 MB model mid-request, which can cause
    request timeouts on slow-start hosting platforms such as Render's free tier.
    This is safe to call even when course-semantic-search is disabled.
    """
    try:
        _get_embedding_model()
        logger.info("[RAG] Embedding model pre-loaded successfully.")
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "[RAG] Embedding model pre-load failed (PDF RAG will attempt lazy load per-request): %s", exc
        )


def clean_extracted_text(text: str) -> str:
    """Normalize extracted document text without inventing or rewriting content."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks suitable for embedding and retrieval."""
    cleaned = clean_extracted_text(text)
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        window = cleaned[start:end]

        if end < len(cleaned):
            split_at = max(window.rfind("\n\n"), window.rfind(". "), window.rfind(" "))
            if split_at > chunk_size // 2:
                end = start + split_at + 1
                window = cleaned[start:end]

        chunk = window.strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(cleaned):
            break
        start = max(0, end - overlap)

    return chunks


def _build_query(cleaned: str, target_competency: str | None) -> str:
    focus = target_competency or "professional statistical training assessment"
    return (
        "Find the most assessment-worthy uploaded-document passages for "
        f"{focus}. Include definitions, procedures, methods, examples, tables, "
        "quality checks, and decision rules.\n\n"
        + cleaned[:2000]
    )


def retrieve_relevant_context(
    text: str,
    *,
    source: str = "uploaded-document",
    target_competency: str | None = None,
    num_chunks: int = TOP_K_CHUNKS,
) -> dict:
    """
    Build an ephemeral vector index for the uploaded document and retrieve
    the most representative chunks for Gemini.

    Falls back to raw cleaned text if embedding or ChromaDB fails, so the
    quiz generation endpoint always receives PDF content.

    Returns
    -------
    dict with keys:
        context               – text string to pass to Gemini
        chunk_count           – total chunks created
        retrieved_chunk_count – chunks actually retrieved (0 on fallback)
        collection_name       – ChromaDB collection name (None on fallback)
        fallback              – True if vector retrieval was skipped
    """
    cleaned = clean_extracted_text(text)

    # Cap raw text before chunking; RAG handles its own retrieval window.
    if len(cleaned) > RAW_TEXT_LIMIT:
        logger.warning(
            "[RAG] source=%s raw_text_length=%d — truncating to %d before chunking.",
            source, len(cleaned), RAW_TEXT_LIMIT,
        )
        cleaned = cleaned[:RAW_TEXT_LIMIT]

    chunks = chunk_text(cleaned)
    logger.info("[CHUNKING] source=%s chunks_created=%d", source, len(chunks))

    if not chunks:
        raise ValueError(
            "Unable to extract readable text from this PDF. "
            "Please upload a text-based learning material PDF."
        )

    # ── Embedding ────────────────────────────────────────────────────────
    try:
        model = _get_embedding_model()
        embeddings = _as_plain_list(model.encode(chunks, show_progress_bar=False))
    except ModuleNotFoundError as exc:
        logger.exception("[EMBEDDING] dependency_missing source=%s", source)
        raise ValueError(
            "Backend PDF RAG dependency is missing. "
            "Install sentence-transformers and redeploy the backend."
        ) from exc
    except Exception as exc:
        logger.warning(
            "[EMBEDDING] status=failed source=%s error=%s — falling back to direct text",
            source, exc,
        )
        fallback_context = cleaned[:12_000]
        return {
            "context": fallback_context,
            "chunk_count": len(chunks),
            "retrieved_chunk_count": 0,
            "collection_name": None,
            "fallback": True,
        }

    if len(embeddings) != len(chunks):
        logger.error(
            "[EMBEDDING] status=invalid chunks=%d embeddings=%d", len(chunks), len(embeddings)
        )
        raise ValueError("Embedding generation returned an invalid number of vectors.")

    logger.info(
        "[EMBEDDING] model=all-MiniLM-L6-v2 embeddings_created=%d",
        len(embeddings),
    )

    # ── ChromaDB (ephemeral per-request collection) ───────────────────────
    collection_name = f"quiz_pdf_{uuid.uuid4().hex}"
    try:
        import chromadb as _chromadb
        client = _chromadb.EphemeralClient()
        collection = client.create_collection(name=collection_name)
        ids = [f"chunk-{idx}" for idx in range(len(chunks))]
        metadatas = [{"source": source, "chunk_index": idx} for idx in range(len(chunks))]
        collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
    except ModuleNotFoundError as exc:
        logger.exception("[CHROMADB] dependency_missing collection_name=%s", collection_name)
        raise ValueError(
            "Backend vector-store dependency is missing. "
            "Install chromadb and redeploy the backend."
        ) from exc
    except Exception as exc:
        logger.warning(
            "[CHROMADB] status=failed collection_name=%s error=%s — falling back to joined chunks",
            collection_name, exc,
        )
        fallback_context = "\n\n".join(chunks[:TOP_K_CHUNKS])[:12_000]
        return {
            "context": fallback_context,
            "chunk_count": len(chunks),
            "retrieved_chunk_count": 0,
            "collection_name": None,
            "fallback": True,
        }

    logger.info(
        "[CHROMADB] documents_added=%d collection_name=%s",
        len(chunks),
        collection_name,
    )

    # ── Retrieval ────────────────────────────────────────────────────────
    query_text = _build_query(cleaned, target_competency)
    try:
        query_embedding = _as_plain_list(model.encode([query_text], show_progress_bar=False))
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(num_chunks, len(chunks)),
            include=["documents", "metadatas"],
        )
    except Exception as exc:
        logger.warning(
            "[RETRIEVAL] status=failed source=%s error=%s — using all chunks directly", source, exc
        )
        context = "\n\n".join(chunks[:TOP_K_CHUNKS])
        return {
            "context": context,
            "chunk_count": len(chunks),
            "retrieved_chunk_count": len(chunks[:TOP_K_CHUNKS]),
            "collection_name": collection_name,
            "fallback": True,
        }

    retrieved = results.get("documents", [[]])[0]
    context = "\n\n".join(doc for doc in retrieved if doc)
    logger.info(
        "[RETRIEVAL] source=%s query_preview=%r results_count=%d context_length=%d",
        source,
        query_text[:120],
        len([doc for doc in retrieved if doc]),
        len(context),
    )

    if not context.strip():
        raise ValueError(
            "Unable to extract readable text from this PDF. "
            "Please upload a text-based learning material PDF."
        )

    return {
        "context": context,
        "chunk_count": len(chunks),
        "retrieved_chunk_count": len(retrieved),
        "collection_name": collection_name,
        "fallback": False,
    }
