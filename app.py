from __future__ import annotations

import hashlib
import html
from pathlib import Path
import re
import time
import uuid
from typing import Any, Dict, List, Mapping

import pandas as pd
import streamlit as st

from app.llm.client import get_llm_client
from app.rag.answer_state import derive_answer_view_state
from app.runtime import (
    BM25_DIR,
    CHUNKS_PATH,
    QDRANT_COLLECTION,
    QDRANT_STATUS_ENABLED,
    UPLOAD_MD_DIR,
    UPLOAD_PDF_DIR,
    create_qdrant_client,
)

INDEX_DIR = CHUNKS_PATH.parent
DEFAULT_COLLECTION = QDRANT_COLLECTION

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
CHUNK_ID_NAMESPACE = uuid.UUID("49b7b7f2-1291-4ed5-a4b6-b68de66f7b8e")


def _clean_markdown(md: str) -> str:
    md = html.unescape(md)
    md = re.sub(r"^(#{1,6})\s+\1\s+", r"\1 ", md, flags=re.MULTILINE)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def _markdown_chunks(md_text: str) -> List[Dict[str, str]]:
    lines = md_text.splitlines()
    chunks: List[Dict[str, str]] = []
    heading_stack: Dict[int, str] = {}
    current_section = "Document"
    body_lines: List[str] = []

    def flush():
        nonlocal body_lines
        text = "\n".join(body_lines).strip()
        if text:
            chunks.append({"section_path": current_section, "chunk_text": text})
        body_lines = []

    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            heading_stack[level] = title
            for key in [k for k in heading_stack if k > level]:
                del heading_stack[key]
            current_section = " > ".join([heading_stack[k] for k in sorted(heading_stack)])
        else:
            body_lines.append(line)
    flush()
    return chunks


def _stable_chunk_id(doc_id: str, idx: int, section_path: str, chunk_text: str) -> str:
    payload = f"{doc_id}::{idx}::{section_path}::{chunk_text[:160]}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"policy::{digest}"


def _point_id_from_chunk_id(chunk_id: str) -> str:
    return str(uuid.uuid5(CHUNK_ID_NAMESPACE, chunk_id))


def _convert_pdf_to_markdown(pdf_path: Path) -> str:
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    return _clean_markdown(result.document.export_to_markdown())


def _uploaded_to_chunks_df(uploaded_files) -> tuple[pd.DataFrame, int]:
    UPLOAD_PDF_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_MD_DIR.mkdir(parents=True, exist_ok=True)
    records: List[Dict] = []
    docs_indexed = 0

    for uploaded in uploaded_files:
        raw = uploaded.getbuffer()
        doc_id = Path(uploaded.name).stem.lower().replace(" ", "_")
        suffix = Path(uploaded.name).suffix.lower()

        if suffix == ".pdf":
            pdf_path = UPLOAD_PDF_DIR / uploaded.name
            pdf_path.write_bytes(raw)
            md_text = _convert_pdf_to_markdown(pdf_path)
            md_path = UPLOAD_MD_DIR / f"{doc_id}.md"
            md_path.write_text(md_text, encoding="utf-8")
        else:
            md_text = _clean_markdown(raw.tobytes().decode("utf-8", errors="replace"))
            md_path = UPLOAD_MD_DIR / f"{doc_id}.md"
            md_path.write_text(md_text, encoding="utf-8")

        doc_title = doc_id.replace("_", " ").title()
        for idx, chunk in enumerate(_markdown_chunks(md_text), start=1):
            chunk_text = chunk["chunk_text"].strip()
            section_path = chunk["section_path"]
            if not chunk_text:
                continue
            records.append(
                {
                    "chunk_id": _stable_chunk_id(doc_id, idx, section_path, chunk_text),
                    "chunk_text": chunk_text,
                    "source_type": "policy_pdf",
                    "control_id": None,
                    "control_part": None,
                    "enhancement_id": None,
                    "control_family": None,
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "section_path": section_path,
                    "page_start": None,
                    "page_end": None,
                }
            )
        docs_indexed += 1

    return pd.DataFrame(records), docs_indexed


def _upsert_chunks_to_qdrant(chunks_df: pd.DataFrame) -> int:
    if chunks_df.empty:
        return 0
    # Lazy imports reduce first-render startup work for the Streamlit app.
    from qdrant_client.models import PointStruct

    from app.index.qdrant_schema import ensure_collection
    from app.utils.embeddings import embed_texts

    client = create_qdrant_client()
    probe = embed_texts([chunks_df.iloc[0]["chunk_text"]])
    vector_size = int(probe.shape[1])
    ensure_collection(client, DEFAULT_COLLECTION, vector_size)

    batch_size = 64
    total = len(chunks_df)
    for start in range(0, total, batch_size):
        batch = chunks_df.iloc[start : start + batch_size]
        vectors = embed_texts(batch["chunk_text"].astype(str).tolist())
        points: List[PointStruct] = []
        for i, (_, row) in enumerate(batch.iterrows()):
            payload = {k: v for k, v in row.to_dict().items() if v is not None}
            points.append(
                PointStruct(
                    id=_point_id_from_chunk_id(str(row["chunk_id"])),
                    vector=vectors[i].tolist(),
                    payload=payload,
                )
            )
        client.upsert(collection_name=DEFAULT_COLLECTION, points=points, wait=True)
    return total


def _merge_chunks_and_rebuild_bm25(new_chunks_df: pd.DataFrame) -> int:
    from app.index.bm25_index import build_index

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if CHUNKS_PATH.exists():
        base_df = pd.read_parquet(CHUNKS_PATH)
    else:
        base_df = pd.DataFrame(columns=new_chunks_df.columns)

    combined = pd.concat([base_df, new_chunks_df], ignore_index=True)
    combined = combined.dropna(subset=["chunk_id", "chunk_text"])
    combined = combined.drop_duplicates(subset=["chunk_id"]).reset_index(drop=True)
    combined.to_parquet(CHUNKS_PATH, index=False)
    build_index(combined, BM25_DIR)
    return len(combined)


def _count_parquet_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        return int((pf.metadata.num_rows if pf.metadata is not None else 0) or 0)
    except Exception:
        try:
            return len(pd.read_parquet(path, columns=["chunk_id"]))
        except Exception:
            return 0


@st.cache_data(ttl=15, show_spinner=False)
def _status_snapshot() -> Dict[str, int]:
    chunks_count = 0
    try:
        chunks_count = _count_parquet_rows(CHUNKS_PATH)
    except Exception:
        chunks_count = 0

    qdrant_points = 0
    if QDRANT_STATUS_ENABLED:
        try:
            client = create_qdrant_client()
            info = client.get_collection(DEFAULT_COLLECTION)
            qdrant_points = int(getattr(info, "points_count", 0) or 0)
        except Exception:
            qdrant_points = 0

    docs_count = 0
    if UPLOAD_MD_DIR.exists():
        docs_count = len(list(UPLOAD_MD_DIR.glob("*.md")))
    return {"docs_indexed": docs_count, "chunks_indexed": chunks_count, "qdrant_points": qdrant_points}


def _is_timeout_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return isinstance(exc, TimeoutError) or any(
        term in message for term in ("timeout", "timed out", "deadline exceeded", "read timed out")
    )


def _unexpected_backend_result(question: str, exc: Exception) -> Dict[str, Any]:
    error_type = "timeout" if _is_timeout_error(exc) else "error"
    if error_type == "timeout":
        draft = "insufficient evidence. Backend request timed out before a reliable answer was available."
    else:
        draft = "insufficient evidence. Backend request failed before a reliable answer was available."
    return {
        "query": question,
        "draft_answer": draft,
        "content": draft,
        "abstained": True,
        "confidence": 0.0,
        "predicted_coverage": None,
        "citations": [],
        "citations_markdown": "- (none)",
        "retrieved_chunks": [],
        "query_mode": "unknown",
        "weak_retrieval": True,
        "retrieval_status": error_type,
        "retrieval_error_type": error_type,
        "llm_status": "not_requested",
        "llm_error_type": None,
    }


def _assistant_message_payload(result: Mapping[str, Any], citations_markdown: str) -> Dict[str, Any]:
    payload = dict(result)
    payload.update(
        {
            "role": "assistant",
            "content": str(result.get("draft_answer", "")),
            "citations_markdown": citations_markdown,
        }
    )
    return payload


def _format_backend_name(name: str) -> str:
    normalized = str(name or "").strip().lower()
    if normalized == "openrouter":
        return "OpenRouter"
    if normalized == "none":
        return "Disabled"
    if not normalized:
        return "Unknown"
    return normalized


def _configured_llm_mode_caption() -> str:
    info = {}
    try:
        info = dict(get_llm_client().describe_backend())
    except Exception:
        info = {}

    backend = str(info.get("backend") or "unknown")
    mode = str(info.get("mode") or "unknown")
    status = str(info.get("status") or "unknown")
    requested_model = str(info.get("requested_model") or "").strip()
    error_type = str(info.get("error_type") or "").strip()

    if backend == "none" or mode == "retrieval_only" and backend == "none":
        return "Mode: Retrieval only. The app will answer from retrieved evidence without LLM generation."
    if backend == "openrouter" and status == "ready":
        model_suffix = f" Requested model: `{requested_model}`." if requested_model else ""
        return f"Mode: Retrieval + OpenRouter.{model_suffix}"
    if backend == "openrouter":
        issue = f" ({error_type})" if error_type else ""
        model_suffix = f" Requested model: `{requested_model}`." if requested_model else ""
        return f"Mode: Retrieval-only fallback. OpenRouter is selected but unavailable{issue}.{model_suffix}"
    return "Mode: Retrieval behavior may vary based on current backend configuration."


def _response_backend_caption(message: Mapping[str, Any]) -> str:
    backend = _format_backend_name(str(message.get("llm_backend") or "unknown"))
    mode = str(message.get("llm_mode") or "unknown")
    status = str(message.get("llm_status") or "unknown")
    requested_model = str(message.get("llm_requested_model") or "").strip()
    used_model = str(message.get("llm_used_model") or "").strip()
    fallback_triggered = bool(message.get("llm_fallback_triggered"))
    retries = int(message.get("llm_retries") or 0)
    latency_ms = message.get("llm_latency_ms")
    error_type = str(message.get("llm_error_type") or "").strip()

    if mode == "retrieval_only" and backend == "Disabled":
        return "Mode: Retrieval only | LLM disabled"

    parts: List[str] = []
    if mode == "retrieval_plus_llm":
        parts.append(f"Mode: Retrieval + {backend}")
    elif mode == "retrieval_only":
        parts.append("Mode: Retrieval-only fallback")
    else:
        parts.append(f"LLM backend: {backend}")

    if requested_model:
        parts.append(f"Requested model: {requested_model}")
    if used_model:
        parts.append(f"Used model: {used_model}")
    if fallback_triggered:
        parts.append("Model fallback triggered")
    if status == "not_requested":
        parts.append("LLM call skipped")
    elif status and status != "unknown":
        status_label = status
        if error_type:
            status_label += f" ({error_type})"
        parts.append(f"LLM status: {status_label}")
    if retries:
        parts.append(f"Retries: {retries}")
    if latency_ms is not None:
        parts.append(f"Latency: {latency_ms} ms")
    return " | ".join(parts)


def _chunk_meta(chunk: Mapping[str, Any]) -> str:
    source_type = chunk.get("source_type") or "unknown"
    control_id = chunk.get("control_id")
    doc_label = chunk.get("doc_title") or chunk.get("doc_id")
    section = chunk.get("section_path") or "n/a"
    page_start = chunk.get("page_start")
    page_end = chunk.get("page_end")
    parts = [str(source_type)]
    if control_id:
        parts.append(str(control_id))
    if doc_label:
        parts.append(str(doc_label))
    parts.append(str(section))
    if page_start is not None and page_end is not None:
        parts.append(f"pages {page_start}-{page_end}")
    elif page_start is not None:
        parts.append(f"page {page_start}")
    return " | ".join(parts)


def _render_retrieved_evidence(chunks: List[Dict[str, Any]]) -> None:
    evidence = chunks[:10]
    if not evidence:
        st.markdown("No retrieved evidence available.")
        return
    for idx, chunk in enumerate(evidence, start=1):
        chunk_label = chunk.get("citation_id") or f"E{idx}"
        title = (
            chunk.get("doc_title")
            or chunk.get("doc_id")
            or chunk.get("control_id")
            or "Retrieved source"
        )
        text = str(chunk.get("chunk_text", "")).replace("\n", " ").strip()
        preview = text[:500] + ("..." if len(text) > 500 else "")
        st.markdown(f"**{chunk_label} - {title}**")
        st.caption(_chunk_meta(chunk))
        st.code(preview or "(empty)")


def _render_assistant_message(message: Mapping[str, Any]) -> None:
    if message.get("message_type") == "rate_limit":
        st.warning(str(message.get("content", "")))
        return

    view = derive_answer_view_state(message)
    banner = {
        "success": st.success,
        "warning": st.warning,
        "info": st.info,
        "error": st.error,
    }.get(view.tone, st.info)
    banner(f"{view.title}: {view.summary}")
    st.caption(view.support_label)
    st.caption(_response_backend_caption(message))
    if str(message.get("llm_backend") or "").strip().lower() == "openrouter" and str(
        message.get("llm_status") or ""
    ).strip().lower() in {"unavailable", "error", "timeout"}:
        llm_note = str(message.get("llm_error_message") or "").strip()
        if llm_note:
            st.caption(f"LLM note: {llm_note}")

    if view.answer_body:
        st.markdown(f"**{view.answer_label}**")
        st.markdown(view.answer_body)

    if view.state in {"strong_answer", "partial_answer", "conflicting_evidence"}:
        st.caption("Generated explanation is shown above. Retrieved citations and raw evidence are listed below.")
    elif view.state == "retrieval_only":
        st.caption("This response is a retrieval-only fallback. Use the citations and excerpts below as the source of truth.")
    else:
        st.caption("No confident answer is being presented.")

    citations_markdown = str(message.get("citations_markdown", "- (none)"))
    with st.expander("Source citations", expanded=view.state in {"strong_answer", "partial_answer", "conflicting_evidence"}):
        if citations_markdown.strip() == "- (none)":
            st.markdown("No citations available.")
        else:
            st.markdown(citations_markdown)
    with st.expander(
        "Retrieved evidence",
        expanded=view.state in {"partial_answer", "conflicting_evidence", "retrieval_only"},
    ):
        _render_retrieved_evidence(list(message.get("retrieved_chunks", []) or []))


st.set_page_config(page_title="RMF Assistant", page_icon=":shield:", layout="wide")
st.title("RMF Assistant")
st.caption("Grounded question answering over indexed OSCAL controls and uploaded policy documents.")
st.caption(_configured_llm_mode_caption())

with st.sidebar:
    st.header("Data")
    uploads = st.file_uploader(
        "Upload policy PDFs/MD",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
    )
    if uploads:
        st.success(f"Selected {len(uploads)} files.")

    if st.button("Ingest Uploaded Files", type="primary", disabled=not uploads):
        try:
            with st.spinner("Converting, chunking, embedding, and indexing..."):
                new_chunks_df, docs_count = _uploaded_to_chunks_df(uploads)
                upserted = _upsert_chunks_to_qdrant(new_chunks_df)
                total_chunks = _merge_chunks_and_rebuild_bm25(new_chunks_df)
            st.success(
                f"Ingest complete: docs={docs_count}, new_chunks={len(new_chunks_df)}, "
                f"qdrant_upserted={upserted}, total_chunks={total_chunks}"
            )
        except Exception as exc:
            st.error(f"Ingest failed: {exc}")

    st.header("Corpus")
    if "show_corpus_status" not in st.session_state:
        st.session_state.show_corpus_status = False
    load_status = st.button("Load Corpus Status")
    refresh_status = st.button("Refresh Corpus Status", disabled=not st.session_state.show_corpus_status)
    if load_status:
        st.session_state.show_corpus_status = True
    if refresh_status:
        _status_snapshot.clear()
    if st.session_state.show_corpus_status:
        status = _status_snapshot()
        st.metric("Docs Indexed", status["docs_indexed"])
        st.metric("Chunks Indexed (local)", status["chunks_indexed"])
        st.metric("Qdrant Points", status["qdrant_points"])
        if not QDRANT_STATUS_ENABLED:
            st.caption("Qdrant status probe disabled for fast startup. Set QDRANT_STATUS_ENABLED=1 to enable.")
        elif status["qdrant_points"] == 0:
            st.caption("Qdrant count unavailable or collection empty.")
    else:
        st.caption("Corpus status is deferred for fast startup. Click 'Load Corpus Status' when needed.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_llm_call_ts" not in st.session_state:
    st.session_state.last_llm_call_ts = 0.0

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            _render_assistant_message(msg)
        else:
            st.markdown(msg["content"])

user_question = st.chat_input("Ask about RMF controls or policy evidence...")
if user_question:
    from app.rag.answer import answer_question
    from app.rag.citations import format_citations_markdown

    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    now = time.monotonic()
    seconds_since_last = now - float(st.session_state.last_llm_call_ts)
    if seconds_since_last < 3.0:
        wait_for = max(0.0, 3.0 - seconds_since_last)
        rate_msg = (
            f"Rate limit: please wait {wait_for:.1f}s before sending another question."
        )
        with st.chat_message("assistant"):
            st.warning(rate_msg)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "message_type": "rate_limit",
                "content": rate_msg,
                "citations_markdown": "- (none)",
                "retrieved_chunks": [],
                "confidence": 0.0,
                "abstained": True,
            }
        )
        st.stop()

    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence and preparing response..."):
            try:
                result = answer_question(
                    user_question,
                    top_k=10,
                )
            except Exception as exc:
                result = _unexpected_backend_result(user_question, exc)
        st.session_state.last_llm_call_ts = time.monotonic()
        citations_markdown = format_citations_markdown(result.get("citations", []))
        assistant_message = _assistant_message_payload(result, citations_markdown)
        _render_assistant_message(assistant_message)

    st.session_state.messages.append(assistant_message)
