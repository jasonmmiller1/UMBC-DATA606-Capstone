from __future__ import annotations

import hashlib
import html
import os
from pathlib import Path
import re
import subprocess
import time
import uuid
from typing import Dict, List

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
UPLOAD_PDF_DIR = DATA_DIR / "uploads_pdf"
UPLOAD_MD_DIR = DATA_DIR / "uploads_md"
INDEX_DIR = DATA_DIR / "index"
CHUNKS_PATH = INDEX_DIR / "chunks.parquet"
BM25_DIR = DATA_DIR / "bm25_index"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_TIMEOUT = float(os.getenv("QDRANT_TIMEOUT", "0.75"))
QDRANT_STATUS_ENABLED = os.getenv("QDRANT_STATUS_ENABLED", "0").strip().lower() not in {"0", "false", "no"}
DEFAULT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rmf_chunks")

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
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct

    from app.index.qdrant_schema import ensure_collection
    from app.utils.embeddings import embed_texts

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=QDRANT_TIMEOUT)
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
            from qdrant_client import QdrantClient

            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=QDRANT_TIMEOUT)
            info = client.get_collection(DEFAULT_COLLECTION)
            qdrant_points = int(getattr(info, "points_count", 0) or 0)
        except Exception:
            qdrant_points = 0

    docs_count = 0
    if UPLOAD_MD_DIR.exists():
        docs_count = len(list(UPLOAD_MD_DIR.glob("*.md")))
    return {"docs_indexed": docs_count, "chunks_indexed": chunks_count, "qdrant_points": qdrant_points}


st.set_page_config(page_title="RMF Assistant - Week 3 MVP", page_icon=":shield:", layout="wide")
st.title("RMF Assistant - Week 3 RAG Chat MVP")
st.caption("Evidence-grounded QA over indexed OSCAL + synthetic policy corpus")

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

    if st.button("Run Week 2 Retrieval Tests"):
        with st.spinner("Running Week 2 retrieval suite..."):
            proc = subprocess.run(
                ["python", "-m", "app.retrieval.run_week2_tests"],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
            )
        if proc.returncode == 0:
            st.success("Week 2 retrieval suite completed.")
        else:
            st.error("Week 2 retrieval suite failed.")
        with st.expander("Week 2 suite logs"):
            if proc.stdout:
                st.code(proc.stdout)
            if proc.stderr:
                st.code(proc.stderr)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_llm_call_ts" not in st.session_state:
    st.session_state.last_llm_call_ts = 0.0

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            st.caption(f"confidence={msg.get('confidence', 0.0)} | abstained={msg.get('abstained', False)}")
            with st.expander("Citations"):
                st.markdown(msg.get("citations_markdown", "- (none)"))
            with st.expander("Retrieved Evidence"):
                chunks = msg.get("retrieved_chunks", [])[:10]
                if not chunks:
                    st.markdown("- (none)")
                for idx, chunk in enumerate(chunks, start=1):
                    source_type = chunk.get("source_type") or "unknown"
                    control_id = chunk.get("control_id")
                    doc_id = chunk.get("doc_id")
                    section = chunk.get("section_path") or "n/a"
                    page_start = chunk.get("page_start")
                    page_end = chunk.get("page_end")
                    page = ""
                    if page_start is not None and page_end is not None:
                        page = f" p.{page_start}-{page_end}"
                    elif page_start is not None:
                        page = f" p.{page_start}"
                    meta = f"{source_type}"
                    if control_id:
                        meta += f" | {control_id}"
                    if doc_id:
                        meta += f" | {doc_id}"
                    meta += f" | {section}{page}"
                    text = str(chunk.get("chunk_text", "")).replace("\n", " ").strip()
                    preview = text[:500] + ("..." if len(text) > 500 else "")
                    st.markdown(f"{idx}. `{chunk.get('chunk_id', 'unknown')}`")
                    st.caption(meta)
                    st.code(preview or "(empty)")

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
                "content": rate_msg,
                "citations_markdown": "- (none)",
                "retrieved_chunks": [],
                "confidence": 0.0,
                "abstained": True,
            }
        )
        st.stop()

    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence..."):
            result = answer_question(
                user_question,
                top_k=10,
            )
        st.session_state.last_llm_call_ts = time.monotonic()
        st.markdown(result["draft_answer"])
        citations_markdown = format_citations_markdown(result.get("citations", []))
        st.caption(f"confidence={result['confidence']} | abstained={result['abstained']}")
        with st.expander("Citations"):
            st.markdown(citations_markdown)
        with st.expander("Retrieved Evidence"):
            chunks = result.get("retrieved_chunks", [])[:10]
            if not chunks:
                st.markdown("- (none)")
            for idx, chunk in enumerate(chunks, start=1):
                source_type = chunk.get("source_type") or "unknown"
                control_id = chunk.get("control_id")
                doc_id = chunk.get("doc_id")
                section = chunk.get("section_path") or "n/a"
                page_start = chunk.get("page_start")
                page_end = chunk.get("page_end")
                page = ""
                if page_start is not None and page_end is not None:
                    page = f" p.{page_start}-{page_end}"
                elif page_start is not None:
                    page = f" p.{page_start}"
                meta = f"{source_type}"
                if control_id:
                    meta += f" | {control_id}"
                if doc_id:
                    meta += f" | {doc_id}"
                meta += f" | {section}{page}"
                text = str(chunk.get("chunk_text", "")).replace("\n", " ").strip()
                preview = text[:500] + ("..." if len(text) > 500 else "")
                st.markdown(f"{idx}. `{chunk.get('chunk_id', 'unknown')}`")
                st.caption(meta)
                st.code(preview or "(empty)")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["draft_answer"],
            "citations_markdown": citations_markdown,
            "retrieved_chunks": result.get("retrieved_chunks", []),
            "confidence": result["confidence"],
            "abstained": result["abstained"],
        }
    )
