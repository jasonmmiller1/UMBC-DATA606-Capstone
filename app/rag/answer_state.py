from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence


INLINE_CITATION_RE = re.compile(r"\[C\d+\]")
CONFLICT_TEXT_MARKERS = (
    "conflicting",
    "conflicting evidence",
    "conflicting sources",
    "sources conflict",
    "contradict",
    "inconsistent",
    "do not agree",
)
CONFLICT_PHRASE_PAIRS = (
    ("shall ", "shall not"),
    ("must ", "must not"),
    ("required", "not required"),
    ("allowed", "prohibited"),
    ("enabled", "disabled"),
)
TIMEOUT_TERMS = ("timeout", "timed out", "deadline exceeded", "read timed out", "connect timeout")


@dataclass(frozen=True)
class AnswerViewState:
    state: str
    title: str
    summary: str
    tone: str
    answer_label: str
    answer_body: str
    support_label: str


def _normalized_text(value: Any) -> str:
    return str(value or "").strip()


def _contains_inline_citations(text: str) -> bool:
    return bool(INLINE_CITATION_RE.search(text or ""))


def _has_conflict_markers(text: str) -> bool:
    lowered = _normalized_text(text).lower()
    return any(marker in lowered for marker in CONFLICT_TEXT_MARKERS)


def _chunk_texts(chunks: Sequence[Mapping[str, Any]]) -> list[str]:
    texts: list[str] = []
    for chunk in chunks[:6]:
        text = _normalized_text(chunk.get("chunk_text"))
        if text:
            texts.append(text.lower())
    return texts


def _has_conflicting_chunk_phrases(chunks: Sequence[Mapping[str, Any]]) -> bool:
    texts = _chunk_texts(chunks)
    if len(texts) < 2:
        return False
    for positive, negative in CONFLICT_PHRASE_PAIRS:
        positive_hits = [idx for idx, text in enumerate(texts) if positive in text]
        negative_hits = [idx for idx, text in enumerate(texts) if negative in text]
        if positive_hits and negative_hits and set(positive_hits) != set(negative_hits):
            return True
    return False


def _extract_between(text: str, start_marker: str, end_marker: str) -> str:
    body = _normalized_text(text)
    if not body:
        return ""
    start_idx = body.find(start_marker)
    if start_idx < 0:
        return ""
    start_idx += len(start_marker)
    end_idx = body.find(end_marker, start_idx)
    if end_idx < 0:
        end_idx = len(body)
    return body[start_idx:end_idx].strip()


def _display_answer_body(raw_answer: str, *, state: str, llm_status: str, abstained: bool) -> str:
    body = _normalized_text(raw_answer)
    if not body:
        return ""

    if state == "retrieval_only":
        prefix = "LLM temporarily unavailable. Showing retrieved evidence excerpts."
        if body.startswith(prefix):
            return body[len(prefix) :].strip()
        return body

    if llm_status == "ok" and not abstained and not body.startswith("Coverage:"):
        evidence_only = _extract_between(body, "Evidence:", "Citations:")
        if evidence_only:
            return evidence_only
    return body


def _support_label(state: str, citation_count: int, retrieved_count: int) -> str:
    strength = {
        "strong_answer": "strong",
        "partial_answer": "limited",
        "conflicting_evidence": "conflicting",
        "retrieval_only": "retrieval only",
        "no_evidence": "none",
        "backend_error": "unavailable",
    }.get(state, "limited")
    return (
        f"Evidence support: {strength} | "
        f"{citation_count} citation(s) | {retrieved_count} retrieved excerpt(s)"
    )


def _is_timeout(result: Mapping[str, Any]) -> bool:
    for key in ("retrieval_error_type", "llm_error_type"):
        if _normalized_text(result.get(key)).lower() == "timeout":
            return True
    combined = " ".join(
        [
            _normalized_text(result.get("draft_answer")),
            _normalized_text(result.get("retrieval_status")),
            _normalized_text(result.get("llm_status")),
        ]
    ).lower()
    return any(term in combined for term in TIMEOUT_TERMS)


def _llm_issue_summary(error_type: str) -> str:
    issue = (error_type or "").strip().lower()
    return {
        "auth": "OpenRouter authentication failed.",
        "missing_api_key": "OpenRouter is selected, but no API key is configured.",
        "configuration": "OpenRouter is selected, but it is not fully configured.",
        "timeout": "OpenRouter timed out.",
        "rate_limit": "OpenRouter hit a rate limit.",
        "quota": "OpenRouter hit a quota or credit limit.",
        "model_error": "OpenRouter could not use the requested model.",
        "provider_error": "OpenRouter returned a provider error.",
        "invalid_request": "OpenRouter rejected the request.",
        "backend": "OpenRouter request failed.",
        "backend_error": "OpenRouter request failed.",
        "unexpected_failure": "OpenRouter failed unexpectedly.",
    }.get(issue, "OpenRouter request failed.")


def derive_answer_view_state(result: Mapping[str, Any]) -> AnswerViewState:
    draft_answer = _normalized_text(result.get("draft_answer"))
    citations = result.get("citations") or []
    retrieved_chunks = result.get("retrieved_chunks") or []
    llm_status = _normalized_text(result.get("llm_status")).lower() or "unknown"
    retrieval_status = _normalized_text(result.get("retrieval_status")).lower() or "unknown"
    abstained = bool(result.get("abstained"))
    confidence = float(result.get("confidence") or 0.0)
    weak_retrieval = bool(result.get("weak_retrieval"))
    citation_count = len(citations)
    retrieved_count = len(retrieved_chunks)
    llm_error_type = _normalized_text(result.get("llm_error_type")).lower()
    missing_inline_citations = llm_status == "ok" and not abstained and not _contains_inline_citations(draft_answer)
    conflicting = _has_conflict_markers(draft_answer) or _has_conflicting_chunk_phrases(retrieved_chunks)
    timeout = _is_timeout(result)

    if (retrieval_status in {"error", "timeout"} and retrieved_count == 0) or (
        llm_status in {"error", "timeout"} and retrieved_count == 0
    ):
        title = "Backend timeout" if timeout else "Backend error"
        summary = (
            "The request timed out before a reliable answer was available."
            if timeout
            else "The request could not be completed, so no reliable answer was shown."
        )
        if llm_status in {"error", "timeout"} and llm_error_type:
            summary += f" {_llm_issue_summary(llm_error_type)}"
        answer_body = _display_answer_body(
            draft_answer,
            state="backend_error",
            llm_status=llm_status,
            abstained=abstained,
        )
        return AnswerViewState(
            state="backend_error",
            title=title,
            summary=summary,
            tone="error",
            answer_label="System response",
            answer_body=answer_body,
            support_label=_support_label("backend_error", citation_count, retrieved_count),
        )

    if llm_status in {"unavailable", "error", "timeout"} and retrieved_count > 0:
        if llm_status == "unavailable":
            if llm_error_type in {"missing_api_key", "configuration"}:
                summary = _llm_issue_summary(llm_error_type) + " Retrieved evidence is available, so the app is showing a retrieval-only fallback."
            else:
                summary = "Retrieved evidence is available, but answer generation is not configured right now."
        elif timeout:
            summary = "Retrieved evidence is available, but answer generation timed out."
        else:
            summary = _llm_issue_summary(llm_error_type) + " Retrieved evidence is available, so the app is showing a retrieval-only fallback."
        if weak_retrieval or citation_count == 0:
            summary += " Treat the evidence below as limited."
        answer_body = _display_answer_body(
            draft_answer,
            state="retrieval_only",
            llm_status=llm_status,
            abstained=abstained,
        )
        return AnswerViewState(
            state="retrieval_only",
            title="Retrieval-only fallback",
            summary=summary,
            tone="info",
            answer_label="Retrieved evidence summary",
            answer_body=answer_body,
            support_label=_support_label("retrieval_only", citation_count, retrieved_count),
        )

    if retrieval_status in {"missing_assets", "no_evidence"} or retrieved_count == 0:
        if retrieval_status == "missing_assets":
            summary = "The indexed corpus is not available yet, so the system is abstaining."
        else:
            summary = "No direct supporting evidence was found in the current corpus, so the system is abstaining."
        answer_body = _display_answer_body(
            draft_answer,
            state="no_evidence",
            llm_status=llm_status,
            abstained=abstained,
        )
        return AnswerViewState(
            state="no_evidence",
            title="No direct evidence found",
            summary=summary,
            tone="warning",
            answer_label="System response",
            answer_body=answer_body,
            support_label=_support_label("no_evidence", citation_count, retrieved_count),
        )

    if conflicting:
        answer_body = _display_answer_body(
            draft_answer,
            state="conflicting_evidence",
            llm_status=llm_status,
            abstained=abstained,
        )
        return AnswerViewState(
            state="conflicting_evidence",
            title="Conflicting evidence found",
            summary="Retrieved sources point in different directions, so the answer should be treated cautiously.",
            tone="warning",
            answer_label="System response" if abstained else "Generated explanation",
            answer_body=answer_body,
            support_label=_support_label("conflicting_evidence", citation_count, retrieved_count),
        )

    limited_support = (
        abstained
        or weak_retrieval
        or citation_count == 0
        or missing_inline_citations
        or confidence < 0.55
    )
    if limited_support:
        if citation_count == 0:
            summary = "Related material was retrieved, but citations are missing, so the answer is not fully traceable."
        elif missing_inline_citations:
            summary = "The explanation was downgraded because it is not clearly grounded in inline citations."
        elif abstained:
            summary = "The system found some related material, but not enough to answer confidently."
        else:
            summary = "The answer is only partially supported by the retrieved evidence."
        answer_body = _display_answer_body(
            draft_answer,
            state="partial_answer",
            llm_status=llm_status,
            abstained=abstained,
        )
        return AnswerViewState(
            state="partial_answer",
            title="Limited evidence",
            summary=summary,
            tone="warning",
            answer_label="System response" if abstained else "Generated explanation",
            answer_body=answer_body,
            support_label=_support_label("partial_answer", citation_count, retrieved_count),
        )

    answer_body = _display_answer_body(
        draft_answer,
        state="strong_answer",
        llm_status=llm_status,
        abstained=abstained,
    )
    return AnswerViewState(
        state="strong_answer",
        title="Strong supporting evidence",
        summary="The explanation is grounded in retrieved evidence and traceable citations.",
        tone="success",
        answer_label="Generated explanation",
        answer_body=answer_body,
        support_label=_support_label("strong_answer", citation_count, retrieved_count),
    )
