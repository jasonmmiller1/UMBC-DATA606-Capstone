from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional, Sequence


_CONTROL_BASE_RE = re.compile(r"\b([A-Z]{2,3}-\d{1,3})\b")
_COVERAGE_LABELS = {"covered", "partial", "missing", "unknown", "abstain"}
_NULL_LABELS = {"", "none", "null", "n/a", "na"}
_NEAR_MATCH_SCORES = {
    ("covered", "partial"): 0.5,
    ("partial", "covered"): 0.5,
    ("missing", "unknown"): 0.5,
    ("unknown", "missing"): 0.5,
}


def _as_payload(citation: Dict[str, Any]) -> Dict[str, Any]:
    payload = citation.get("payload")
    if isinstance(payload, dict):
        merged = dict(payload)
        for key, value in citation.items():
            if key not in merged:
                merged[key] = value
        return merged
    return dict(citation)


def _as_list(value: Any) -> Sequence[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _norm_control_id(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    if not text:
        return ""
    match = _CONTROL_BASE_RE.search(text)
    return match.group(1) if match else text


def _norm_doc_id(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if text.endswith(".md"):
        text = text[:-3]
    return text


def _norm_coverage_label(value: Any) -> Optional[str]:
    if isinstance(value, dict):
        for key in ("coverage", "predicted_coverage", "expected_coverage", "label"):
            if key in value:
                return _norm_coverage_label(value.get(key))
        return None

    if value is None:
        return None

    text = str(value).strip().lower()
    if text in _NULL_LABELS:
        return None
    if text in _COVERAGE_LABELS:
        return text

    # Parse free-form answer text in deterministic priority order.
    if "insufficient evidence" in text or "abstain" in text or "out of scope" in text:
        return "abstain"
    if "not covered" in text or "no policy evidence" in text or "no evidence" in text:
        return "missing"
    if re.search(r"\bpartial(?:ly)?\b", text):
        return "partial"
    if re.search(r"\bmissing\b", text):
        return "missing"
    if re.search(r"\bunknown\b|cannot determine|unclear|undetermined", text):
        return "unknown"
    if re.search(r"\bcovered\b|fully addressed|fully meets|\bmeets\b", text):
        return "covered"
    return None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def score_context_precision(expected: Dict[str, Any], citations: Sequence[Dict[str, Any]], intent: str) -> float:
    """
    Deterministic citation precision score in [0, 1].

    Rules:
    - Precision = relevant_citations / total_citations.
    - Relevance is based on expected control IDs and/or policy doc IDs.
    - Additional type-coverage factor penalizes mixed queries that only hit one source type.
    - If no expected targets (typical abstain set), score is 1 only when no citations are present.
    """
    expected = expected or {}
    citations = list(citations or [])
    _ = intent  # reserved for future deterministic intent-specific rules

    expected_controls = {
        _norm_control_id(value)
        for value in _as_list(expected.get("expected_control_ids"))
        if _norm_control_id(value)
    }
    expected_docs = {
        _norm_doc_id(value)
        for value in _as_list(expected.get("expected_policy_doc_ids"))
        if _norm_doc_id(value)
    }

    if not expected_controls and not expected_docs:
        return 1.0 if not citations else 0.0
    if not citations:
        return 0.0

    relevant = 0
    hit_control = False
    hit_doc = False

    for citation in citations:
        payload = _as_payload(citation)
        control_id = _norm_control_id(payload.get("control_id"))
        doc_id = _norm_doc_id(payload.get("doc_id"))

        is_relevant = False
        if control_id and control_id in expected_controls:
            is_relevant = True
            hit_control = True
        if doc_id and doc_id in expected_docs:
            is_relevant = True
            hit_doc = True

        if is_relevant:
            relevant += 1

    precision = relevant / float(len(citations))

    required_types = 0
    matched_types = 0
    if expected_controls:
        required_types += 1
        if hit_control:
            matched_types += 1
    if expected_docs:
        required_types += 1
        if hit_doc:
            matched_types += 1

    type_coverage_factor = (matched_types / float(required_types)) if required_types else 1.0
    return round(precision * type_coverage_factor, 4)


def score_coverage_accuracy(expected_coverage: Any, predicted_text_or_field: Any) -> float:
    """
    Deterministic coverage-label score in [0, 1].

    - Exact label match: 1.0
    - Known near-miss pairs: 0.5
    - Otherwise: 0.0
    - If expected coverage is null/NA, returns 1.0 only when prediction is also null/NA.
    """
    expected_label = _norm_coverage_label(expected_coverage)
    predicted_label = _norm_coverage_label(predicted_text_or_field)

    if expected_label is None:
        return 1.0 if predicted_label is None else 0.0
    if predicted_label is None:
        return 0.0
    if expected_label == predicted_label:
        return 1.0
    return float(_NEAR_MATCH_SCORES.get((expected_label, predicted_label), 0.0))


def score_abstention(
    expected_intent: str,
    expected_coverage: Any,
    abstained_flag: Any,
    citations: Iterable[Dict[str, Any]],
) -> float:
    """
    Deterministic abstention score in [0, 1].

    Should abstain when:
    - intent contains 'abstain', or
    - expected coverage label is 'abstain'

    Scoring:
    - Should abstain:
      - abstained + no citations: 1.0
      - abstained + citations: 0.8
      - did not abstain: 0.0
    - Should not abstain:
      - abstained: 0.0
      - not abstained + citations: 1.0
      - not abstained + no citations: 0.6
    """
    intent = (expected_intent or "").strip().lower()
    expected_label = _norm_coverage_label(expected_coverage)
    should_abstain = ("abstain" in intent) or (expected_label == "abstain")

    abstained = _to_bool(abstained_flag)
    citation_count = len(list(citations or []))

    if should_abstain:
        if not abstained:
            return 0.0
        return 1.0 if citation_count == 0 else 0.8

    if abstained:
        return 0.0
    return 1.0 if citation_count > 0 else 0.6


__all__ = [
    "score_context_precision",
    "score_coverage_accuracy",
    "score_abstention",
]
