from __future__ import annotations

import argparse
import math
from pathlib import Path
import pickle
import re
from typing import Dict, Iterable, List, Tuple

import pandas as pd


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


class LocalBM25:
    def __init__(self, tokenized_docs: List[List[str]], chunk_ids: List[str], k1: float = 1.5, b: float = 0.75):
        self.tokenized_docs = tokenized_docs
        self.chunk_ids = chunk_ids
        self.k1 = float(k1)
        self.b = float(b)
        self.doc_lens = [len(doc) for doc in tokenized_docs]
        self.avg_doc_len = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 0.0

        df: Dict[str, int] = {}
        tf_docs: List[Dict[str, int]] = []
        for doc in tokenized_docs:
            freqs: Dict[str, int] = {}
            for tok in doc:
                freqs[tok] = freqs.get(tok, 0) + 1
            tf_docs.append(freqs)
            for tok in freqs:
                df[tok] = df.get(tok, 0) + 1
        self.tf_docs = tf_docs
        self.df = df
        self.n_docs = len(tokenized_docs)

    def _idf(self, term: str) -> float:
        n_q = self.df.get(term, 0)
        if n_q == 0:
            return 0.0
        return math.log(1 + (self.n_docs - n_q + 0.5) / (n_q + 0.5))

    def query(self, query_text: str, top_k: int = 20) -> List[Tuple[str, float]]:
        q_terms = tokenize(query_text)
        if not q_terms or self.n_docs == 0:
            return []

        scores: List[float] = [0.0] * self.n_docs
        for idx, tf in enumerate(self.tf_docs):
            dl = self.doc_lens[idx] or 1
            for term in q_terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                idf = self._idf(term)
                denom = f + self.k1 * (1 - self.b + self.b * (dl / max(self.avg_doc_len, 1e-9)))
                scores[idx] += idf * ((f * (self.k1 + 1)) / max(denom, 1e-9))

        ranked = sorted(
            [(self.chunk_ids[i], s) for i, s in enumerate(scores) if s > 0.0],
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]


def build_index(chunks_df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_ids = chunks_df["chunk_id"].astype(str).tolist()
    tokenized_docs = [tokenize(t) for t in chunks_df["chunk_text"].astype(str).tolist()]
    index = LocalBM25(tokenized_docs=tokenized_docs, chunk_ids=chunk_ids)

    out_path = out_dir / "bm25_index.pkl"
    with out_path.open("wb") as f:
        # Persist plain data so loading is stable across module execution contexts.
        pickle.dump(
            {
                "tokenized_docs": tokenized_docs,
                "chunk_ids": chunk_ids,
                "k1": index.k1,
                "b": index.b,
            },
            f,
        )
    return out_path


def load_index(index_path: Path) -> LocalBM25:
    with index_path.open("rb") as f:
        data = pickle.load(f)
    if isinstance(data, LocalBM25):
        return data
    if isinstance(data, dict) and "tokenized_docs" in data and "chunk_ids" in data:
        return LocalBM25(
            tokenized_docs=data["tokenized_docs"],
            chunk_ids=data["chunk_ids"],
            k1=float(data.get("k1", 1.5)),
            b=float(data.get("b", 0.75)),
        )
    raise ValueError(f"Unsupported BM25 index format: {index_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or query local BM25 index.")
    parser.add_argument("--chunks-path", default="data/index/chunks.parquet")
    parser.add_argument("--index-dir", default="data/bm25_index")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--query", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    chunks_path = repo_root / args.chunks_path
    index_dir = repo_root / args.index_dir
    index_path = index_dir / "bm25_index.pkl"

    if args.build:
        if not chunks_path.exists():
            raise FileNotFoundError(f"Missing chunks parquet: {chunks_path}")
        chunks_df = pd.read_parquet(chunks_path)
        written = build_index(chunks_df, index_dir)
        print(f"Wrote BM25 index: {written}")
        return

    if not args.query:
        raise ValueError("Use --build to create index, or provide --query for retrieval.")
    if not index_path.exists():
        raise FileNotFoundError(f"BM25 index not found: {index_path}. Run with --build first.")

    bm25 = load_index(index_path)
    ranked = bm25.query(args.query, top_k=args.top_k)
    for rank, (chunk_id, score) in enumerate(ranked, start=1):
        print(f"{rank:2d}. {chunk_id} score={score:.4f}")


if __name__ == "__main__":
    main()
