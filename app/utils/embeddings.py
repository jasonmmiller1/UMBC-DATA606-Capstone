from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


_EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


@lru_cache(maxsize=2)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(_EMBED_MODEL_NAME)


def embed_texts(texts: List[str]) -> np.ndarray:
    text_list: List[str] = [t if isinstance(t, str) else str(t) for t in texts]
    if not text_list:
        return np.empty((0, 0), dtype=np.float32)
    model = _get_model()
    vectors = model.encode(
        text_list,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.astype(np.float32)
