from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


MODEL_NAME = "jhgan/ko-sroberta-multitask"


class TextEncoder(Protocol):
    backend_name: str

    def encode(self, texts: list[str]) -> np.ndarray:
        ...


def build_shop_document(row: pd.Series) -> str:
    fields = [
        str(row.get("shop_name", "")),
        str(row.get("categories", "")),
        str(row.get("awards", "")),
    ]
    return " ".join(value for value in fields if value and value != "nan").strip()


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


@dataclass
class HashingTextEncoder:
    dimensions: int = 512
    backend_name: str = "hashing"

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            for token in self._tokens(text):
                digest = hashlib.md5(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dimensions
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vectors[row_idx, bucket] += sign
        return _normalize_rows(vectors)

    def _tokens(self, text: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", str(text).lower()).strip()
        words = re.findall(r"[0-9a-zA-Z가-힣]+", normalized)
        char_ngrams: list[str] = []
        for word in words:
            if len(word) <= 2:
                char_ngrams.append(word)
                continue
            char_ngrams.extend(word[i : i + 2] for i in range(len(word) - 1))
            char_ngrams.extend(word[i : i + 3] for i in range(len(word) - 2))
        return words + char_ngrams


class SentenceTransformerEncoder:
    backend_name = "sentence-transformers"

    def __init__(self, model_name: str = MODEL_NAME):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
        ).astype(np.float32)


def load_text_encoder(backend: str = "auto") -> TextEncoder:
    if backend == "hashing":
        return HashingTextEncoder()
    if backend in {"auto", "sentence-transformers"}:
        try:
            return SentenceTransformerEncoder()
        except Exception:
            if backend == "sentence-transformers":
                raise
            return HashingTextEncoder()
    raise ValueError(f"Unsupported embedding backend: {backend}")


def encode_shops(shops: pd.DataFrame, encoder: TextEncoder) -> np.ndarray:
    docs = shops.apply(build_shop_document, axis=1).tolist()
    return encoder.encode(docs)


def encode_query(query: str, encoder: TextEncoder) -> np.ndarray:
    return encoder.encode([query])[0]
