from __future__ import annotations

import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import settings
from src.recommender.query_parser import encode_query, load_text_encoder
from src.recommender.ranker import rank_candidates


class RecommendationEngine:
    def __init__(
        self,
        index_dir: str | Path = settings.processed_data_dir,
        embedding_backend: str = settings.embedding_backend,
    ):
        index_path = Path(index_dir)
        self.index_metadata = _load_index_metadata(index_path)
        index_backend = self.index_metadata.get("embedding_backend")
        runtime_backend = embedding_backend
        if embedding_backend == "auto" and index_backend:
            runtime_backend = str(index_backend)
        elif index_backend and embedding_backend != index_backend:
            raise ValueError(
                "Embedding backend mismatch: "
                f"index was built with '{index_backend}', "
                f"but server is using '{embedding_backend}'. "
                "Rebuild the index with the same backend or update EMBEDDING_BACKEND."
            )

        self.encoder = load_text_encoder(runtime_backend)
        self.shop_embeddings = np.load(index_path / "shop_embeddings.npy")
        self.shops = pd.read_pickle(index_path / "shops_indexed.pkl")

        with open(index_path / "behavior_scores.pkl", "rb") as f:
            behavior_payload = pickle.load(f)

        if "query_scores" in behavior_payload:
            self.behavior_scores = behavior_payload.get("query_scores", {})
            self.global_scores = behavior_payload.get("global_scores", {})
        else:
            self.behavior_scores = behavior_payload
            self.global_scores = {}

        self.shops = self.shops.assign(shop_id=self.shops["shop_id"].astype(str))
        self.shop_ids = self.shops["shop_id"].tolist()
        self.shops_by_id = self.shops.set_index("shop_id", drop=False)

    def recommend(self, query: str, top_k: int = 5, mode: str = "hybrid") -> list[dict]:
        if not query.strip():
            return []

        query_embedding = encode_query(query, self.encoder)
        if self.shop_embeddings.shape[1] != query_embedding.shape[0]:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"shop index has {self.shop_embeddings.shape[1]} dimensions, "
                f"but query embedding has {query_embedding.shape[0]} dimensions. "
                "Rebuild the index or use the embedding backend recorded in index_metadata.json."
            )

        behavior_by_shop = self.behavior_scores.get(query, {})
        ranked = rank_candidates(
            query_embedding=query_embedding,
            shop_embeddings=self.shop_embeddings,
            shop_ids=self.shop_ids,
            behavior_by_shop=behavior_by_shop,
            global_behavior_by_shop=self.global_scores,
            top_k=top_k,
            mode=mode,
        )

        results = []
        for row in ranked:
            meta = self.shops_by_id.loc[row["shop_id"]].to_dict()
            results.append(
                {
                    **row,
                    "shop_name": _clean(meta.get("shop_name")),
                    "address": _clean(meta.get("address")),
                    "categories": _clean(meta.get("categories")),
                    "menus": _clean(meta.get("menus")),
                    "facilities": _clean(meta.get("facilities")),
                    "awards": _clean(meta.get("awards")),
                    "rank_reason": make_rank_reason(row, meta),
                }
            )
        return results


def _clean(value):
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    return text or None


def _load_index_metadata(index_path: Path) -> dict:
    metadata_path = index_path / "index_metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def make_rank_reason(score_row: dict, meta: dict) -> str:
    categories = _clean(meta.get("categories"))
    awards = _clean(meta.get("awards"))
    if score_row["behavior_score"] >= 0.7:
        return "동일 검색어에서 예약, 북마크, 상세 조회 행동 점수가 높습니다."
    if awards and score_row["semantic_score"] >= 0.6:
        return "검색어와 식당 카테고리 및 수상 정보의 의미 유사도가 높습니다."
    if categories and score_row["semantic_score"] >= 0.6:
        return "검색어와 식당 카테고리의 의미 유사도가 높습니다."
    return "검색어 의미와 전체 행동 로그 점수를 종합해 상위로 선정되었습니다."
