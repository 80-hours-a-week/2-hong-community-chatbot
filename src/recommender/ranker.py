from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


EVENT_WEIGHTS = {
    "impression": 0.1,
    "click": 1.0,
    "view": 2.0,
    "bookmark": 5.0,
    "reservation": 10.0,
}


def build_behavior_scores(logs: pd.DataFrame) -> dict[str, object]:
    df = logs.dropna(subset=["search_query", "shop_id"]).copy()
    df["event_weight"] = df["event_type"].map(EVENT_WEIGHTS).fillna(0.0)

    query_scores = _build_query_scores(df)
    global_scores = _build_global_scores(df)
    return {
        "query_scores": query_scores,
        "global_scores": global_scores,
        "event_weights": EVENT_WEIGHTS,
    }


def _build_query_scores(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    grouped = (
        df.groupby(["search_query", "shop_id"], as_index=False)["event_weight"]
        .sum()
        .rename(columns={"event_weight": "raw_behavior_score"})
    )
    if grouped.empty:
        return {}

    query_group = grouped.groupby("search_query")["raw_behavior_score"]
    min_v = query_group.transform("min")
    max_v = query_group.transform("max")
    denom = (max_v - min_v).replace(0, 1.0)
    grouped["behavior_score"] = (grouped["raw_behavior_score"] - min_v) / denom
    grouped.loc[max_v == min_v, "behavior_score"] = 1.0
    normalized = grouped
    return {
        str(query): dict(zip(items["shop_id"].astype(str), items["behavior_score"]))
        for query, items in normalized.groupby("search_query")
    }


def _build_global_scores(df: pd.DataFrame) -> dict[str, float]:
    grouped = (
        df.groupby("shop_id", as_index=False)["event_weight"]
        .sum()
        .rename(columns={"event_weight": "raw_behavior_score"})
    )
    if grouped.empty:
        return {}
    normalized = _normalize_group(grouped)
    return dict(zip(normalized["shop_id"].astype(str), normalized["behavior_score"]))


def _normalize_group(group: pd.DataFrame) -> pd.DataFrame:
    group = group.copy()
    min_v = group["raw_behavior_score"].min()
    max_v = group["raw_behavior_score"].max()
    if max_v == min_v:
        group["behavior_score"] = 1.0
    else:
        group["behavior_score"] = (group["raw_behavior_score"] - min_v) / (max_v - min_v)
    return group


def select_weights(matched_behavior_count: int) -> tuple[float, float]:
    if matched_behavior_count == 0:
        return 1.0, 0.0
    if matched_behavior_count < 5:
        return 0.6, 0.4
    return 0.4, 0.6


def minmax(values: np.ndarray) -> np.ndarray:
    min_v = float(np.min(values))
    max_v = float(np.max(values))
    if max_v == min_v:
        return np.ones_like(values, dtype=np.float32)
    return ((values - min_v) / (max_v - min_v)).astype(np.float32)


def rank_candidates(
    query_embedding: np.ndarray,
    shop_embeddings: np.ndarray,
    shop_ids: list[str],
    behavior_by_shop: dict[str, float],
    global_behavior_by_shop: dict[str, float] | None = None,
    top_k: int = 5,
    mode: str = "hybrid",
) -> list[dict]:
    rows, _diagnostics = score_candidates(
        query_embedding=query_embedding,
        shop_embeddings=shop_embeddings,
        shop_ids=shop_ids,
        behavior_by_shop=behavior_by_shop,
        global_behavior_by_shop=global_behavior_by_shop,
        mode=mode,
    )
    return sorted(rows, key=lambda row: row["score"], reverse=True)[:top_k]


def score_candidates(
    query_embedding: np.ndarray,
    shop_embeddings: np.ndarray,
    shop_ids: list[str],
    behavior_by_shop: dict[str, float],
    global_behavior_by_shop: dict[str, float] | None = None,
    mode: str = "hybrid",
) -> tuple[list[dict], dict[str, float | int | str]]:
    raw_semantic_scores = shop_embeddings @ query_embedding
    semantic_scores = minmax(raw_semantic_scores)
    alpha, beta = select_weights(len(behavior_by_shop))
    global_behavior_by_shop = global_behavior_by_shop or {}

    rows = []
    for idx, shop_id in enumerate(shop_ids):
        shop_id = str(shop_id)
        semantic = float(semantic_scores[idx])
        raw_semantic = float(raw_semantic_scores[idx])
        behavior = float(behavior_by_shop.get(shop_id, 0.0))
        global_behavior = float(global_behavior_by_shop.get(shop_id, 0.0))

        if mode == "semantic_only":
            score = semantic
        elif mode == "behavior_only":
            score = behavior if behavior_by_shop else global_behavior
        elif mode == "popularity":
            score = global_behavior
        else:
            score = alpha * semantic + beta * behavior

        rows.append(
            {
                "shop_id": shop_id,
                "score": score,
                "semantic_score": semantic,
                "raw_semantic_score": raw_semantic,
                "behavior_score": behavior,
                "global_behavior_score": global_behavior,
            }
        )

    score_values = np.array([row["score"] for row in rows], dtype=np.float32)
    diagnostics: dict[str, float | int | str] = {
        "mode": mode,
        "candidate_count": len(rows),
        "matched_behavior_count": len(behavior_by_shop),
        "global_behavior_count": len(global_behavior_by_shop),
        "alpha": alpha,
        "beta": beta,
        "raw_semantic_min": float(np.min(raw_semantic_scores)),
        "raw_semantic_max": float(np.max(raw_semantic_scores)),
        "raw_semantic_mean": float(np.mean(raw_semantic_scores)),
        "score_min": float(np.min(score_values)),
        "score_max": float(np.max(score_values)),
        "score_mean": float(np.mean(score_values)),
    }
    return rows, diagnostics


def build_global_popularity_frame(behavior_payload: dict[str, object]) -> pd.DataFrame:
    global_scores = behavior_payload.get("global_scores", {})
    if not isinstance(global_scores, dict):
        global_scores = defaultdict(float)
    return pd.DataFrame(
        [{"shop_id": shop_id, "score": score} for shop_id, score in global_scores.items()]
    )
