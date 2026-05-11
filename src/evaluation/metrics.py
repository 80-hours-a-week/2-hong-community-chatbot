from __future__ import annotations

import math


def hit_rate_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if relevant.intersection(recommended[:k]) else 0.0


def precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return len(relevant.intersection(recommended[:k])) / k


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(relevant.intersection(recommended[:k])) / len(relevant)


def mrr(recommended: list[str], relevant: set[str]) -> float:
    for rank, shop_id in enumerate(recommended, start=1):
        if shop_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    dcg = 0.0
    for idx, shop_id in enumerate(recommended[:k], start=1):
        if shop_id in relevant:
            dcg += 1.0 / math.log2(idx + 1)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def coverage(all_recommendations: list[list[str]], total_shop_count: int) -> float:
    if total_shop_count == 0:
        return 0.0
    recommended_unique = {shop_id for recs in all_recommendations for shop_id in recs}
    return len(recommended_unique) / total_shop_count
