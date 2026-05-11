from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable

import pandas as pd

from src.config import settings
from src.evaluation.metrics import (
    coverage,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.recommender.engine import RecommendationEngine
from src.recommender.indexer import _resolve_data_file
from src.recommender.ranker import EVENT_WEIGHTS
from src.recommender.session import split_sub_sessions


RankerFn = Callable[[str, int], list[str]]


def load_test_cases(raw_data_dir: str | Path, max_log_rows: int | None = None) -> pd.DataFrame:
    logs_path = _resolve_data_file(Path(raw_data_dir), "logs.csv")
    logs = pd.read_csv(logs_path, nrows=max_log_rows)
    logs = split_sub_sessions(logs)
    test_cases = logs[
        (logs["event_type"] == "reservation")
        & logs["search_query"].notna()
        & logs["shop_id"].notna()
    ][["sub_session_id", "search_query", "shop_id"]]
    return test_cases.drop_duplicates()


def evaluate_ranker(
    test_cases: pd.DataFrame,
    ranker_fn: RankerFn,
    all_shop_ids: list[str],
    k: int = 5,
) -> dict[str, float]:
    rows = []
    all_recs = []

    for _, case in test_cases.iterrows():
        query = str(case["search_query"])
        relevant = {str(case["shop_id"])}
        recommended = [str(shop_id) for shop_id in ranker_fn(query, k)]
        all_recs.append(recommended)

        rows.append(
            {
                "precision": precision_at_k(recommended, relevant, k),
                "recall": recall_at_k(recommended, relevant, k),
                "hit": hit_rate_at_k(recommended, relevant, k),
                "mrr": mrr(recommended, relevant),
                "ndcg": ndcg_at_k(recommended, relevant, k),
            }
        )

    if not rows:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "hit": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
            "coverage": 0.0,
        }

    metrics = pd.DataFrame(rows).mean().to_dict()
    metrics["coverage"] = coverage(all_recs, total_shop_count=len(all_shop_ids))
    return metrics


def random_ranker_factory(all_shop_ids: list[str]) -> RankerFn:
    def ranker(query: str, k: int) -> list[str]:
        return random.sample(all_shop_ids, min(k, len(all_shop_ids)))

    return ranker


def popularity_ranker_factory(raw_data_dir: str | Path, max_log_rows: int | None = None) -> RankerFn:
    logs_path = _resolve_data_file(Path(raw_data_dir), "logs.csv")
    logs = pd.read_csv(logs_path, nrows=max_log_rows)
    logs["score"] = logs["event_type"].map(EVENT_WEIGHTS).fillna(0.0)
    ranked = (
        logs.dropna(subset=["shop_id"])
        .groupby("shop_id", as_index=False)["score"]
        .sum()
        .sort_values("score", ascending=False)["shop_id"]
        .astype(str)
        .tolist()
    )

    def ranker(query: str, k: int) -> list[str]:
        return ranked[:k]

    return ranker


def engine_ranker_factory(engine: RecommendationEngine, mode: str) -> RankerFn:
    def ranker(query: str, k: int) -> list[str]:
        return [item["shop_id"] for item in engine.recommend(query=query, top_k=k, mode=mode)]

    return ranker


def run_baseline_report(
    raw_data_dir: str | Path = settings.raw_data_dir,
    index_dir: str | Path = settings.processed_data_dir,
    k: int = 5,
    max_log_rows: int | None = None,
) -> pd.DataFrame:
    engine = RecommendationEngine(index_dir=index_dir)
    test_cases = load_test_cases(raw_data_dir, max_log_rows=max_log_rows)
    all_shop_ids = engine.shop_ids

    rankers: dict[str, RankerFn] = {
        "Random": random_ranker_factory(all_shop_ids),
        "Popularity": popularity_ranker_factory(raw_data_dir, max_log_rows=max_log_rows),
        "Semantic": engine_ranker_factory(engine, mode="semantic_only"),
        "Behavior": engine_ranker_factory(engine, mode="behavior_only"),
        "Hybrid": engine_ranker_factory(engine, mode="hybrid"),
    }

    return pd.DataFrame(
        [
            {"method": name, **evaluate_ranker(test_cases, fn, all_shop_ids, k=k)}
            for name, fn in rankers.items()
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline recommendation evaluation.")
    parser.add_argument("--raw-data-dir", default=str(settings.raw_data_dir))
    parser.add_argument("--index-dir", default=str(settings.processed_data_dir))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-log-rows", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report = run_baseline_report(
        raw_data_dir=args.raw_data_dir,
        index_dir=args.index_dir,
        k=args.k,
        max_log_rows=args.max_log_rows,
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.out, index=False)
    print(json.dumps(report.to_dict(orient="records"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
