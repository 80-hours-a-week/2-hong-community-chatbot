from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import settings
from src.recommender.query_parser import encode_shops, load_text_encoder
from src.recommender.ranker import build_behavior_scores
from src.recommender.session import split_sub_sessions


def build_index(
    raw_data_dir: str | Path = settings.raw_data_dir,
    out_dir: str | Path = settings.processed_data_dir,
    embedding_backend: str = settings.embedding_backend,
    max_log_rows: int | None = None,
) -> dict[str, object]:
    raw_path = Path(raw_data_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logs_path = _resolve_data_file(raw_path, "logs.csv")
    shops_path = _resolve_data_file(raw_path, "shops.csv")

    logs = pd.read_csv(logs_path, nrows=max_log_rows)
    shops = pd.read_csv(shops_path)

    logs = split_sub_sessions(logs)
    behavior_payload = build_behavior_scores(logs)

    encoder = load_text_encoder(embedding_backend)
    shop_embeddings = encode_shops(shops, encoder)

    np.save(out_path / "shop_embeddings.npy", shop_embeddings)
    shops.to_pickle(out_path / "shops_indexed.pkl")
    with open(out_path / "behavior_scores.pkl", "wb") as f:
        pickle.dump(behavior_payload, f)

    metadata = {
        "shop_count": int(len(shops)),
        "log_count": int(len(logs)),
        "embedding_backend": encoder.backend_name,
        "embedding_dimension": int(shop_embeddings.shape[1]),
        "behavior_query_count": int(len(behavior_payload.get("query_scores", {}))),
    }
    (out_path / "index_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def _resolve_data_file(base_dir: Path, filename: str) -> Path:
    candidates = [
        base_dir / filename,
        Path(filename),
        Path("data/raw") / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {filename}. Tried: {candidates}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build recommendation index files.")
    parser.add_argument("--raw-data-dir", default=str(settings.raw_data_dir))
    parser.add_argument("--out-dir", default=str(settings.processed_data_dir))
    parser.add_argument("--embedding-backend", default=settings.embedding_backend)
    parser.add_argument("--max-log-rows", type=int, default=None)
    args = parser.parse_args()

    metadata = build_index(
        raw_data_dir=args.raw_data_dir,
        out_dir=args.out_dir,
        embedding_backend=args.embedding_backend,
        max_log_rows=args.max_log_rows,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
