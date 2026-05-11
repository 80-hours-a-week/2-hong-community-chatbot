from __future__ import annotations

import pandas as pd


REQUIRED_LOG_COLUMNS = {
    "event_type",
    "event_timestamp",
    "session_id",
    "shop_id",
    "search_query",
}


def parse_event_timestamp(series: pd.Series) -> pd.Series:
    """Parse epoch timestamps from logs.csv.

    The provided logs use microsecond epoch values. This helper also accepts
    normal datetime strings for easier testing.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.9:
        median = numeric.dropna().median()
        if median > 1e14:
            return pd.to_datetime(numeric, unit="us", errors="coerce")
        if median > 1e11:
            return pd.to_datetime(numeric, unit="ms", errors="coerce")
        return pd.to_datetime(numeric, unit="s", errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def split_sub_sessions(logs: pd.DataFrame, time_gap_minutes: int = 30) -> pd.DataFrame:
    missing = REQUIRED_LOG_COLUMNS - set(logs.columns)
    if missing:
        raise ValueError(f"logs.csv missing columns: {sorted(missing)}")

    df = logs.copy()
    df["event_timestamp"] = parse_event_timestamp(df["event_timestamp"])
    df = df.sort_values(["session_id", "event_timestamp"]).reset_index(drop=True)

    df["non_null_query"] = df["search_query"].where(df["search_query"].notna())
    filled_query = df.groupby("session_id")["non_null_query"].ffill()
    df["prev_non_null_query"] = filled_query.groupby(df["session_id"]).shift(1)
    df["time_diff"] = df.groupby("session_id")["event_timestamp"].diff()

    query_changed = (
        df["search_query"].notna()
        & df["prev_non_null_query"].notna()
        & (df["search_query"] != df["prev_non_null_query"])
    )
    time_gap = df["time_diff"] > pd.Timedelta(minutes=time_gap_minutes)
    first_in_session = df.groupby("session_id").cumcount() == 0

    df["is_boundary"] = first_in_session | query_changed | time_gap
    df["sub_session_no"] = df.groupby("session_id")["is_boundary"].cumsum() - 1
    df["sub_session_id"] = (
        df["session_id"].astype(str) + "_" + df["sub_session_no"].astype(str)
    )
    df["search_query"] = df.groupby("sub_session_id")["search_query"].ffill()

    return df.drop(
        columns=["non_null_query", "prev_non_null_query", "time_diff", "is_boundary"],
        errors="ignore",
    )
