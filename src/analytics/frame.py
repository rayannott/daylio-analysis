"""Dataset -> pandas DataFrames used by all analytics modules."""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.dataset import Dataset

ACT_PREFIX = "act:"


def build_entry_frame(df: "Dataset", min_count: int = 1) -> pd.DataFrame:
    """Per-entry DataFrame with one-hot activity columns.

    Rows sorted ascending by full_date.  Columns: full_date, date, mood,
    weekday (0=Mon), hour, plus one bool column per activity (prefixed with
    ``act:``) that occurs >= *min_count* times across the whole dataset.
    """
    activities_counter = df.activities()
    keep = {a for a, n in activities_counter.items() if n >= min_count}

    rows: list[dict] = []
    for e in reversed(df.entries):
        row: dict = {
            "full_date": e.full_date,
            "date": e.full_date.date(),
            "mood": e.mood,
            "weekday": e.full_date.weekday(),
            "hour": e.full_date.hour,
        }
        for act in keep:
            row[f"{ACT_PREFIX}{act}"] = act in e.activities
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def build_daily_mood(df: "Dataset") -> pd.DataFrame:
    """Daily mood series, gap-filled with linear interpolation.

    Returns a DatetimeIndex-ed DataFrame with columns:
        mood_mean, n_entries, imputed (True for interpolated days).
    """
    groups = df.group_by("day")
    dates = sorted(groups.keys())
    records = [
        {
            "date": d,
            "mood_mean": np.mean([e.mood for e in groups[d]]),
            "n_entries": len(groups[d]),
        }
        for d in dates
    ]
    daily = pd.DataFrame(records).set_index("date")
    daily.index = pd.to_datetime(daily.index)

    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_range)
    daily["imputed"] = daily["mood_mean"].isna()
    daily["mood_mean"] = daily["mood_mean"].interpolate(method="linear")
    daily["n_entries"] = daily["n_entries"].fillna(0).astype(int)
    return daily


def activity_columns(frame: pd.DataFrame) -> list[str]:
    """Return activity column names (with prefix) from an entry frame."""
    return [c for c in frame.columns if c.startswith(ACT_PREFIX)]


def strip_prefix(col: str) -> str:
    """Remove the act: prefix to get the original activity name."""
    return col.removeprefix(ACT_PREFIX)
