"""Trend decomposition, rolling statistics, and habit tracking."""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.tsa.seasonal import STL

from src.analytics.frame import build_entry_frame, build_daily_mood, activity_columns, strip_prefix
from src.analytics.schemas import (
    Decomposition,
    RollingPoint,
    RollingStats,
    HabitTrend,
    TrendReport,
)

if TYPE_CHECKING:
    from src.dataset import Dataset


def decompose_mood(
    df: "Dataset",
    *,
    period: int = 7,
) -> Decomposition:
    """STL decomposition of the daily mood series into trend + seasonal + residual."""
    daily = build_daily_mood(df)
    mood = daily["mood_mean"]

    stl = STL(mood, period=period, robust=True)
    result = stl.fit()

    dates = [d.date() for d in daily.index]
    return Decomposition(
        dates=dates,
        observed=[float(v) for v in mood.values],
        trend=[float(v) for v in result.trend.values],
        seasonal=[float(v) for v in result.seasonal.values],
        residual=[float(v) for v in result.resid.values],
        imputed=daily["imputed"].tolist(),
    )


def rolling_mood(
    df: "Dataset",
    *,
    window: int = 30,
) -> RollingStats:
    """Rolling mean and standard deviation of daily mood."""
    daily = build_daily_mood(df)
    mood = daily["mood_mean"]

    rolling_mean = mood.rolling(window, min_periods=window // 2).mean()
    rolling_std = mood.rolling(window, min_periods=window // 2).std()

    points: list[RollingPoint] = []
    for i in range(len(daily)):
        if pd.isna(rolling_mean.iloc[i]):
            continue
        points.append(
            RollingPoint(
                date=daily.index[i].date(),
                mean=float(rolling_mean.iloc[i]),
                std=float(rolling_std.iloc[i]) if not pd.isna(rolling_std.iloc[i]) else 0.0,
            )
        )

    return RollingStats(window=window, points=points)


def habit_trends(
    df: "Dataset",
    *,
    min_count: int = 10,
    freq: str = "ME",
    stable_threshold: float = 0.005,
) -> list[HabitTrend]:
    """Per-activity frequency slope over time (monthly buckets)."""
    entry_frame = build_entry_frame(df, min_count=min_count)
    acts = activity_columns(entry_frame)
    if not acts:
        return []

    entry_frame = entry_frame.set_index("date")
    monthly = entry_frame[acts].resample(freq).sum()
    n_entries_monthly = entry_frame["mood"].resample(freq).count()

    # normalize to frequency (fraction of entries per month that have the activity)
    freq_df = monthly.div(n_entries_monthly, axis=0).fillna(0)

    if len(freq_df) < 3:
        return []

    x = np.arange(len(freq_df), dtype=float)
    results: list[HabitTrend] = []

    for act in acts:
        y = freq_df[act].values.astype(float)
        slope, _, _, p_value, _ = sp_stats.linregress(x, y)

        baseline = float(y[: len(y) // 2].mean()) if len(y) >= 4 else float(y.mean())
        recent = float(y[-(len(y) // 4 or 1) :].mean())

        if abs(slope) < stable_threshold:
            direction = "stable"
        elif slope > 0:
            direction = "rising"
        else:
            direction = "declining"

        results.append(
            HabitTrend(
                activity=strip_prefix(act),
                slope_per_month=float(slope),
                recent_freq=recent,
                baseline_freq=baseline,
                direction=direction,
                p_value=float(p_value),
            )
        )

    results.sort(key=lambda h: abs(h.slope_per_month), reverse=True)
    return results


def analyze_trends(
    df: "Dataset",
    *,
    stl_period: int = 7,
    rolling_window: int = 30,
    habit_min_count: int = 10,
) -> TrendReport:
    """Run all trend analyses and return a bundled report."""
    return TrendReport(
        decomposition=decompose_mood(df, period=stl_period),
        rolling=rolling_mood(df, window=rolling_window),
        habit_trends=habit_trends(df, min_count=habit_min_count),
    )
