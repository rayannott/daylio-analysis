"""Monthly journal report: stats for one month + comparison to baseline."""

import datetime
from collections import Counter
from statistics import mean, stdev, median
from typing import TYPE_CHECKING

import numpy as np

from src.analytics.correlations import activity_mood_effects
from src.analytics.anomalies import mood_outliers, mood_streaks
from src.analytics.schemas import (
    ActivityFrequency,
    MonthComparison,
    MonthSummary,
    MonthlyReport,
)

if TYPE_CHECKING:
    from src.dataset import Dataset


def _month_range(month: datetime.date) -> tuple[str, str]:
    """Return (start, end) date strings for slicing a Dataset via __getitem__."""
    start = month.replace(day=1)
    if start.month == 12:
        end = start.replace(year=start.year + 1, month=1)
    else:
        end = start.replace(month=start.month + 1)
    return f"{start:%d.%m.%Y}", f"{end:%d.%m.%Y}"


def _shift_month(d: datetime.date, months: int) -> datetime.date:
    m = d.month + months
    y = d.year + (m - 1) // 12
    m = (m - 1) % 12 + 1
    return d.replace(year=y, month=m, day=1)


def get_monthly_report(
    df: "Dataset",
    month: datetime.date,
    *,
    baseline_months: int = 2,
    top_n_activities: int = 15,
) -> MonthlyReport:
    """Compute a MonthlyReport for the given month with comparison to baseline."""
    month = month.replace(day=1)
    start_str, end_str = _month_range(month)
    df_month = df[start_str:end_str]

    if len(df_month) == 0:
        raise ValueError(f"No entries for {month:%B %Y}")

    mood_mean = mean(e.mood for e in df_month)
    mood_std_val = stdev(e.mood for e in df_month) if len(df_month) > 1 else 0.0

    days = df_month.group_by("day")
    daily_moods = {
        day: mean(e.mood for e in entries) for day, entries in days.items()
    }
    best_day = max(daily_moods, key=daily_moods.get)  # type: ignore[arg-type]
    worst_day = min(daily_moods, key=daily_moods.get)  # type: ignore[arg-type]

    act_counter = df_month.activities()
    top_acts = [
        ActivityFrequency(
            activity=act,
            count=cnt,
            frequency=cnt / len(df_month),
        )
        for act, cnt in act_counter.most_common(top_n_activities)
    ]

    # comparison to previous N months
    comparison = None
    baseline_start = _shift_month(month, -baseline_months)
    bs_start_str = f"{baseline_start:%d.%m.%Y}"
    df_baseline = df[bs_start_str:start_str]
    if len(df_baseline) > 0:
        baseline_mood = mean(e.mood for e in df_baseline)
        baseline_note_len = median(len(e.note) for e in df_baseline)
        month_note_len = median(len(e.note) for e in df_month)
        comparison = MonthComparison(
            mood_delta=mood_mean - baseline_mood,
            mood_delta_pct=(mood_mean - baseline_mood) / baseline_mood if baseline_mood else 0.0,
            entries_delta=len(df_month) - len(df_baseline),
            note_length_delta=month_note_len - baseline_note_len,
            baseline_months=baseline_months,
            baseline_mood=baseline_mood,
        )

    outliers = [
        o for o in mood_outliers(df, window=30) if o.date.month == month.month and o.date.year == month.year
    ]
    streaks = [
        s for s in mood_streaks(df)
        if not (s.end < month or s.start >= _shift_month(month, 1))
    ]

    effects = activity_mood_effects(df_month, min_count=3, alpha=0.10)

    summary = MonthSummary(
        month=month,
        n_entries=len(df_month),
        n_days_logged=len(days),
        mood_mean=mood_mean,
        mood_std=mood_std_val,
        best_day=best_day,
        worst_day=worst_day,
        top_activities=top_acts,
        comparison=comparison,
        outliers=outliers,
        streaks=streaks,
    )

    return MonthlyReport(summary=summary, activity_effects=effects)
