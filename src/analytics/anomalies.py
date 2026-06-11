"""Anomaly detection: outlier days, change-points, mood streaks."""

import datetime
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy import stats as sp_stats

from src.analytics.frame import build_daily_mood
from src.analytics.schemas import (
    MoodAnomaly,
    ChangePoint,
    Streak,
    AnomalyReport,
)

if TYPE_CHECKING:
    from src.dataset import Dataset


def mood_outliers(
    df: "Dataset",
    *,
    window: int = 30,
    threshold: float = 3.0,
) -> list[MoodAnomaly]:
    """Flag days whose mood deviates > *threshold* MAD-units from a rolling median."""
    daily = build_daily_mood(df)
    mood = daily["mood_mean"].values
    imputed = daily["imputed"].values
    dates = daily.index

    outliers: list[MoodAnomaly] = []
    for i in range(window, len(mood)):
        if imputed[i]:
            continue
        win = mood[max(0, i - window) : i]
        med = np.median(win)
        mad = np.median(np.abs(win - med))
        if mad == 0:
            continue
        score = (mood[i] - med) / (mad * 1.4826)  # scale MAD to match σ
        if abs(score) >= threshold:
            outliers.append(
                MoodAnomaly(
                    date=dates[i].date(),
                    mood=float(mood[i]),
                    baseline=float(med),
                    score=float(score),
                    direction="high" if score > 0 else "low",
                )
            )

    outliers.sort(key=lambda a: abs(a.score), reverse=True)
    return outliers


def change_points(
    df: "Dataset",
    *,
    min_segment: int = 21,
    alpha: float = 0.01,
    max_points: int = 10,
) -> list[ChangePoint]:
    """Binary segmentation with Mann-Whitney validation per split."""
    daily = build_daily_mood(df)
    mood = daily["mood_mean"].values
    dates = daily.index

    def _best_split(arr: np.ndarray, offset: int) -> ChangePoint | None:
        best_p = 1.0
        best_idx = -1
        for i in range(min_segment, len(arr) - min_segment):
            left, right = arr[:i], arr[i:]
            _, p = sp_stats.mannwhitneyu(left, right, alternative="two-sided")
            if p < best_p:
                best_p = p
                best_idx = i
        if best_p >= alpha or best_idx < 0:
            return None
        left, right = arr[:best_idx], arr[best_idx:]
        return ChangePoint(
            date=dates[offset + best_idx].date(),
            mean_before=float(left.mean()),
            mean_after=float(right.mean()),
            delta=float(right.mean() - left.mean()),
            p_value=float(best_p),
        )

    # Iterative binary segmentation
    segments: list[tuple[int, int]] = [(0, len(mood))]
    found: list[ChangePoint] = []

    while segments and len(found) < max_points:
        start, end = segments.pop(0)
        seg = mood[start:end]
        if len(seg) < 2 * min_segment:
            continue
        cp = _best_split(seg, start)
        if cp is None:
            continue
        found.append(cp)
        split_idx = start + np.argmax(
            dates[start:end] >= np.datetime64(cp.date)
        )
        segments.append((start, int(split_idx)))
        segments.append((int(split_idx), end))

    found.sort(key=lambda c: c.date)
    return found


def _classify_day(mood: float, low: float, high: float) -> Literal["low", "high"] | None:
    if mood <= low:
        return "low"
    if mood >= high:
        return "high"
    return None


def mood_streaks(
    df: "Dataset",
    *,
    low: float = 3.5,
    high: float = 4.5,
    min_streak: int = 3,
) -> list[Streak]:
    """Consecutive-day runs of low/high mood plus logging gaps."""
    daily = build_daily_mood(df)
    mood = daily["mood_mean"].values
    imputed = daily["imputed"].values
    dates = daily.index

    streaks: list[Streak] = []

    # mood streaks (skip imputed days — only count real observations)
    current_kind: Literal["low", "high"] | None = None
    start_idx = 0
    run_moods: list[float] = []

    for i in range(len(mood)):
        if imputed[i]:
            if current_kind and len(run_moods) >= min_streak:
                streaks.append(
                    Streak(
                        kind=current_kind,
                        start=dates[start_idx].date(),
                        end=dates[i - 1].date(),
                        length_days=(dates[i - 1] - dates[start_idx]).days + 1,
                        mean_mood=float(np.mean(run_moods)),
                    )
                )
            current_kind = None
            run_moods = []
            continue

        kind = _classify_day(mood[i], low, high)
        if kind == current_kind:
            run_moods.append(mood[i])
        else:
            if current_kind and len(run_moods) >= min_streak:
                streaks.append(
                    Streak(
                        kind=current_kind,
                        start=dates[start_idx].date(),
                        end=dates[i - 1].date(),
                        length_days=(dates[i - 1] - dates[start_idx]).days + 1,
                        mean_mood=float(np.mean(run_moods)),
                    )
                )
            current_kind = kind
            start_idx = i
            run_moods = [mood[i]] if kind else []

    if current_kind and len(run_moods) >= min_streak:
        streaks.append(
            Streak(
                kind=current_kind,
                start=dates[start_idx].date(),
                end=dates[len(mood) - 1].date(),
                length_days=(dates[len(mood) - 1] - dates[start_idx]).days + 1,
                mean_mood=float(np.mean(run_moods)),
            )
        )

    # logging gaps (days with 0 real entries, using the original dataset)
    groups = df.group_by("day")
    real_dates = sorted(groups.keys())
    for i in range(1, len(real_dates)):
        gap_days = (real_dates[i] - real_dates[i - 1]).days - 1
        if gap_days >= min_streak:
            gap_start = real_dates[i - 1] + datetime.timedelta(days=1)
            gap_end = real_dates[i] - datetime.timedelta(days=1)
            streaks.append(
                Streak(
                    kind="logging_gap",
                    start=gap_start,
                    end=gap_end,
                    length_days=gap_days,
                    mean_mood=None,
                )
            )

    streaks.sort(key=lambda s: s.length_days, reverse=True)
    return streaks


def detect_anomalies(
    df: "Dataset",
    *,
    outlier_window: int = 30,
    outlier_threshold: float = 3.0,
    cp_min_segment: int = 21,
    streak_low: float = 3.5,
    streak_high: float = 4.5,
    streak_min: int = 3,
) -> AnomalyReport:
    """Run all anomaly detectors and return a bundled report."""
    return AnomalyReport(
        outliers=mood_outliers(df, window=outlier_window, threshold=outlier_threshold),
        change_points=change_points(df, min_segment=cp_min_segment),
        streaks=mood_streaks(df, low=streak_low, high=streak_high, min_streak=streak_min),
    )
