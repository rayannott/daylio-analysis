"""Pydantic result schemas for all analytics modules.

Every model is JSON-serializable via model_dump_json() for a future UI.
"""

import datetime
from typing import Literal

from pydantic import BaseModel


# ── correlations ──────────────────────────────────────────────────────

class ActivityMoodEffect(BaseModel):
    activity: str
    n_with: int
    n_without: int
    mean_with: float
    mean_without: float
    delta: float
    cohens_d: float
    p_value: float
    p_value_adjusted: float
    significant: bool


class ActivityMoodReport(BaseModel):
    test: str = "mann_whitney_u"
    alpha: float
    effects: list[ActivityMoodEffect]


class RegressionCoefficient(BaseModel):
    name: str
    coefficient: float
    std_err: float
    p_value: float
    ci_low: float
    ci_high: float


class MoodRegressionResult(BaseModel):
    model: str = "ols"
    n_observations: int
    r_squared: float
    covariates: list[str]
    coefficients: list[RegressionCoefficient]


class ActivityAssociation(BaseModel):
    activity_a: str
    activity_b: str
    lift: float
    pmi: float
    phi: float


class ActivityAssociationReport(BaseModel):
    n_entries: int
    min_count: int
    associations: list[ActivityAssociation]


class LaggedEffect(BaseModel):
    activity: str
    lag_days: int
    n_with: int
    n_without: int
    mean_with: float
    mean_without: float
    delta: float
    p_value: float


class CorrelationReport(BaseModel):
    activity_mood: ActivityMoodReport
    mood_regression: MoodRegressionResult
    associations: ActivityAssociationReport
    lagged_effects: list[LaggedEffect]


# ── anomalies ─────────────────────────────────────────────────────────

class MoodAnomaly(BaseModel):
    date: datetime.date
    mood: float
    baseline: float
    score: float
    direction: Literal["low", "high"]


class ChangePoint(BaseModel):
    date: datetime.date
    mean_before: float
    mean_after: float
    delta: float
    p_value: float


class Streak(BaseModel):
    kind: Literal["low", "high", "logging_gap"]
    start: datetime.date
    end: datetime.date
    length_days: int
    mean_mood: float | None


class AnomalyReport(BaseModel):
    outliers: list[MoodAnomaly]
    change_points: list[ChangePoint]
    streaks: list[Streak]


# ── trends ────────────────────────────────────────────────────────────

class Decomposition(BaseModel):
    dates: list[datetime.date]
    observed: list[float]
    trend: list[float]
    seasonal: list[float]
    residual: list[float]
    imputed: list[bool]


class RollingPoint(BaseModel):
    date: datetime.date
    mean: float
    std: float


class RollingStats(BaseModel):
    window: int
    points: list[RollingPoint]


class HabitTrend(BaseModel):
    activity: str
    slope_per_month: float
    recent_freq: float
    baseline_freq: float
    direction: Literal["rising", "declining", "stable"]
    p_value: float


class TrendReport(BaseModel):
    decomposition: Decomposition
    rolling: RollingStats
    habit_trends: list[HabitTrend]


# ── monthly ───────────────────────────────────────────────────────────

class ActivityFrequency(BaseModel):
    activity: str
    count: int
    frequency: float


class MonthComparison(BaseModel):
    mood_delta: float
    mood_delta_pct: float
    entries_delta: int
    note_length_delta: float
    baseline_months: int
    baseline_mood: float


class MonthSummary(BaseModel):
    month: datetime.date
    n_entries: int
    n_days_logged: int
    mood_mean: float
    mood_std: float
    best_day: datetime.date
    worst_day: datetime.date
    top_activities: list[ActivityFrequency]
    comparison: MonthComparison | None
    outliers: list[MoodAnomaly]
    streaks: list[Streak]


class MonthlyReport(BaseModel):
    summary: MonthSummary
    activity_effects: ActivityMoodReport


# ── top-level bundle ──────────────────────────────────────────────────

class FullReport(BaseModel):
    correlations: CorrelationReport
    anomalies: AnomalyReport
    trends: TrendReport
