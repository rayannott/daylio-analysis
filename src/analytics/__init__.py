"""Advanced analytics for Daylio journal data.

Usage::

    from src.analytics import analyze
    report = analyze(df)
    print(report.model_dump_json(indent=2))
"""

from typing import TYPE_CHECKING

from src.analytics.schemas import (
    FullReport,
    CorrelationReport,
    AnomalyReport,
    TrendReport,
    ActivityMoodReport,
    MoodRegressionResult,
    ActivityAssociationReport,
)
from src.analytics.correlations import (
    activity_mood_effects,
    mood_regression,
    activity_associations,
    lagged_activity_effects,
)
from src.analytics.anomalies import detect_anomalies
from src.analytics.trends import analyze_trends

if TYPE_CHECKING:
    from src.dataset import Dataset

__all__ = [
    "analyze",
    "activity_mood_effects",
    "mood_regression",
    "activity_associations",
    "lagged_activity_effects",
    "detect_anomalies",
    "analyze_trends",
    "FullReport",
    "CorrelationReport",
    "AnomalyReport",
    "TrendReport",
    "ActivityMoodReport",
    "MoodRegressionResult",
    "ActivityAssociationReport",
]


def analyze(
    df: "Dataset",
    *,
    min_count: int = 10,
    alpha: float = 0.05,
) -> FullReport:
    """Run all analytics and return a single FullReport."""
    corr = CorrelationReport(
        activity_mood=activity_mood_effects(df, min_count=min_count, alpha=alpha),
        mood_regression=mood_regression(df, min_count=min_count * 2, alpha=alpha),
        associations=activity_associations(df, min_count=min_count),
        lagged_effects=lagged_activity_effects(df, min_count=min_count),
    )
    anomalies = detect_anomalies(df)
    trends = analyze_trends(df)

    return FullReport(
        correlations=corr,
        anomalies=anomalies,
        trends=trends,
    )
