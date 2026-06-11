"""HTML report generation using report-creator."""

import datetime
import pathlib
from typing import TYPE_CHECKING

import pandas as pd
import report_creator as rc

from src.analytics import analyze
from src.analytics.monthly import get_monthly_report
from src.analytics.frame import build_daily_mood
from src.analytics.schemas import FullReport, MonthlyReport

if TYPE_CHECKING:
    from src.dataset import Dataset

REPORTS_DIR = pathlib.Path("reports")


def _ensure_dir(path: pathlib.Path) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ── full report ───────────────────────────────────────────────────────

def _full_header(report: FullReport, df: "Dataset") -> rc.Group:
    first = df.entries[-1].full_date.date()
    last = df.entries[0].full_date.date()
    days = (last - first).days + 1
    entries_per_day = len(df) / days if days else 0
    return rc.Group(
        rc.Metric("Mood", f"{report.trends.rolling.points[-1].mean:.2f}", label="30-day rolling avg"),
        rc.Metric("Entries", len(df)),
        rc.Metric("Days tracked", days),
        rc.Metric("Entries/day", f"{entries_per_day:.1f}"),
    )


def _mood_trend_chart(df: "Dataset") -> rc.Widget:
    daily = build_daily_mood(df)
    rolling = daily["mood_mean"].rolling(30, min_periods=15).mean()
    trend_df = pd.DataFrame({
        "date": daily.index,
        "daily": daily["mood_mean"].values,
        "30d avg": rolling.values,
    })
    return rc.Line(trend_df, x="date", y=["daily", "30d avg"], label="Mood over time")


def _activity_effects_chart(report: FullReport) -> rc.Widget:
    sig = [e for e in report.correlations.activity_mood.effects if e.significant]
    top = sorted(sig, key=lambda e: abs(e.cohens_d), reverse=True)[:20]
    if not top:
        return rc.Markdown("*No statistically significant activity effects found.*")
    effects_df = pd.DataFrame([
        {"activity": e.activity, "Cohen's d": round(e.cohens_d, 3), "n": e.n_with}
        for e in top
    ])
    return rc.Bar(
        effects_df, x="Cohen's d", y="activity",
        label="Top activity effects on mood (significant, by Cohen's d)",
        orientation="h",
    )


def _change_points_section(report: FullReport) -> rc.Widget:
    cps = report.anomalies.change_points
    if not cps:
        return rc.Markdown("*No structural change-points detected.*")
    rows = "\n".join(
        f"| {cp.date} | {cp.mean_before:.3f} | {cp.mean_after:.3f} | {cp.delta:+.3f} | {cp.p_value:.4f} |"
        for cp in cps
    )
    md = f"| Date | Before | After | Delta | p-value |\n|---|---|---|---|---|\n{rows}"
    return rc.Markdown(md, label="Mood change-points")


def _anomalies_section(report: FullReport) -> rc.Widget:
    blocks = []
    outliers = report.anomalies.outliers[:10]
    if outliers:
        rows = "\n".join(
            f"| {o.date} | {o.mood:.2f} | {o.baseline:.2f} | {o.score:+.1f} | {o.direction} |"
            for o in outliers
        )
        blocks.append(rc.Markdown(
            f"| Date | Mood | Baseline | z-score | Dir |\n|---|---|---|---|---|\n{rows}",
            label="Outlier days",
        ))
    streaks = [s for s in report.anomalies.streaks if s.kind != "logging_gap"][:10]
    if streaks:
        rows = "\n".join(
            f"| {s.kind} | {s.start} | {s.end} | {s.length_days} | {s.mean_mood:.2f} |"
            for s in streaks if s.mean_mood is not None
        )
        blocks.append(rc.Markdown(
            f"| Kind | Start | End | Days | Mood |\n|---|---|---|---|---|\n{rows}",
            label="Mood streaks",
        ))
    if not blocks:
        return rc.Markdown("*No notable anomalies.*")
    return rc.Block(*blocks)


def _habit_trends_section(report: FullReport) -> rc.Widget:
    moving = [h for h in report.trends.habit_trends if h.direction != "stable"]
    if not moving:
        return rc.Markdown("*All habits stable.*")
    ht_df = pd.DataFrame([
        {
            "activity": h.activity,
            "slope/month": round(h.slope_per_month, 4),
            "direction": h.direction,
            "baseline": round(h.baseline_freq, 3),
            "recent": round(h.recent_freq, 3),
        }
        for h in moving
    ])
    return rc.Bar(
        ht_df, x="slope/month", y="activity", dimension="direction",
        label="Habit trends (rising / declining)",
        orientation="h",
    )


def generate_full_report(
    df: "Dataset",
    output: pathlib.Path | None = None,
) -> pathlib.Path:
    """Generate a full-journal HTML report and return the file path."""
    output = output or _ensure_dir(REPORTS_DIR / "report_full.html")

    report = analyze(df)
    creator = rc.ReportCreator(
        title="Journal Report",
        description=f"{len(df):,} entries",
    )
    view = rc.Block(
        _full_header(report, df),
        _mood_trend_chart(df),
        _activity_effects_chart(report),
        _change_points_section(report),
        _anomalies_section(report),
        _habit_trends_section(report),
    )
    creator.save(view, output)
    print(f"Full report saved to {output}")
    return output


# ── monthly report ────────────────────────────────────────────────────

def _monthly_header(mr: MonthlyReport) -> rc.Group:
    s = mr.summary
    metrics = [
        rc.Metric("Mood", f"{s.mood_mean:.2f}", label=f"std {s.mood_std:.2f}"),
        rc.Metric("Entries", s.n_entries),
        rc.Metric("Days logged", s.n_days_logged),
    ]
    if s.comparison:
        c = s.comparison
        arrow = "+" if c.mood_delta >= 0 else ""
        metrics.append(
            rc.Metric("vs baseline", f"{arrow}{c.mood_delta:.3f}", label=f"{c.mood_delta_pct:+.1%} vs prev {c.baseline_months}mo")
        )
    return rc.Group(*metrics)


def _monthly_mood_chart(df: "Dataset", month: datetime.date) -> rc.Widget:
    daily = build_daily_mood(df)
    start = pd.Timestamp(month.replace(day=1))
    if month.month == 12:
        end = pd.Timestamp(month.replace(year=month.year + 1, month=1, day=1))
    else:
        end = pd.Timestamp(month.replace(month=month.month + 1, day=1))
    mask = (daily.index >= start) & (daily.index < end)
    month_daily = daily[mask].copy()
    if month_daily.empty:
        return rc.Markdown("*No daily data for this month.*")
    month_daily = month_daily.reset_index()
    month_daily.columns = ["date", "mood", "n_entries", "imputed"]
    return rc.Scatter(month_daily, x="date", y="mood", label=f"Daily mood — {month:%B %Y}")


def _monthly_top_activities(mr: MonthlyReport) -> rc.Widget:
    acts = mr.summary.top_activities
    if not acts:
        return rc.Markdown("*No activities recorded.*")
    acts_df = pd.DataFrame([
        {"activity": a.activity, "count": a.count, "frequency": round(a.frequency, 3)}
        for a in acts
    ])
    return rc.Bar(acts_df, x="count", y="activity", label="Top activities", orientation="h")


def _monthly_effects(mr: MonthlyReport) -> rc.Widget:
    effects = mr.activity_effects.effects[:10]
    if not effects:
        return rc.Markdown("*Not enough data for activity effects this month.*")
    eff_df = pd.DataFrame([
        {"activity": e.activity, "delta": round(e.delta, 3), "n": e.n_with}
        for e in sorted(effects, key=lambda e: abs(e.delta), reverse=True)
    ])
    return rc.Bar(eff_df, x="delta", y="activity", label="Activity mood effects", orientation="h")


def generate_monthly_report(
    df: "Dataset",
    month: datetime.date,
    output: pathlib.Path | None = None,
) -> pathlib.Path:
    """Generate a monthly HTML report and return the file path."""
    month = month.replace(day=1)
    output = output or _ensure_dir(REPORTS_DIR / f"report_{month:%Y-%m}.html")

    mr = get_monthly_report(df, month)
    creator = rc.ReportCreator(
        title=f"Journal — {month:%B %Y}",
        description=f"{mr.summary.n_entries} entries, {mr.summary.n_days_logged} days logged",
    )
    view = rc.Block(
        _monthly_header(mr),
        _monthly_mood_chart(df, month),
        _monthly_top_activities(mr),
        _monthly_effects(mr),
    )
    creator.save(view, output)
    print(f"Monthly report saved to {output}")
    return output
