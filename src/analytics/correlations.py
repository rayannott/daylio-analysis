"""Activity-mood correlations with proper statistical inference."""

import math
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

from src.analytics.frame import build_entry_frame, build_daily_mood, activity_columns, strip_prefix
from src.analytics.schemas import (
    ActivityMoodEffect,
    ActivityMoodReport,
    RegressionCoefficient,
    MoodRegressionResult,
    ActivityAssociation,
    ActivityAssociationReport,
    LaggedEffect,
)

if TYPE_CHECKING:
    from src.dataset import Dataset


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled_std = math.sqrt(
        ((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2)
        / (na + nb - 2)
    )
    if pooled_std == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def activity_mood_effects(
    df: "Dataset",
    *,
    min_count: int = 10,
    alpha: float = 0.05,
) -> ActivityMoodReport:
    """Per-activity mood effect with Cohen's d, Mann-Whitney U, and BH FDR."""
    frame = build_entry_frame(df, min_count=min_count)
    acts = activity_columns(frame)
    mood = frame["mood"].values

    raw: list[dict] = []
    p_values: list[float] = []

    for act in acts:
        mask = frame[act].values.astype(bool)
        with_mood = mood[mask]
        without_mood = mood[~mask]
        if len(with_mood) < 2 or len(without_mood) < 2:
            continue
        _, p = sp_stats.mannwhitneyu(with_mood, without_mood, alternative="two-sided")
        d = _cohens_d(with_mood, without_mood)
        raw.append(
            {
                "activity": strip_prefix(act),
                "n_with": int(mask.sum()),
                "n_without": int((~mask).sum()),
                "mean_with": float(with_mood.mean()),
                "mean_without": float(without_mood.mean()),
                "delta": float(with_mood.mean() - without_mood.mean()),
                "cohens_d": d,
                "p_value": float(p),
            }
        )
        p_values.append(float(p))

    if not raw:
        return ActivityMoodReport(alpha=alpha, effects=[])

    _, p_adj, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")

    effects = []
    for r, pa in zip(raw, p_adj):
        effects.append(
            ActivityMoodEffect(
                **r,
                p_value_adjusted=float(pa),
                significant=bool(pa < alpha),
            )
        )

    effects.sort(key=lambda e: (-e.significant, e.p_value_adjusted, -abs(e.cohens_d)))
    return ActivityMoodReport(alpha=alpha, effects=effects)


def mood_regression(
    df: "Dataset",
    *,
    min_count: int = 20,
    covariates: tuple[str, ...] = ("weekday",),
    alpha: float = 0.05,
) -> MoodRegressionResult:
    """OLS regression: mood ~ activities + covariates, with p-values and CIs."""
    frame = build_entry_frame(df, min_count=min_count)
    acts = activity_columns(frame)

    X = frame[acts].astype(float)
    for cov in covariates:
        if cov == "weekday":
            dummies = pd.get_dummies(frame["weekday"], prefix="wd", drop_first=True, dtype=float)
            X = pd.concat([X, dummies], axis=1)
        elif cov == "hour":
            X[cov] = frame["hour"].astype(float)
    X = sm.add_constant(X)
    y = frame["mood"].astype(float)

    model = sm.OLS(y, X).fit()
    ci = model.conf_int(alpha=alpha)

    coefficients = []
    for name in model.params.index:
        clean_name = strip_prefix(str(name))
        coefficients.append(
            RegressionCoefficient(
                name=clean_name,
                coefficient=float(model.params[name]),
                std_err=float(model.bse[name]),
                p_value=float(model.pvalues[name]),
                ci_low=float(ci.loc[name, 0]),
                ci_high=float(ci.loc[name, 1]),
            )
        )

    coefficients.sort(key=lambda c: c.p_value)
    return MoodRegressionResult(
        n_observations=int(model.nobs),
        r_squared=float(model.rsquared),
        covariates=list(covariates),
        coefficients=coefficients,
    )


def activity_associations(
    df: "Dataset",
    *,
    min_count: int = 10,
) -> ActivityAssociationReport:
    """Symmetric activity co-occurrence: lift, PMI, phi."""
    frame = build_entry_frame(df, min_count=min_count)
    acts = activity_columns(frame)
    n = len(frame)
    if n == 0:
        return ActivityAssociationReport(n_entries=0, min_count=min_count, associations=[])

    freqs = {a: frame[a].sum() / n for a in acts}

    results: list[ActivityAssociation] = []
    for a, b in combinations(acts, 2):
        pa, pb = freqs[a], freqs[b]
        if pa == 0 or pb == 0:
            continue
        pab = (frame[a] & frame[b]).sum() / n
        if pab == 0:
            continue

        lift = pab / (pa * pb)
        pmi = math.log2(pab / (pa * pb))

        # phi coefficient (Matthews correlation)
        n11 = (frame[a] & frame[b]).sum()
        n10 = (frame[a] & ~frame[b]).sum()
        n01 = (~frame[a] & frame[b]).sum()
        n00 = (~frame[a] & ~frame[b]).sum()
        denom = math.sqrt(
            (n11 + n10) * (n11 + n01) * (n00 + n10) * (n00 + n01)
        )
        phi = (n11 * n00 - n10 * n01) / denom if denom else 0.0

        results.append(
            ActivityAssociation(
                activity_a=strip_prefix(a),
                activity_b=strip_prefix(b),
                lift=float(lift),
                pmi=float(pmi),
                phi=float(phi),
            )
        )

    results.sort(key=lambda x: abs(x.phi), reverse=True)
    return ActivityAssociationReport(
        n_entries=n, min_count=min_count, associations=results
    )


def lagged_activity_effects(
    df: "Dataset",
    *,
    max_lag: int = 3,
    min_count: int = 15,
) -> list[LaggedEffect]:
    """Does activity on day t predict mood on day t+lag?"""
    entry_frame = build_entry_frame(df, min_count=min_count)
    daily = build_daily_mood(df)
    acts = activity_columns(entry_frame)

    daily_acts = entry_frame.groupby("date")[acts].any()
    daily_acts.index = pd.to_datetime(daily_acts.index)
    daily_acts = daily_acts.reindex(daily.index, fill_value=False)

    results: list[LaggedEffect] = []
    mood_arr = daily["mood_mean"].values
    not_imputed = ~daily["imputed"].values

    for act in acts:
        act_arr = daily_acts[act].values.astype(bool)
        for lag in range(1, max_lag + 1):
            valid = np.arange(len(mood_arr) - lag)
            valid = valid[not_imputed[valid + lag]]

            has_act = act_arr[valid]
            future_mood = mood_arr[valid + lag]

            with_mood = future_mood[has_act]
            without_mood = future_mood[~has_act]

            if len(with_mood) < 3 or len(without_mood) < 3:
                continue

            _, p = sp_stats.mannwhitneyu(
                with_mood, without_mood, alternative="two-sided"
            )
            results.append(
                LaggedEffect(
                    activity=strip_prefix(act),
                    lag_days=lag,
                    n_with=len(with_mood),
                    n_without=len(without_mood),
                    mean_with=float(with_mood.mean()),
                    mean_without=float(without_mood.mean()),
                    delta=float(with_mood.mean() - without_mood.mean()),
                    p_value=float(p),
                )
            )

    results.sort(key=lambda e: e.p_value)
    return results
