"""Statistical aggregation: median, CI, CV%."""

from __future__ import annotations

import math
from dataclasses import dataclass


# Hardcoded t-values for 95% CI (two-tailed) to avoid scipy dependency
# Key: degrees of freedom, Value: t-value
_T_TABLE = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    25: 2.060, 30: 2.042, 40: 2.021, 50: 2.009, 60: 2.000,
    80: 1.990, 100: 1.984, 200: 1.972,
}


def _get_t_value(df: int) -> float:
    """Get t-value for given degrees of freedom, interpolating if needed."""
    if df in _T_TABLE:
        return _T_TABLE[df]
    if df >= 200:
        return 1.96  # approximate z-value
    # Find nearest keys and interpolate
    keys = sorted(_T_TABLE.keys())
    for i in range(len(keys) - 1):
        if keys[i] <= df <= keys[i + 1]:
            lo, hi = keys[i], keys[i + 1]
            frac = (df - lo) / (hi - lo)
            return _T_TABLE[lo] + frac * (_T_TABLE[hi] - _T_TABLE[lo])
    return 1.96


@dataclass
class AggregatedMetric:
    median: float
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    cv_percent: float
    n: int
    values: list[float]
    unreliable: bool = False  # True if CV% > 10%

    @property
    def ci_str(self) -> str:
        return f"[{self.ci_lower:.2f}, {self.ci_upper:.2f}]"


def aggregate(values: list[float]) -> AggregatedMetric:
    """Compute median, mean, std, 95% CI, and CV% for a list of values."""
    n = len(values)
    if n == 0:
        return AggregatedMetric(
            median=0.0, mean=0.0, std=0.0,
            ci_lower=0.0, ci_upper=0.0, cv_percent=0.0,
            n=0, values=[], unreliable=True,
        )

    if n == 1:
        v = values[0]
        return AggregatedMetric(
            median=v, mean=v, std=0.0,
            ci_lower=v, ci_upper=v, cv_percent=0.0,
            n=1, values=list(values),
        )

    sorted_vals = sorted(values)

    # Median
    mid = n // 2
    if n % 2 == 0:
        median = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    else:
        median = sorted_vals[mid]

    # Mean
    mean = sum(values) / n

    # Standard deviation (sample)
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(variance)

    # CV%
    cv_percent = (std / mean * 100.0) if mean != 0 else 0.0

    # 95% CI
    df = n - 1
    t = _get_t_value(df)
    se = std / math.sqrt(n)
    ci_lower = mean - t * se
    ci_upper = mean + t * se

    return AggregatedMetric(
        median=median,
        mean=mean,
        std=std,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        cv_percent=cv_percent,
        n=n,
        values=list(values),
        unreliable=cv_percent > 10.0,
    )
