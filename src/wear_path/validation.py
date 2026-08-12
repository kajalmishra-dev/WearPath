"""Data validation and structured quality reporting for C-MAPSS frames."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wear_path.config import CMAPSS_COLUMNS, OUTPUTS_DIR, QUALITY_REPORT_PATH

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Structured quality metrics for a sensor DataFrame."""

    row_count: int
    column_count: int
    missing_value_counts: dict[str, int]
    duplicate_count: int
    invalid_value_counts: dict[str, int]
    n_engines: int
    cycle_min: int | None
    cycle_max: int | None
    checks: dict[str, bool] = field(default_factory=dict)
    critical_failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def passed(self) -> bool:
        return len(self.critical_failures) == 0


class DataQualityError(ValueError):
    """Raised when critical quality checks fail."""


def _missing_counts(df: pd.DataFrame) -> dict[str, int]:
    return {col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().any()}


def _invalid_counts(df: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        n_inf = int(np.isinf(df[col].to_numpy(dtype=float, copy=False)).sum())
        if n_inf:
            counts[f"{col}_infinite"] = n_inf
    if "engine_id" in df.columns:
        n_bad = int((df["engine_id"] <= 0).sum())
        if n_bad:
            counts["engine_id_non_positive"] = n_bad
    if "cycle" in df.columns:
        n_bad = int((df["cycle"] <= 0).sum())
        if n_bad:
            counts["cycle_non_positive"] = n_bad
    return counts


def validate_schema(df: pd.DataFrame, required_columns: tuple[str, ...] = CMAPSS_COLUMNS) -> list[str]:
    """Return missing required column names."""
    return [c for c in required_columns if c not in df.columns]


def validate_numeric_types(df: pd.DataFrame, columns: tuple[str, ...] | None = None) -> list[str]:
    """Return columns that are not numeric."""
    cols = columns or CMAPSS_COLUMNS
    bad: list[str] = []
    for col in cols:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            bad.append(col)
    return bad


def check_engine_cycle_ordering(df: pd.DataFrame) -> bool:
    """
    Return True if each engine's cycles are strictly increasing after sort key order.

    Expects rows already represent one observation per engine/cycle.
    """
    if "engine_id" not in df.columns or "cycle" not in df.columns:
        return False
    grouped = df.groupby("engine_id", sort=False)["cycle"]
    for _, cycles in grouped:
        values = cycles.to_numpy()
        if len(values) > 1 and not np.all(values[1:] > values[:-1]):
            return False
    return True


def build_quality_report(
    df: pd.DataFrame,
    stage: str = "raw",
    required_columns: tuple[str, ...] | None = None,
) -> DataQualityReport:
    """Run quality checks and return a structured report."""
    required = required_columns if required_columns is not None else CMAPSS_COLUMNS
    missing_required = validate_schema(df, required_columns=required)
    non_numeric = validate_numeric_types(df, columns=required)
    duplicate_count = int(df.duplicated().sum())
    missing_value_counts = _missing_counts(df)
    invalid_value_counts = _invalid_counts(df)

    n_engines = int(df["engine_id"].nunique()) if "engine_id" in df.columns else 0
    cycle_min = int(df["cycle"].min()) if "cycle" in df.columns and len(df) else None
    cycle_max = int(df["cycle"].max()) if "cycle" in df.columns and len(df) else None

    ordering_ok = False
    if "engine_id" in df.columns and "cycle" in df.columns and len(df):
        # Ordering check on sorted copy so unsorted input is still flagged separately.
        ordered = df.sort_values(["engine_id", "cycle"])
        ordering_ok = check_engine_cycle_ordering(ordered)

    checks = {
        "required_columns_present": len(missing_required) == 0,
        "numeric_types_valid": len(non_numeric) == 0,
        "engine_id_valid": invalid_value_counts.get("engine_id_non_positive", 0) == 0,
        "cycle_positive": invalid_value_counts.get("cycle_non_positive", 0) == 0,
        "no_infinite_values": not any(k.endswith("_infinite") for k in invalid_value_counts),
        "no_exact_duplicates": duplicate_count == 0,
        "engine_cycle_ordering_valid": ordering_ok,
        "has_rows": len(df) > 0,
    }

    critical_failures: list[str] = []
    if not checks["required_columns_present"]:
        critical_failures.append(f"missing_columns:{missing_required}")
    if not checks["has_rows"]:
        critical_failures.append("empty_dataframe")
    if not checks["numeric_types_valid"]:
        critical_failures.append(f"non_numeric_columns:{non_numeric}")
    if not checks["engine_id_valid"]:
        critical_failures.append("invalid_engine_id")
    if not checks["cycle_positive"]:
        critical_failures.append("non_positive_cycle")
    if not checks["no_infinite_values"]:
        critical_failures.append("infinite_values_present")

    report = DataQualityReport(
        row_count=int(len(df)),
        column_count=int(df.shape[1]),
        missing_value_counts=missing_value_counts,
        duplicate_count=duplicate_count,
        invalid_value_counts=invalid_value_counts,
        n_engines=n_engines,
        cycle_min=cycle_min,
        cycle_max=cycle_max,
        checks=checks,
        critical_failures=critical_failures,
    )
    report_dict = report.to_dict()
    report_dict["stage"] = stage
    logger.info(
        "Quality report (%s): rows=%s engines=%s duplicates=%s critical_failures=%s",
        stage,
        report.row_count,
        report.n_engines,
        report.duplicate_count,
        report.critical_failures or "none",
    )
    return report


def save_quality_report(
    report: DataQualityReport,
    path: Path | None = None,
    stage: str = "raw",
) -> Path:
    """Persist a quality report as JSON under outputs/."""
    out = Path(path) if path is not None else QUALITY_REPORT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    payload["stage"] = stage
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote quality report to %s", out)
    return out


def assert_quality(report: DataQualityReport, *, fail_on_critical: bool = True) -> None:
    """Optionally fail the pipeline when critical checks fail."""
    if fail_on_critical and not report.passed:
        raise DataQualityError(
            "Critical data quality checks failed: " + ", ".join(report.critical_failures)
        )


def run_quality_checks(
    df: pd.DataFrame,
    *,
    stage: str = "raw",
    save_path: Path | None = None,
    fail_on_critical: bool = False,
    required_columns: tuple[str, ...] | None = None,
) -> DataQualityReport:
    """Reusable quality-check entry point used by the pipeline and notebook."""
    report = build_quality_report(df, stage=stage, required_columns=required_columns)
    if save_path is not None or stage in {"raw", "final"}:
        target = save_path
        if target is None and stage == "final":
            target = OUTPUTS_DIR / "data_quality_report.json"
        elif target is None:
            target = OUTPUTS_DIR / f"data_quality_report_{stage}.json"
        save_quality_report(report, path=target, stage=stage)
    assert_quality(report, fail_on_critical=fail_on_critical)
    return report
