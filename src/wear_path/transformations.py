"""Reusable time-series transformations for per-engine sensor sequences."""

from __future__ import annotations

import logging

import pandas as pd

from wear_path.config import FEATURE_SENSOR_COLS, ROLLING_WINDOW

logger = logging.getLogger(__name__)


def sort_observations(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by engine and cycle."""
    return df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)


def normalize_sensors(
    df: pd.DataFrame,
    sensor_cols: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """
    Z-score normalize selected sensors using global train statistics.

    Adds columns named ``{sensor}_norm``.
    """
    out = sort_observations(df)
    cols = [c for c in (sensor_cols or FEATURE_SENSOR_COLS) if c in out.columns]
    for col in cols:
        mean = out[col].mean()
        std = out[col].std(ddof=0)
        if std == 0 or pd.isna(std):
            out[f"{col}_norm"] = 0.0
        else:
            out[f"{col}_norm"] = (out[col] - mean) / std
    logger.info("Normalized sensors: %s", ", ".join(cols))
    return out


def add_previous_reading(
    df: pd.DataFrame,
    sensor_cols: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Add previous cycle reading per engine for selected sensors."""
    out = sort_observations(df)
    cols = [c for c in (sensor_cols or FEATURE_SENSOR_COLS) if c in out.columns]
    for col in cols:
        out[f"{col}_prev"] = out.groupby("engine_id")[col].shift(1)
    return out


def add_sensor_delta(
    df: pd.DataFrame,
    sensor_cols: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Cycle-to-cycle sensor change within each engine."""
    out = sort_observations(df)
    cols = [c for c in (sensor_cols or FEATURE_SENSOR_COLS) if c in out.columns]
    for col in cols:
        out[f"{col}_delta"] = out.groupby("engine_id")[col].diff()
    logger.info("Added deltas for %s sensors", len(cols))
    return out


def add_rate_of_change(
    df: pd.DataFrame,
    sensor_cols: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """
    Rate of change ≈ delta / cycle_delta.

    Since C-MAPSS cycles increment by 1, this equals the sensor delta when
    observations are contiguous.
    """
    out = sort_observations(df)
    cols = [c for c in (sensor_cols or FEATURE_SENSOR_COLS) if c in out.columns]
    cycle_delta = out.groupby("engine_id")["cycle"].diff()
    for col in cols:
        sensor_delta = out.groupby("engine_id")[col].diff()
        out[f"{col}_roc"] = sensor_delta / cycle_delta.replace(0, pd.NA)
    return out


def add_rolling_stats(
    df: pd.DataFrame,
    sensor_cols: tuple[str, ...] | None = None,
    window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    """Per-engine rolling mean and standard deviation."""
    if window < 2:
        raise ValueError("window must be >= 2")
    out = sort_observations(df)
    cols = [c for c in (sensor_cols or FEATURE_SENSOR_COLS) if c in out.columns]
    for col in cols:
        grouped = out.groupby("engine_id")[col]
        out[f"{col}_rolling_mean"] = grouped.transform(
            lambda s: s.rolling(window=window, min_periods=1).mean()
        )
        out[f"{col}_rolling_std"] = grouped.transform(
            lambda s: s.rolling(window=window, min_periods=1).std()
        )
    logger.info("Added rolling stats (window=%s) for %s sensors", window, len(cols))
    return out


def apply_transformations(
    df: pd.DataFrame,
    sensor_cols: tuple[str, ...] | None = None,
    window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    """Apply the standard transformation suite used by the pipeline."""
    cols = sensor_cols or FEATURE_SENSOR_COLS
    present = tuple(c for c in cols if c in df.columns)
    out = sort_observations(df)
    out = normalize_sensors(out, sensor_cols=present)
    out = add_sensor_delta(out, sensor_cols=present)
    out = add_rolling_stats(out, sensor_cols=present, window=window)
    return out
