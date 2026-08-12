"""Cleaning routines for C-MAPSS sensor frames.

Cleaning strategy (documented explicitly):
1. Drop exact duplicate rows (same values across all columns).
2. Coerce measurement columns to numeric (invalid tokens → NaN).
3. Replace ±inf with NaN, then drop rows that still contain NaN in core columns.
   We do **not** impute with zeros — zero is a physically meaningful sensor reading
   and would distort degradation signals.
4. Drop near-constant sensor channels for FD001-style analysis (variance ~ 0),
   keeping operating settings and informative sensors.
5. Sort by engine_id, cycle for deterministic downstream time-series work.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from wear_path.config import CMAPSS_COLUMNS, OPERATING_SETTING_COLS, SENSOR_COLS

logger = logging.getLogger(__name__)

CORE_ID_COLS = ("engine_id", "cycle")


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate records."""
    before = len(df)
    cleaned = df.drop_duplicates().copy()
    removed = before - len(cleaned)
    if removed:
        logger.info("Removed %s exact duplicate rows", removed)
    return cleaned


def enforce_numeric_types(df: pd.DataFrame, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Coerce selected columns to numeric; invalid values become NaN."""
    out = df.copy()
    cols = columns or CMAPSS_COLUMNS
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def replace_infinites_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Replace infinite values with NaN so they can be handled explicitly."""
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    n_inf = int(np.isinf(out[numeric_cols].to_numpy(dtype=float, copy=False)).sum())
    if n_inf:
        logger.info("Replacing %s infinite values with NaN", n_inf)
        out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return out


def drop_rows_with_missing_core(df: pd.DataFrame, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    """
    Drop rows with missing values in core identity / measurement columns.

    Strategy: prefer dropping incomplete observations over zero-imputation.
    """
    out = df.copy()
    cols = [c for c in (columns or CMAPSS_COLUMNS) if c in out.columns]
    before = len(out)
    out = out.dropna(subset=cols)
    removed = before - len(out)
    if removed:
        logger.info("Dropped %s rows with missing core values", removed)
    return out


def drop_near_constant_sensors(
    df: pd.DataFrame,
    sensor_cols: tuple[str, ...] = SENSOR_COLS,
    variance_epsilon: float = 1e-9,
) -> pd.DataFrame:
    """Drop sensor columns with near-zero variance (common in FD001)."""
    out = df.copy()
    dropped: list[str] = []
    for col in sensor_cols:
        if col not in out.columns:
            continue
        if float(out[col].var(ddof=0)) <= variance_epsilon:
            dropped.append(col)
    if dropped:
        logger.info("Dropping near-constant sensors: %s", ", ".join(dropped))
        out = out.drop(columns=dropped)
    return out


def sort_by_engine_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """Sort observations by engine_id and cycle."""
    return df.sort_values(list(CORE_ID_COLS)).reset_index(drop=True)


def clean_sensor_data(
    df: pd.DataFrame,
    *,
    drop_constant_sensors: bool = True,
) -> pd.DataFrame:
    """Apply the full cleaning sequence and return a deterministic frame."""
    logger.info("Cleaning sensor data (%s rows)", f"{len(df):,}")
    out = drop_exact_duplicates(df)
    out = enforce_numeric_types(out)
    out = replace_infinites_with_nan(out)
    # Keep whatever schema columns remain present.
    present_core = tuple(c for c in CMAPSS_COLUMNS if c in out.columns)
    out = drop_rows_with_missing_core(out, columns=present_core)
    if drop_constant_sensors:
        out = drop_near_constant_sensors(out)
    # Ensure IDs remain integer after numeric coercion.
    out["engine_id"] = out["engine_id"].astype("int64")
    out["cycle"] = out["cycle"].astype("int64")
    out = sort_by_engine_cycle(out)
    keep_settings = [c for c in OPERATING_SETTING_COLS if c in out.columns]
    keep_sensors = [c for c in SENSOR_COLS if c in out.columns]
    out = out.loc[:, ["engine_id", "cycle", *keep_settings, *keep_sensors]]
    logger.info(
        "Cleaned frame: %s rows, %s columns, %s engines",
        f"{len(out):,}",
        out.shape[1],
        out["engine_id"].nunique(),
    )
    return out
