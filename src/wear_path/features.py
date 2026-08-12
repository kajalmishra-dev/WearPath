"""Predictive-maintenance feature engineering for C-MAPSS sequences.

Remaining Useful Life (RUL)
---------------------------
For the **training** split, each engine runs until failure. The final observed
cycle for an engine is the end of its run. Therefore:

    RUL(engine, cycle) = max_cycle(engine) - cycle

This is the standard label construction for C-MAPSS training data and does not
invent labels beyond what the run-to-failure trajectories support.

For the **test** split, engines are censored before failure; official RUL
targets are provided in ``RUL_*.txt`` and can be joined separately.
"""

from __future__ import annotations

import logging

import pandas as pd

from wear_path.config import FEATURE_SENSOR_COLS, ROLLING_WINDOW
from wear_path.transformations import apply_transformations

logger = logging.getLogger(__name__)


def add_normalized_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """Cycle position within each engine run, scaled to [0, 1]."""
    out = df.sort_values(["engine_id", "cycle"]).copy()
    max_cycle = out.groupby("engine_id")["cycle"].transform("max")
    out["cycle_norm"] = out["cycle"] / max_cycle
    return out


def add_remaining_useful_life(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive RUL for run-to-failure (training) trajectories.

    RUL = max_cycle(engine) - cycle
    """
    out = df.sort_values(["engine_id", "cycle"]).copy()
    max_cycle = out.groupby("engine_id")["cycle"].transform("max")
    out["rul"] = (max_cycle - out["cycle"]).astype("int64")
    logger.info(
        "Derived RUL: min=%s max=%s mean=%.2f",
        int(out["rul"].min()),
        int(out["rul"].max()),
        float(out["rul"].mean()),
    )
    return out


def select_feature_columns(
    df: pd.DataFrame,
    sensor_cols: tuple[str, ...] | None = None,
) -> list[str]:
    """Return the analytics column set: ids + base sensors + engineered features."""
    sensors = [c for c in (sensor_cols or FEATURE_SENSOR_COLS) if c in df.columns]
    engineered: list[str] = []
    for col in sensors:
        for suffix in ("_norm", "_delta", "_rolling_mean", "_rolling_std"):
            name = f"{col}{suffix}"
            if name in df.columns:
                engineered.append(name)
    optional = [c for c in ("cycle_norm", "rul") if c in df.columns]
    base = ["engine_id", "cycle", *sensors]
    # Preserve operating settings when present.
    settings = [c for c in df.columns if c.startswith("op_setting_")]
    return base + settings + engineered + optional


def engineer_features(
    df: pd.DataFrame,
    *,
    sensor_cols: tuple[str, ...] | None = None,
    window: int = ROLLING_WINDOW,
    include_rul: bool = True,
) -> pd.DataFrame:
    """
    Build the analytics-ready feature set.

    Steps:
    1. Normalized cycle position
    2. Sensor transforms (norm / delta / rolling)
    3. RUL for run-to-failure data (optional)
    """
    sensors = tuple(c for c in (sensor_cols or FEATURE_SENSOR_COLS) if c in df.columns)
    if not sensors:
        raise ValueError("No feature sensor columns present in DataFrame")

    logger.info("Engineering features for sensors: %s", ", ".join(sensors))
    out = add_normalized_cycle(df)
    out = apply_transformations(out, sensor_cols=sensors, window=window)
    if include_rul:
        out = add_remaining_useful_life(out)

    keep = select_feature_columns(out, sensor_cols=sensors)
    out = out.loc[:, keep].reset_index(drop=True)
    logger.info("Feature frame shape: %s", out.shape)
    return out
