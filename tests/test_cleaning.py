"""Tests for cleaning behavior."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wear_path.cleaning import (
    clean_sensor_data,
    drop_exact_duplicates,
    enforce_numeric_types,
    replace_infinites_with_nan,
    sort_by_engine_cycle,
)
from wear_path.config import CMAPSS_COLUMNS


def _frame() -> pd.DataFrame:
    data = {
        "engine_id": [2, 2, 1, 1, 1],
        "cycle": [2, 1, 2, 1, 1],
        "op_setting_1": [0.0, 0.0, 0.0, 0.0, 0.0],
        "op_setting_2": [0.0, 0.0, 0.0, 0.0, 0.0],
        "op_setting_3": [100.0, 100.0, 100.0, 100.0, 100.0],
    }
    for i in range(1, 22):
        # Make most sensors vary; keep sensor_01 constant to exercise drop logic.
        if i == 1:
            data[f"sensor_{i:02d}"] = [10.0, 10.0, 10.0, 10.0, 10.0]
        else:
            data[f"sensor_{i:02d}"] = [float(i + c) for c in (2, 1, 2, 1, 1)]
    return pd.DataFrame(data)[list(CMAPSS_COLUMNS)]


def test_drop_exact_duplicates() -> None:
    df = _frame()
    out = drop_exact_duplicates(df)
    assert len(out) == len(df) - 1


def test_sort_by_engine_cycle() -> None:
    df = drop_exact_duplicates(_frame())
    out = sort_by_engine_cycle(df)
    assert out["engine_id"].tolist() == [1, 1, 2, 2]
    assert out["cycle"].tolist() == [1, 2, 1, 2]


def test_enforce_numeric_and_infinites() -> None:
    df = _frame().head(3).copy()
    # Simulate dirty ingestion: object dtype with a non-numeric token.
    df["sensor_02"] = df["sensor_02"].astype(object)
    df.loc[0, "sensor_02"] = "bad"
    df.loc[1, "sensor_03"] = np.inf
    out = enforce_numeric_types(df)
    assert pd.isna(out.loc[0, "sensor_02"])
    out = replace_infinites_with_nan(out)
    assert pd.isna(out.loc[1, "sensor_03"])


def test_clean_sensor_data_pipeline() -> None:
    out = clean_sensor_data(_frame())
    assert out["engine_id"].is_monotonic_increasing or set(out["engine_id"]) == {1, 2}
    assert list(out.columns[:2]) == ["engine_id", "cycle"]
    assert "sensor_01" not in out.columns  # near-constant dropped
    assert out["cycle"].min() >= 1
    assert not out.duplicated().any()
