"""Tests for predictive-maintenance feature engineering."""

from __future__ import annotations

import pandas as pd
import pytest

from wear_path.features import (
    add_normalized_cycle,
    add_remaining_useful_life,
    engineer_features,
)


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 1, 2, 2, 2],
            "cycle": [1, 2, 3, 4, 1, 2, 3],
            "op_setting_1": [0.0] * 7,
            "sensor_02": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "sensor_03": [2.0, 2.5, 3.0, 3.5, 1.0, 1.5, 2.0],
            "sensor_04": [10.0, 11.0, 12.0, 13.0, 8.0, 9.0, 10.0],
            "sensor_07": [1.0] * 7,
            "sensor_11": [4.0, 4.1, 4.2, 4.3, 5.0, 5.1, 5.2],
            "sensor_12": [7.0] * 7,
            "sensor_15": [0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3],
        }
    )


def test_rul_calculation() -> None:
    out = add_remaining_useful_life(_df())
    engine1 = out[out["engine_id"] == 1].sort_values("cycle")
    assert engine1["rul"].tolist() == [3, 2, 1, 0]
    engine2 = out[out["engine_id"] == 2].sort_values("cycle")
    assert engine2["rul"].tolist() == [2, 1, 0]


def test_normalized_cycle() -> None:
    out = add_normalized_cycle(_df())
    e1 = out[out["engine_id"] == 1].sort_values("cycle")
    assert e1["cycle_norm"].iloc[-1] == pytest.approx(1.0)
    assert e1["cycle_norm"].iloc[0] == pytest.approx(0.25)


def test_engineer_features_contains_expected_columns() -> None:
    out = engineer_features(_df(), sensor_cols=("sensor_02", "sensor_04"), window=2)
    for col in (
        "engine_id",
        "cycle",
        "sensor_02",
        "sensor_04",
        "sensor_02_delta",
        "sensor_02_rolling_mean",
        "sensor_02_rolling_std",
        "cycle_norm",
        "rul",
    ):
        assert col in out.columns
    assert out["rul"].min() == 0
