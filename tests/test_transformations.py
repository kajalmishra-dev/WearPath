"""Tests for time-series transformations."""

from __future__ import annotations

import pandas as pd
import pytest

from wear_path.transformations import (
    add_rolling_stats,
    add_sensor_delta,
    normalize_sensors,
    sort_observations,
)


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 2, 2, 2],
            "cycle": [1, 2, 3, 1, 2, 3],
            "sensor_02": [1.0, 2.0, 4.0, 10.0, 10.0, 12.0],
            "sensor_04": [5.0, 5.0, 7.0, 1.0, 3.0, 5.0],
        }
    )


def test_sort_observations() -> None:
    shuffled = _df().sample(frac=1.0, random_state=0)
    out = sort_observations(shuffled)
    assert out["engine_id"].tolist() == [1, 1, 1, 2, 2, 2]
    assert out["cycle"].tolist() == [1, 2, 3, 1, 2, 3]


def test_sensor_delta_is_per_engine() -> None:
    out = add_sensor_delta(_df(), sensor_cols=("sensor_02",))
    # First row per engine is NaN; engine 1: 2-1=1, 4-2=2
    assert pd.isna(out.loc[0, "sensor_02_delta"])
    assert out.loc[1, "sensor_02_delta"] == pytest.approx(1.0)
    assert out.loc[2, "sensor_02_delta"] == pytest.approx(2.0)
    assert pd.isna(out.loc[3, "sensor_02_delta"])
    assert out.loc[4, "sensor_02_delta"] == pytest.approx(0.0)


def test_rolling_mean_window() -> None:
    out = add_rolling_stats(_df(), sensor_cols=("sensor_02",), window=2)
    assert out.loc[0, "sensor_02_rolling_mean"] == pytest.approx(1.0)
    assert out.loc[1, "sensor_02_rolling_mean"] == pytest.approx(1.5)
    assert out.loc[2, "sensor_02_rolling_mean"] == pytest.approx(3.0)
    # Engine 2 should not leak engine 1 values
    assert out.loc[3, "sensor_02_rolling_mean"] == pytest.approx(10.0)


def test_normalize_sensors() -> None:
    out = normalize_sensors(_df(), sensor_cols=("sensor_02",))
    assert "sensor_02_norm" in out.columns
    assert abs(out["sensor_02_norm"].mean()) < 1e-9
