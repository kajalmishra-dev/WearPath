"""Tests for validation and quality reporting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wear_path.config import CMAPSS_COLUMNS
from wear_path.validation import (
    DataQualityError,
    assert_quality,
    build_quality_report,
    check_engine_cycle_ordering,
    validate_schema,
)


def _valid_df() -> pd.DataFrame:
    data = {col: [] for col in CMAPSS_COLUMNS}
    for engine in (1, 2):
        for cycle in (1, 2, 3):
            data["engine_id"].append(engine)
            data["cycle"].append(cycle)
            data["op_setting_1"].append(0.0)
            data["op_setting_2"].append(0.0)
            data["op_setting_3"].append(100.0)
            for i in range(1, 22):
                data[f"sensor_{i:02d}"].append(float(i + cycle))
    return pd.DataFrame(data)


def test_validate_schema_detects_missing() -> None:
    df = _valid_df().drop(columns=["sensor_21"])
    missing = validate_schema(df)
    assert "sensor_21" in missing


def test_build_quality_report_counts() -> None:
    df = _valid_df()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    report = build_quality_report(df, stage="test")
    assert report.row_count == 7
    assert report.n_engines == 2
    assert report.duplicate_count == 1
    assert report.cycle_min == 1
    assert report.cycle_max == 3
    assert report.checks["required_columns_present"] is True


def test_invalid_engine_and_cycle_fail_critical() -> None:
    df = _valid_df()
    df.loc[0, "engine_id"] = 0
    df.loc[1, "cycle"] = -1
    df.loc[2, "sensor_02"] = np.inf
    report = build_quality_report(df)
    assert not report.passed
    with pytest.raises(DataQualityError):
        assert_quality(report, fail_on_critical=True)


def test_engine_cycle_ordering() -> None:
    df = _valid_df()
    assert check_engine_cycle_ordering(df) is True
    bad = df.copy()
    bad.loc[bad["engine_id"] == 1, "cycle"] = [3, 2, 1]
    assert check_engine_cycle_ordering(bad) is False
