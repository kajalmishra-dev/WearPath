"""Tests for C-MAPSS ingestion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from wear_path.config import CMAPSS_COLUMNS
from wear_path.ingestion import DatasetNotFoundError, load_dataset_variant, load_raw_data


def _write_mini_cmapss(path: Path) -> None:
    # Two engines, a few cycles, 3 settings + 21 sensors
    rows = []
    for engine in (1, 2):
        for cycle in (1, 2, 3):
            settings = [0.0, 0.0, 100.0]
            sensors = [float(i + engine + cycle) for i in range(21)]
            rows.append(" ".join(str(x) for x in [engine, cycle, *settings, *sensors]))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_load_dataset_variant_assigns_columns(tmp_path: Path) -> None:
    raw = tmp_path / "CMAPSSData"
    raw.mkdir()
    _write_mini_cmapss(raw / "train_FD001.txt")
    df = load_dataset_variant(variant="FD001", split="train", raw_dir=tmp_path)
    assert list(df.columns) == list(CMAPSS_COLUMNS)
    assert len(df) == 6
    assert df["engine_id"].tolist() == [1, 1, 1, 2, 2, 2]
    assert df["cycle"].min() == 1
    assert pd.api.types.is_numeric_dtype(df["sensor_01"])


def test_load_raw_data_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetNotFoundError):
        load_raw_data(raw_dir=tmp_path)


def test_required_columns_present(tmp_path: Path) -> None:
    raw = tmp_path / "CMAPSSData"
    raw.mkdir()
    _write_mini_cmapss(raw / "train_FD001.txt")
    df = load_raw_data(raw_dir=tmp_path)
    for col in ("engine_id", "cycle", "op_setting_1", "sensor_01", "sensor_21"):
        assert col in df.columns
