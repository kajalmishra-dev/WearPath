"""Ingest NASA C-MAPSS turbofan sensor files into Pandas DataFrames."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from wear_path.config import (
    CMAPSS_COLUMNS,
    DEFAULT_DATASET_VARIANT,
    RAW_DATA_DIR,
)

logger = logging.getLogger(__name__)


class DatasetNotFoundError(FileNotFoundError):
    """Raised when expected raw C-MAPSS files are missing."""


def _resolve_raw_dir(raw_dir: Path | None = None) -> Path:
    return Path(raw_dir) if raw_dir is not None else RAW_DATA_DIR


def locate_variant_file(
    variant: str = DEFAULT_DATASET_VARIANT,
    split: str = "train",
    raw_dir: Path | None = None,
) -> Path:
    """
    Locate a C-MAPSS text file for a given variant and split.

    Looks in ``data/raw`` and one-level subdirectories (e.g. ``CMAPSSData/``).
    """
    base = _resolve_raw_dir(raw_dir)
    filename = f"{split}_{variant}.txt"
    candidates = [
        base / filename,
        base / "CMAPSSData" / filename,
        base / "CMaps" / filename,
        base / "cmapss" / filename,
    ]
    # Also search one level of arbitrary nested folders.
    if base.exists():
        for child in base.iterdir():
            if child.is_dir():
                candidates.append(child / filename)

    for path in candidates:
        if path.is_file():
            logger.info("Located %s at %s", filename, path)
            return path

    searched = ", ".join(str(p) for p in candidates[:4])
    raise DatasetNotFoundError(
        f"Could not find {filename} under {base}. "
        f"Searched: {searched}. "
        "Run: python scripts/download_cmapss.py"
    )


def load_dataset_variant(
    variant: str = DEFAULT_DATASET_VARIANT,
    split: str = "train",
    raw_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Load one C-MAPSS variant/split into a typed DataFrame.

    Parameters
    ----------
    variant:
        Dataset subset such as ``FD001``.
    split:
        ``train`` or ``test``.
    raw_dir:
        Optional override for the raw data directory.
    """
    path = locate_variant_file(variant=variant, split=split, raw_dir=raw_dir)
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=list(CMAPSS_COLUMNS),
        engine="python",
    )
    # Trailing whitespace can produce an empty column in some dumps.
    df = df.dropna(axis=1, how="all")
    if list(df.columns) != list(CMAPSS_COLUMNS):
        # Keep only the expected schema columns if extras appeared.
        missing = [c for c in CMAPSS_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing expected columns after parse: {missing}")
        df = df.loc[:, list(CMAPSS_COLUMNS)]

    df["engine_id"] = df["engine_id"].astype("int64")
    df["cycle"] = df["cycle"].astype("int64")
    for col in CMAPSS_COLUMNS[2:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.attrs["variant"] = variant
    df.attrs["split"] = split
    df.attrs["source_path"] = str(path)

    logger.info(
        "Loaded %s %s: %s rows, %s columns, %s engines from %s",
        variant,
        split,
        f"{len(df):,}",
        df.shape[1],
        df["engine_id"].nunique(),
        path.name,
    )
    return df


def load_rul_targets(
    variant: str = DEFAULT_DATASET_VARIANT,
    raw_dir: Path | None = None,
) -> pd.DataFrame:
    """Load official RUL labels for the test split (one value per engine)."""
    path = locate_variant_file(variant=variant, split="RUL", raw_dir=raw_dir)
    rul = pd.read_csv(path, sep=r"\s+", header=None, names=["rul"], engine="python")
    rul = rul.dropna(axis=1, how="all")
    rul["engine_id"] = range(1, len(rul) + 1)
    rul["engine_id"] = rul["engine_id"].astype("int64")
    rul["rul"] = pd.to_numeric(rul["rul"], errors="coerce").astype("float64")
    logger.info("Loaded RUL targets for %s: %s engines", variant, len(rul))
    return rul[["engine_id", "rul"]]


def load_raw_data(
    variant: str = DEFAULT_DATASET_VARIANT,
    split: str = "train",
    raw_dir: Path | None = None,
) -> pd.DataFrame:
    """Convenience wrapper used by the pipeline to load primary raw data."""
    return load_dataset_variant(variant=variant, split=split, raw_dir=raw_dir)


def list_available_files(raw_dir: Path | None = None) -> list[Path]:
    """Return discovered ``.txt`` files under the raw data directory."""
    base = _resolve_raw_dir(raw_dir)
    if not base.exists():
        return []
    return sorted(base.rglob("*.txt"))
