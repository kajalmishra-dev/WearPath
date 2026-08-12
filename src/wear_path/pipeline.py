"""End-to-end WearPath batch pipeline orchestration."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from wear_path.cleaning import clean_sensor_data
from wear_path.config import (
    DEFAULT_DATASET_VARIANT,
    FEATURES_PARQUET,
    OUTPUTS_DIR,
    PIPELINE_SUMMARY_PATH,
    PROCESSED_DATA_DIR,
    QUALITY_REPORT_PATH,
)
from wear_path.database import build_analytics_database
from wear_path.features import engineer_features
from wear_path.ingestion import load_raw_data
from wear_path.validation import DataQualityError, run_quality_checks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("wear_path.pipeline")


def write_parquet(df: pd.DataFrame, path: Path = FEATURES_PARQUET) -> Path:
    """Write the analytics dataset to Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info("Wrote Parquet dataset: %s (%s rows)", path, f"{len(df):,}")
    return path


def run_pipeline(
    variant: str = DEFAULT_DATASET_VARIANT,
    split: str = "train",
    fail_on_critical: bool = True,
) -> dict[str, Any]:
    """
    Execute the full batch pipeline:

    RAW → INGEST → VALIDATE → CLEAN → TRANSFORM/FEATURES →
    QUALITY CHECK → PARQUET → DUCKDB → SQL → REPORTS
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "variant": variant,
        "split": split,
        "stages": {},
    }

    logger.info("=== STAGE 1: INGEST (%s %s) ===", variant, split)
    raw = load_raw_data(variant=variant, split=split)
    summary["stages"]["ingest"] = {
        "rows": int(len(raw)),
        "columns": int(raw.shape[1]),
        "engines": int(raw["engine_id"].nunique()),
    }

    logger.info("=== STAGE 2: VALIDATE (raw) ===")
    raw_report = run_quality_checks(
        raw,
        stage="raw",
        save_path=OUTPUTS_DIR / "data_quality_report_raw.json",
        fail_on_critical=fail_on_critical,
    )
    summary["stages"]["validate_raw"] = {
        "passed": raw_report.passed,
        "duplicate_count": raw_report.duplicate_count,
        "critical_failures": raw_report.critical_failures,
    }

    logger.info("=== STAGE 3: CLEAN ===")
    cleaned = clean_sensor_data(raw)
    summary["stages"]["clean"] = {
        "rows": int(len(cleaned)),
        "columns": int(cleaned.shape[1]),
    }

    logger.info("=== STAGE 4: FEATURE ENGINEERING ===")
    features = engineer_features(cleaned, include_rul=(split == "train"))
    summary["stages"]["features"] = {
        "rows": int(len(features)),
        "columns": int(features.shape[1]),
        "feature_columns": list(features.columns),
    }

    logger.info("=== STAGE 5: QUALITY CHECK (final) ===")
    final_required = ("engine_id", "cycle", "rul") if "rul" in features.columns else ("engine_id", "cycle")
    final_report = run_quality_checks(
        features,
        stage="final",
        save_path=QUALITY_REPORT_PATH,
        fail_on_critical=False,
        required_columns=final_required,
    )
    # Critical final checks: non-empty, positive ids/cycles.
    if len(features) == 0:
        raise DataQualityError("Final feature dataset is empty")
    if (features["engine_id"] <= 0).any() or (features["cycle"] <= 0).any():
        raise DataQualityError("Final dataset contains invalid engine_id or cycle values")
    summary["stages"]["validate_final"] = {
        "rows": final_report.row_count,
        "engines": final_report.n_engines,
        "cycle_min": final_report.cycle_min,
        "cycle_max": final_report.cycle_max,
        "duplicate_count": final_report.duplicate_count,
        "missing_value_counts": final_report.missing_value_counts,
    }

    logger.info("=== STAGE 6: WRITE PARQUET ===")
    parquet_path = write_parquet(features, FEATURES_PARQUET)
    summary["stages"]["parquet"] = {"path": str(parquet_path)}

    logger.info("=== STAGE 7: DUCKDB + SQL ANALYTICS ===")
    sql_results = build_analytics_database(parquet_path=parquet_path)
    summary["stages"]["sql"] = {
        name: {"rows": int(len(df)), "columns": list(df.columns)[:12]}
        for name, df in sql_results.items()
    }

    # Compact engine summary artifact for README / reviewers.
    engine_csv = OUTPUTS_DIR / "sql_results" / "engine_summary.csv"
    if engine_csv.is_file():
        engine_df = pd.read_csv(engine_csv)
        summary["sample_engine_stats"] = {
            "n_engines_in_summary": int(len(engine_df)),
            "avg_cycles_mean": float(engine_df["n_cycles"].mean()) if "n_cycles" in engine_df else None,
        }

    PIPELINE_SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("=== PIPELINE COMPLETE ===")
    logger.info("Summary written to %s", PIPELINE_SUMMARY_PATH)
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = list(sys.argv[1:] if argv is None else argv)
    variant = DEFAULT_DATASET_VARIANT
    split = "train"
    if "--variant" in args:
        variant = args[args.index("--variant") + 1]
    if "--split" in args:
        split = args[args.index("--split") + 1]
    try:
        run_pipeline(variant=variant, split=split)
    except Exception:
        logger.exception("Pipeline failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
