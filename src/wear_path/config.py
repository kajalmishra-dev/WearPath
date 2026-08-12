"""Project paths and C-MAPSS schema configuration."""

from __future__ import annotations

from pathlib import Path

# Repository root: .../WearPath (parent of src/)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
SQL_DIR: Path = PROJECT_ROOT / "sql"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

# Default analytical artifacts
FEATURES_PARQUET: Path = PROCESSED_DATA_DIR / "sensor_features.parquet"
DUCKDB_PATH: Path = OUTPUTS_DIR / "wear_path.duckdb"
QUALITY_REPORT_PATH: Path = OUTPUTS_DIR / "data_quality_report.json"
PIPELINE_SUMMARY_PATH: Path = OUTPUTS_DIR / "pipeline_summary.json"
SQL_RESULTS_DIR: Path = OUTPUTS_DIR / "sql_results"

# C-MAPSS FD001 is the primary variant used by this project.
DEFAULT_DATASET_VARIANT: str = "FD001"

# Official / documented download sources (checked at setup time).
# Primary: NASA Prognostics Center of Excellence C-MAPSS archive.
CMAPSS_DOWNLOAD_URLS: tuple[str, ...] = (
    # NASA / data.gov style mirrors commonly used for C-MAPSS redistribution.
    "https://ti.arc.nasa.gov/c/6/",
    "https://data.nasa.gov/download/vrks-gjie/application%2Fzip",
)

# Space-separated C-MAPSS train/test files have 26 columns:
# engine_id, cycle, 3 operating settings, 21 sensors.
OPERATING_SETTING_COLS: tuple[str, ...] = (
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
)
SENSOR_COLS: tuple[str, ...] = tuple(f"sensor_{i:02d}" for i in range(1, 22))
CMAPSS_COLUMNS: tuple[str, ...] = (
    "engine_id",
    "cycle",
    *OPERATING_SETTING_COLS,
    *SENSOR_COLS,
)

# Sensors commonly retained for FD001 feature engineering after dropping
# near-constant channels (documented in cleaning / features modules).
FEATURE_SENSOR_COLS: tuple[str, ...] = (
    "sensor_02",
    "sensor_03",
    "sensor_04",
    "sensor_07",
    "sensor_11",
    "sensor_12",
    "sensor_15",
)

ROLLING_WINDOW: int = 5
