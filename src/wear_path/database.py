"""DuckDB analytical store and SQL analytics for processed sensor features."""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

from wear_path.config import DUCKDB_PATH, FEATURES_PARQUET, SQL_DIR, SQL_RESULTS_DIR

logger = logging.getLogger(__name__)


def connect(db_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    """Open (or create) the local DuckDB database file."""
    path = Path(db_path) if db_path is not None else DUCKDB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    logger.info("Connected to DuckDB at %s", path)
    return con


def load_features_table(
    con: duckdb.DuckDBPyConnection,
    parquet_path: Path | None = None,
    table_name: str = "sensor_features",
) -> None:
    """Create/replace the primary analytical table from Parquet."""
    path = Path(parquet_path) if parquet_path is not None else FEATURES_PARQUET
    if not path.is_file():
        raise FileNotFoundError(f"Parquet dataset not found: {path}")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * FROM read_parquet(?)
        """,
        [str(path)],
    )
    n = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    logger.info("Loaded table %s with %s rows from %s", table_name, f"{n:,}", path.name)


def create_analytical_views(con: duckdb.DuckDBPyConnection) -> None:
    """
    Create supporting views for SQL analytics.

    - engine_lifecycle: one row per engine with cycle span and mean RUL context
    - sensor_features_enriched: features joined to engine lifecycle attributes
    """
    con.execute(
        """
        CREATE OR REPLACE VIEW engine_lifecycle AS
        SELECT
            engine_id,
            MIN(cycle) AS first_cycle,
            MAX(cycle) AS last_cycle,
            COUNT(*) AS n_cycles,
            MAX(rul) AS max_rul,
            MIN(rul) AS min_rul
        FROM sensor_features
        GROUP BY engine_id
        """
    )
    con.execute(
        """
        CREATE OR REPLACE VIEW sensor_features_enriched AS
        SELECT
            f.*,
            e.n_cycles AS engine_total_cycles,
            e.last_cycle AS engine_last_cycle
        FROM sensor_features AS f
        INNER JOIN engine_lifecycle AS e
            ON f.engine_id = e.engine_id
        """
    )
    logger.info("Created views: engine_lifecycle, sensor_features_enriched")


def _read_sql_file(name: str) -> str:
    path = SQL_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"SQL file not found: {path}")
    return path.read_text(encoding="utf-8")


def run_sql_file(con: duckdb.DuckDBPyConnection, name: str) -> pd.DataFrame:
    """Execute a SQL file and return the result as a DataFrame."""
    sql = _read_sql_file(name)
    logger.info("Running SQL: %s", name)
    return con.execute(sql).df()


def run_all_sql_analytics(
    con: duckdb.DuckDBPyConnection,
    output_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Run packaged SQL analyses and persist CSV summaries."""
    out_dir = Path(output_dir) if output_dir is not None else SQL_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, pd.DataFrame] = {}
    for name in ("engine_summary.sql", "sensor_analysis.sql", "window_analysis.sql"):
        df = run_sql_file(con, name)
        results[name] = df
        csv_path = out_dir / name.replace(".sql", ".csv")
        df.to_csv(csv_path, index=False)
        logger.info("Wrote %s (%s rows)", csv_path.name, len(df))
    return results


def build_analytics_database(
    parquet_path: Path | None = None,
    db_path: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Load Parquet into DuckDB, create views, and run SQL analytics."""
    con = connect(db_path)
    try:
        load_features_table(con, parquet_path=parquet_path)
        create_analytical_views(con)
        return run_all_sql_analytics(con)
    finally:
        con.close()
