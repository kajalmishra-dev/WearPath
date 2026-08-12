#!/usr/bin/env python3
"""
Download and extract the NASA C-MAPSS turbofan dataset into data/raw/.

Source
------
NASA Prognostics Center of Excellence — Turbofan Engine Degradation Simulation
(C-MAPSS). The archive typically contains train/test/RUL files for FD001–FD004.

Redistribution of the raw files may be restricted by NASA terms; this project
keeps ``data/raw/*`` gitignored. Another developer must run this script (or
manually place the files) before executing the pipeline.

Usage
-----
    python scripts/download_cmapss.py
    python scripts/download_cmapss.py --url <zip-url>
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import zipfile
from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

# Documented candidate locations. NASA endpoints change over time; the script
# tries each URL and also accepts a manual --url override.
CANDIDATE_URLS: tuple[str, ...] = (
    # Commonly cited PHM / NASA turbofan archive mirror (ZIP)
    "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip",
    # NASA PCoE short link historically used for C-MAPSS
    "https://ti.arc.nasa.gov/c/6/",
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("download_cmapss")


def _download_bytes(url: str, timeout: int = 120) -> bytes:
    request = Request(
        url,
        headers={"User-Agent": "WearPath/0.1 (data-engineering-portfolio)"},
    )
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def _extract_zip(payload: bytes, dest: Path) -> list[Path]:
    """
    Extract a ZIP payload into dest.

    Some NASA mirrors ship an outer archive that contains ``CMAPSSData.zip``.
    Nested ZIP members are extracted recursively until text files appear.
    """
    dest.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            member_bytes = zf.read(info)
            if name.lower().endswith(".zip") or member_bytes[:2] == b"PK":
                logger.info("Extracting nested archive: %s", name)
                extracted.extend(_extract_zip(member_bytes, dest))
                continue
            if name.lower().endswith(".txt") or "readme" in name.lower():
                target = dest / "CMAPSSData" / name
            else:
                target = dest / "CMAPSSData" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(member_bytes)
            extracted.append(target)
    return extracted


def has_fd001(raw_dir: Path = RAW_DIR) -> bool:
    """Return True if train_FD001.txt is already present."""
    patterns = (
        "train_FD001.txt",
        "*/train_FD001.txt",
    )
    for pattern in patterns:
        if list(raw_dir.glob(pattern)):
            return True
    return any(raw_dir.rglob("train_FD001.txt"))


def download_cmapss(url: str | None = None) -> Path:
    """Download C-MAPSS into data/raw and return the directory used."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if has_fd001(RAW_DIR):
        logger.info("C-MAPSS FD001 already present under %s — skipping download", RAW_DIR)
        return RAW_DIR

    urls = (url,) if url else CANDIDATE_URLS
    last_error: Exception | None = None
    for candidate in urls:
        if not candidate:
            continue
        logger.info("Trying download: %s", candidate)
        try:
            payload = _download_bytes(candidate)
            if payload[:2] == b"PK":
                files = _extract_zip(payload, RAW_DIR)
                logger.info("Extracted %s files into %s", len(files), RAW_DIR)
                if has_fd001(RAW_DIR):
                    return RAW_DIR
                logger.warning("Zip downloaded but train_FD001.txt not found; trying next URL")
            else:
                logger.warning("Response was not a ZIP archive (size=%s bytes)", len(payload))
        except (HTTPError, URLError, zipfile.BadZipFile, TimeoutError, OSError) as exc:
            last_error = exc
            logger.warning("Download failed for %s: %s", candidate, exc)

    manual = RAW_DIR / "MANUAL_DOWNLOAD.md"
    manual.write_text(
        """# Manual C-MAPSS download

Automatic download did not succeed. Obtain the NASA C-MAPSS archive manually:

1. Visit the NASA Prognostics Center of Excellence data repository:
   https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository
   or search for "C-MAPSS Turbofan Engine Degradation Simulation Data Set".
2. Download the C-MAPSS / Turbofan zip archive.
3. Extract so that this path exists:

   `data/raw/CMAPSSData/train_FD001.txt`

   (Also place `test_FD001.txt` and `RUL_FD001.txt` alongside it.)

4. Re-run: `python scripts/run_pipeline.py`
""",
        encoding="utf-8",
    )
    message = (
        "Could not automatically download C-MAPSS. "
        f"See {manual} for manual setup instructions."
    )
    if last_error is not None:
        message += f" Last error: {last_error}"
    raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download NASA C-MAPSS into data/raw/")
    parser.add_argument("--url", default=None, help="Optional direct ZIP URL override")
    args = parser.parse_args()
    try:
        path = download_cmapss(url=args.url)
    except Exception as exc:
        logger.error("%s", exc)
        return 1
    logger.info("Raw data ready at %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
