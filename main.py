"""Generate synthetic e-commerce data and save it as Parquet files."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd

import src.config as config
from src.data_generator import SyntheticDataGenerator

LOGGER: logging.Logger = logging.getLogger(__name__)
DATASET_NAMES: tuple[str, ...] = ("customers", "products", "orders")


def _format_file_size(size_in_bytes: int) -> str:
    """Format a byte count as a human-readable binary file size."""
    size = float(size_in_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def save_dataframes(
    dataframes: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    output_dir: Path,
) -> dict[str, Path]:
    """Save generated pandas DataFrames as named Parquet files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files: dict[str, Path] = {}
    for name, dataframe in zip(DATASET_NAMES, dataframes, strict=True):
        output_path = output_dir / f"{name}.parquet"
        dataframe.to_parquet(output_path, engine="pyarrow", index=False)
        saved_files[name] = output_path
        LOGGER.info("Saved %s dataset to %s", name, output_path)
    return saved_files


def main() -> int:
    """Generate all datasets, save Parquet files, and report timing and sizes."""
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    total_start = perf_counter()
    try:
        generator = SyntheticDataGenerator()
        generation_start = perf_counter()
        customers, products, orders = generator.generate_all()
        generation_seconds = perf_counter() - generation_start

        saved_files = save_dataframes((customers, products, orders), config.RAW_DATA_DIR)
        total_seconds = perf_counter() - total_start

        print(f"Data generation time: {generation_seconds:.2f} seconds")
        print("Parquet file sizes:")
        for name, path in saved_files.items():
            print(f"  {name}: {_format_file_size(path.stat().st_size)} ({path})")
        print(f"Total elapsed time: {total_seconds:.2f} seconds")
        return 0
    except Exception:
        LOGGER.exception("Synthetic data generation or Parquet output failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
