"""Project paths and default settings for the e-commerce pipeline."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RAW_DATA_DIR: Path = Path(os.getenv("RAW_DATA_DIR", PROJECT_ROOT / "data" / "raw"))
PROCESSED_DATA_DIR: Path = Path(
    os.getenv("PROCESSED_DATA_DIR", PROJECT_ROOT / "data" / "processed")
)

DEFAULT_CUSTOMER_COUNT: int = 1_000
DEFAULT_PRODUCT_COUNT: int = 100
DEFAULT_ORDER_COUNT: int = 5_000
DEFAULT_RANDOM_SEED: int = 42
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
