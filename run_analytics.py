"""Run and display all e-commerce sales analytics from local Parquet inputs."""

from __future__ import annotations

import logging
import sys
from time import perf_counter

from pyspark.sql import DataFrame

from src.config import RAW_DATA_DIR
from src.spark_analytics import SalesAnalytics

LOGGER: logging.Logger = logging.getLogger(__name__)


def _display_timed_result(operation_name: str, result: DataFrame, rows: int = 20) -> float:
    """Display a Spark result and return its execution time in seconds."""
    started_at = perf_counter()
    print(f"\n{operation_name}")
    result.show(rows, truncate=False)
    elapsed_seconds = perf_counter() - started_at
    print(f"{operation_name} execution time: {elapsed_seconds:.2f} seconds")
    return elapsed_seconds


def main() -> int:
    """Load raw Parquet inputs, run each report, and stop Spark on exit."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    analytics = SalesAnalytics()
    try:
        analytics.create_spark_session()
        customers_df = analytics.load_parquet(RAW_DATA_DIR / "customers.parquet")
        products_df = analytics.load_parquet(RAW_DATA_DIR / "products.parquet")
        orders_df = analytics.load_parquet(RAW_DATA_DIR / "orders.parquet")
        LOGGER.info("Loaded customers dataset with columns: %s", customers_df.columns)

        _display_timed_result(
            "Top customers by revenue",
            analytics.top_customers_by_revenue(orders_df, products_df, n=10),
            rows=10,
        )
        _display_timed_result(
            "Sales by category",
            analytics.sales_by_category(orders_df, products_df),
        )
        _display_timed_result(
            "Monthly revenue trends",
            analytics.monthly_trends(orders_df, products_df),
        )
        return 0
    except Exception:
        LOGGER.exception("Analytics runner failed")
        return 1
    finally:
        analytics.stop()


if __name__ == "__main__":
    sys.exit(main())
