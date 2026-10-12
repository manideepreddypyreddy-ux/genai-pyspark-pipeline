"""Analyze generated e-commerce Parquet datasets with PySpark."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from src.config import LOG_LEVEL, PROCESSED_DATA_DIR, RAW_DATA_DIR

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER: logging.Logger = logging.getLogger(__name__)


class SalesAnalytics:
    """Run sales analytics against e-commerce orders and product DataFrames."""

    def __init__(self, spark: SparkSession | None = None) -> None:
        """Create the analytics helper, optionally using an existing Spark session."""
        self.spark: SparkSession | None = spark

    def create_spark_session(self) -> SparkSession:
        """Start local Spark with 4 GB memory, AQE, and Kryo serialization."""
        LOGGER.info("Creating local Spark session")
        spark_session = (
            SparkSession.builder.appName("EcommerceSalesAnalytics")
            .master("local[*]")
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate()
        )
        self.spark = spark_session
        return spark_session

    def load_parquet(self, path: str | Path) -> DataFrame:
        """Load a Parquet file or directory as a Spark DataFrame."""
        spark_session = self.spark or self.create_spark_session()
        parquet_path = Path(path)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet input does not exist: {parquet_path}")
        LOGGER.info("Loading Parquet data from %s", parquet_path)
        return spark_session.read.parquet(str(parquet_path))

    @staticmethod
    def _require_columns(dataframe: DataFrame, required: tuple[str, ...], label: str) -> None:
        """Raise a useful error if a DataFrame is missing required columns."""
        missing_columns = sorted(set(required) - set(dataframe.columns))
        if missing_columns:
            raise ValueError(f"{label} DataFrame is missing required columns: {missing_columns}")

    def _orders_with_products(self, orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Join orders to product prices and add a line revenue column."""
        self._require_columns(orders_df, ("product_id", "quantity"), "orders")
        self._require_columns(products_df, ("product_id", "price", "category"), "products")
        product_details = products_df.select("product_id", "price", "category")
        joined = orders_df.join(product_details, on="product_id", how="inner")
        return joined.withColumn(
            "line_revenue", F.col("quantity").cast("double") * F.col("price").cast("double")
        )

    def top_customers_by_revenue(
        self,
        orders_df: DataFrame,
        products_df: DataFrame,
        n: int = 10,
    ) -> DataFrame:
        """Return the top N customers ranked by summed order line revenue."""
        if n < 1:
            raise ValueError("n must be greater than zero")
        self._require_columns(orders_df, ("customer_id",), "orders")
        joined = self._orders_with_products(orders_df, products_df)
        return (
            joined.groupBy("customer_id")
            .agg(F.round(F.sum("line_revenue"), 2).alias("total_spend"))
            .orderBy(F.desc("total_spend"))
            .limit(n)
        )

    def sales_by_category(self, orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Return revenue and units sold, grouped by product category."""
        joined = self._orders_with_products(orders_df, products_df)
        return (
            joined.groupBy("category")
            .agg(
                F.round(F.sum("line_revenue"), 2).alias("revenue"),
                F.sum(F.col("quantity").cast("long")).alias("units_sold"),
            )
            .orderBy(F.desc("revenue"))
        )

    def monthly_trends(self, orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Return monthly revenue and growth percentage from the previous month."""
        self._require_columns(orders_df, ("order_date",), "orders")
        joined = self._orders_with_products(orders_df, products_df)
        monthly_revenue = (
            joined.withColumn(
                "month",
                F.date_trunc("month", F.to_timestamp(F.col("order_date"))),
            )
            .groupBy("month")
            .agg(F.round(F.sum("line_revenue"), 2).alias("revenue"))
        )
        month_window = Window.orderBy("month")
        with_previous_month = monthly_revenue.withColumn(
            "previous_month_revenue", F.lag("revenue").over(month_window)
        )
        growth = F.when(
            F.col("previous_month_revenue").isNull() | (F.col("previous_month_revenue") == 0),
            F.lit(None).cast("double"),
        ).otherwise(
            F.round(
                (F.col("revenue") - F.col("previous_month_revenue"))
                / F.col("previous_month_revenue")
                * F.lit(100.0),
                2,
            )
        )
        return (
            with_previous_month.withColumn("growth_pct", growth)
            .select(
                F.date_format("month", "yyyy-MM").alias("month"),
                "revenue",
                "previous_month_revenue",
                "growth_pct",
            )
            .orderBy("month")
        )

    def stop(self) -> None:
        """Stop the Spark session if this instance has started one."""
        if self.spark is not None:
            LOGGER.info("Stopping Spark session")
            self.spark.stop()
            self.spark = None


def main() -> int:
    """Load generated Parquet files, save analytics outputs, and stop Spark."""
    analytics = SalesAnalytics()
    try:
        orders_df = analytics.load_parquet(RAW_DATA_DIR / "orders.parquet")
        products_df = analytics.load_parquet(RAW_DATA_DIR / "products.parquet")
        results = {
            "top_customers_by_revenue": analytics.top_customers_by_revenue(orders_df, products_df),
            "sales_by_category": analytics.sales_by_category(orders_df, products_df),
            "monthly_trends": analytics.monthly_trends(orders_df, products_df),
        }
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        for name, result in results.items():
            output_path = PROCESSED_DATA_DIR / name
            LOGGER.info("Writing analytics result to %s", output_path)
            result.write.mode("overwrite").parquet(str(output_path))
            print(f"Wrote {name} to {output_path}")
        return 0
    except Exception:
        LOGGER.exception("Sales analytics failed")
        return 1
    finally:
        analytics.stop()


if __name__ == "__main__":
    sys.exit(main())
