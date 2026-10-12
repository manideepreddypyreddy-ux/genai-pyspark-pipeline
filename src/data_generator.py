"""Generate reproducible synthetic e-commerce data as CSV files."""

from __future__ import annotations

import csv
import logging
import math
import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from faker import Faker
from numpy.random import Generator
from tqdm.auto import tqdm

from src.config import (
    DEFAULT_CUSTOMER_COUNT,
    DEFAULT_ORDER_COUNT,
    DEFAULT_PRODUCT_COUNT,
    DEFAULT_RANDOM_SEED,
    LOG_LEVEL,
    RAW_DATA_DIR,
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER: logging.Logger = logging.getLogger(__name__)

PRODUCT_CATEGORIES: tuple[str, ...] = (
    "Books",
    "Clothing",
    "Electronics",
    "Home & Kitchen",
    "Sports",
    "Toys",
)
ORDER_STATUSES: tuple[str, ...] = ("completed", "completed", "completed", "cancelled", "refunded")


class SyntheticDataGenerator:
    """Generate synthetic e-commerce datasets as pandas DataFrames.

    The default customer age distribution is normal with mean 35 and standard
    deviation 12, clipped to ages 18 through 100. The highest-volume 20% of
    customers receive 80% of assignments (rounded up to whole orders); Pareto
    weights make order counts within that segment uneven.
    """

    CATEGORIES: tuple[str, ...] = ("Electronics", "Clothing", "Home", "Sports", "Books")

    def __init__(
        self,
        customer_count: int = 100_000,
        product_count: int = 10_000,
        order_count: int = 1_000_000,
        seed: int = DEFAULT_RANDOM_SEED,
        mean_age: float = 35.0,
        age_std_dev: float = 12.0,
        pareto_shape: float = 1.5,
        progress: bool = True,
    ) -> None:
        """Set dataset sizes, random seed, distributions, and progress display."""
        if min(customer_count, product_count, order_count) < 0:
            raise ValueError("dataset counts must be zero or greater")
        if mean_age <= 0 or age_std_dev <= 0:
            raise ValueError("mean_age and age_std_dev must be greater than zero")
        if pareto_shape <= 0:
            raise ValueError("pareto_shape must be greater than zero")

        self.customer_count = customer_count
        self.product_count = product_count
        self.order_count = order_count
        self.mean_age = mean_age
        self.age_std_dev = age_std_dev
        self.pareto_shape = pareto_shape
        self.progress = progress
        self._rng: Generator = np.random.default_rng(seed)
        self._faker: Faker = Faker()
        self._faker.seed_instance(seed)
        LOGGER.info(
            "Initialized synthetic generator (customers=%d, products=%d, orders=%d, seed=%d)",
            customer_count,
            product_count,
            order_count,
            seed,
        )

    def generate_customers(self, count: int | None = None) -> pd.DataFrame:
        """Generate customers with Faker profiles and normally distributed ages."""
        row_count = self.customer_count if count is None else count
        if row_count < 0:
            raise ValueError("count must be zero or greater")
        ages = np.clip(
            np.rint(self._rng.normal(self.mean_age, self.age_std_dev, row_count)),
            18,
            100,
        ).astype(np.int16)
        rows: list[dict[str, Any]] = []
        for index in tqdm(
            range(row_count),
            desc="Generating customers",
            unit="customer",
            disable=not self.progress,
        ):
            rows.append(
                {
                    "customer_id": f"CUST-{index + 1:07d}",
                    "name": self._faker.name(),
                    "email": self._faker.unique.email(),
                    "age": int(ages[index]),
                    "city": self._faker.city(),
                    "country": self._faker.country(),
                    "registration_date": self._faker.date_between(
                        start_date="-5y", end_date="today"
                    ).isoformat(),
                }
            )
        dataframe = pd.DataFrame(
            rows,
            columns=[
                "customer_id",
                "name",
                "email",
                "age",
                "city",
                "country",
                "registration_date",
            ],
        )
        LOGGER.info("Generated customer DataFrame with %d rows", len(dataframe))
        return dataframe

    def generate_products(self, count: int | None = None) -> pd.DataFrame:
        """Generate products with categories, prices, stock, and ratings."""
        row_count = self.product_count if count is None else count
        if row_count < 0:
            raise ValueError("count must be zero or greater")
        rows: list[dict[str, Any]] = []
        for index in tqdm(
            range(row_count),
            desc="Generating products",
            unit="product",
            disable=not self.progress,
        ):
            rows.append(
                {
                    "product_id": f"PROD-{index + 1:06d}",
                    "name": self._faker.catch_phrase(),
                    "category": str(self._rng.choice(self.CATEGORIES)),
                    "price": round(float(self._rng.uniform(10.0, 500.0)), 2),
                    "stock": int(self._rng.integers(0, 1_001)),
                    "rating": round(float(self._rng.uniform(1.0, 5.0)), 1),
                }
            )
        dataframe = pd.DataFrame(
            rows,
            columns=["product_id", "name", "category", "price", "stock", "rating"],
        )
        LOGGER.info("Generated product DataFrame with %d rows", len(dataframe))
        return dataframe

    def _customer_order_indices(self, customer_count: int, order_count: int) -> np.ndarray:
        """Assign 80% of orders, rounded up, to a Pareto-weighted top customer group."""
        if order_count == 0:
            return np.empty(0, dtype=np.int64)
        if customer_count == 0:
            raise ValueError("at least one customer is required to generate orders")

        high_volume_count = max(1, math.ceil(customer_count * 0.20))
        top_order_count = math.ceil(order_count * 0.80)
        other_order_count = order_count - top_order_count

        pareto_weights = self._rng.pareto(self.pareto_shape, high_volume_count) + 1.0
        top_probabilities = pareto_weights / pareto_weights.sum()
        top_assignments = self._rng.choice(
            high_volume_count,
            size=top_order_count,
            replace=True,
            p=top_probabilities,
        )
        if other_order_count:
            other_assignments = self._rng.integers(
                high_volume_count,
                customer_count,
                size=other_order_count,
            ) if high_volume_count < customer_count else self._rng.choice(
                high_volume_count,
                size=other_order_count,
                replace=True,
            )
            assignments = np.concatenate((top_assignments, other_assignments))
        else:
            assignments = top_assignments
        self._rng.shuffle(assignments)
        return assignments

    def generate_orders(
        self,
        customers: pd.DataFrame,
        products: pd.DataFrame,
        count: int | None = None,
        chunk_size: int = 50_000,
    ) -> pd.DataFrame:
        """Generate orders referencing the provided customers and products."""
        row_count = self.order_count if count is None else count
        if row_count < 0:
            raise ValueError("count must be zero or greater")
        if chunk_size < 1:
            raise ValueError("chunk_size must be greater than zero")
        if row_count and (customers.empty or products.empty):
            raise ValueError("customers and products must be non-empty when count is positive")
        if not {"customer_id"}.issubset(customers.columns):
            raise ValueError("customers DataFrame must include a customer_id column")
        if not {"product_id"}.issubset(products.columns):
            raise ValueError("products DataFrame must include a product_id column")

        customer_indices = self._customer_order_indices(len(customers), row_count)
        product_indices = self._rng.integers(0, len(products), size=row_count) if row_count else np.empty(0, dtype=np.int64)
        customer_ids = customers["customer_id"].to_numpy()
        product_ids = products["product_id"].to_numpy()
        now = datetime.now(timezone.utc).replace(microsecond=0)
        start_timestamp = int((now - timedelta(days=730)).timestamp())
        end_timestamp = int(now.timestamp())
        rows: list[pd.DataFrame] = []
        chunk_starts = range(0, row_count, chunk_size)
        for start in tqdm(
            chunk_starts,
            total=math.ceil(row_count / chunk_size),
            desc="Generating orders",
            unit="chunk",
            disable=not self.progress,
        ):
            stop = min(start + chunk_size, row_count)
            chunk_count = stop - start
            timestamps = self._rng.integers(start_timestamp, end_timestamp + 1, size=chunk_count)
            order_dates = pd.to_datetime(timestamps, unit="s", utc=True)
            rows.append(
                pd.DataFrame(
                    {
                        "order_id": [f"ORD-{index + 1:09d}" for index in range(start, stop)],
                        "customer_id": customer_ids[customer_indices[start:stop]],
                        "product_id": product_ids[product_indices[start:stop]],
                        "quantity": self._rng.integers(1, 11, size=chunk_count, dtype=np.int8),
                        "order_date": order_dates,
                    }
                )
            )
        if rows:
            dataframe = pd.concat(rows, ignore_index=True)
        else:
            dataframe = pd.DataFrame(
                columns=["order_id", "customer_id", "product_id", "quantity", "order_date"]
            )
        LOGGER.info("Generated order DataFrame with %d rows", len(dataframe))
        return dataframe

    def generate_all(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate and return customer, product, and order DataFrames."""
        customers = self.generate_customers()
        products = self.generate_products()
        orders = self.generate_orders(customers, products)
        return customers, products, orders


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Write rows to a UTF-8 CSV file, creating its parent directory first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Wrote %d rows to %s", len(rows), path)


def generate_customers(
    count: int = DEFAULT_CUSTOMER_COUNT,
    seed: int = DEFAULT_RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Create fake customers with stable IDs and basic profile fields."""
    if count < 0:
        raise ValueError("count must be zero or greater")
    fake = Faker()
    fake.seed_instance(seed)
    rng = random.Random(seed)
    today = date.today()
    customers: list[dict[str, Any]] = []
    for customer_number in range(1, count + 1):
        joined_on = today - timedelta(days=rng.randint(30, 1_800))
        customers.append(
            {
                "customer_id": f"CUST-{customer_number:06d}",
                "first_name": fake.first_name(),
                "last_name": fake.last_name(),
                "email": fake.unique.email(),
                "city": fake.city(),
                "country": fake.country_code(),
                "signup_date": joined_on.isoformat(),
            }
        )
    LOGGER.info("Generated %d customers", len(customers))
    return customers


def generate_products(
    count: int = DEFAULT_PRODUCT_COUNT,
    seed: int = DEFAULT_RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Create fake product catalog records with categories and prices."""
    if count < 0:
        raise ValueError("count must be zero or greater")
    fake = Faker()
    fake.seed_instance(seed)
    rng = random.Random(seed)
    products: list[dict[str, Any]] = []
    for product_number in range(1, count + 1):
        products.append(
            {
                "product_id": f"PROD-{product_number:05d}",
                "product_name": fake.catch_phrase(),
                "category": rng.choice(PRODUCT_CATEGORIES),
                "unit_price": round(rng.uniform(5.0, 500.0), 2),
            }
        )
    LOGGER.info("Generated %d products", len(products))
    return products


def generate_orders(
    customers: list[dict[str, Any]],
    products: list[dict[str, Any]],
    count: int = DEFAULT_ORDER_COUNT,
    seed: int = DEFAULT_RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Create orders that reference supplied customers and products."""
    if count < 0:
        raise ValueError("count must be zero or greater")
    if count and (not customers or not products):
        raise ValueError("customers and products must be non-empty when count is positive")
    rng = random.Random(seed)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    orders: list[dict[str, Any]] = []
    for order_number in range(1, count + 1):
        product = rng.choice(products)
        quantity = rng.randint(1, 5)
        discount = rng.choice((0.0, 0.0, 0.05, 0.10, 0.15))
        unit_price = float(product["unit_price"])
        total = round(quantity * unit_price * (1.0 - discount), 2)
        ordered_at = now - timedelta(days=rng.randint(0, 730), minutes=rng.randint(0, 1_439))
        orders.append(
            {
                "order_id": f"ORD-{order_number:07d}",
                "customer_id": rng.choice(customers)["customer_id"],
                "product_id": product["product_id"],
                "order_date": ordered_at.isoformat(),
                "quantity": quantity,
                "discount_pct": round(discount * 100, 2),
                "order_total": total,
                "status": rng.choice(ORDER_STATUSES),
            }
        )
    LOGGER.info("Generated %d orders", len(orders))
    return orders


def generate_dataset(
    raw_data_dir: Path = RAW_DATA_DIR,
    customer_count: int = DEFAULT_CUSTOMER_COUNT,
    product_count: int = DEFAULT_PRODUCT_COUNT,
    order_count: int = DEFAULT_ORDER_COUNT,
    seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[Path, Path, Path]:
    """Generate all three related datasets and return their file paths."""
    customers = generate_customers(customer_count, seed)
    products = generate_products(product_count, seed + 1)
    orders = generate_orders(customers, products, order_count, seed + 2)

    customer_path = raw_data_dir / "customers.csv"
    product_path = raw_data_dir / "products.csv"
    order_path = raw_data_dir / "orders.csv"
    _write_csv(
        customer_path,
        ["customer_id", "first_name", "last_name", "email", "city", "country", "signup_date"],
        customers,
    )
    _write_csv(product_path, ["product_id", "product_name", "category", "unit_price"], products)
    _write_csv(
        order_path,
        ["order_id", "customer_id", "product_id", "order_date", "quantity", "discount_pct", "order_total", "status"],
        orders,
    )
    return customer_path, product_path, order_path


def main() -> None:
    """Generate the default dataset in the configured raw data directory."""
    generate_dataset()


if __name__ == "__main__":
    main()
