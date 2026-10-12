"""Unit tests for synthetic data generation."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.data_generator import generate_customers, generate_dataset, generate_orders, generate_products


class DataGeneratorTests(unittest.TestCase):
    """Check generated records and their CSV serialization."""

    def test_generation_is_reproducible(self) -> None:
        """A fixed seed should produce identical customers and products."""
        self.assertEqual(generate_customers(4, 7), generate_customers(4, 7))
        self.assertEqual(generate_products(4, 7), generate_products(4, 7))

    def test_orders_reference_existing_entities(self) -> None:
        """Each generated order should reference a supplied customer and product."""
        customers = generate_customers(3, 11)
        products = generate_products(2, 12)
        orders = generate_orders(customers, products, 10, 13)
        customer_ids = {customer["customer_id"] for customer in customers}
        product_ids = {product["product_id"] for product in products}
        self.assertTrue(all(order["customer_id"] in customer_ids for order in orders))
        self.assertTrue(all(order["product_id"] in product_ids for order in orders))
        self.assertTrue(all(order["order_total"] > 0 for order in orders))

    def test_generate_dataset_writes_csv_headers_and_rows(self) -> None:
        """The dataset helper should create all CSV files with requested rows."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = generate_dataset(Path(temporary_directory), 2, 3, 5, 19)
            self.assertEqual(len(paths), 3)
            expected_counts = {"customers.csv": 2, "products.csv": 3, "orders.csv": 5}
            for path in paths:
                with path.open(newline="", encoding="utf-8") as csv_file:
                    records = list(csv.DictReader(csv_file))
                self.assertTrue(path.is_file())
                self.assertEqual(len(records), expected_counts[path.name])


if __name__ == "__main__":
    unittest.main()
