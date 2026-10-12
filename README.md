# E-commerce Data Pipeline

Generate fictional customer, product, and order data with Python, then analyze sales using PySpark.

## Project structure

```text
genai-pyspark-pipeline/
|-- data/
|   |-- raw/                 # Generated Parquet inputs
|   `-- processed/           # Spark analytics outputs
|-- notebooks/               # Jupyter notebook workspace
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- data_generator.py
|   `-- spark_analytics.py
|-- tests/
|   `-- test_data_generator.py
|-- main.py
|-- requirements.txt
`-- README.md
```

Generated data and Spark output are ignored by Git. `.gitkeep` files retain the empty data directories.

## Requirements

- Python 3.10 or newer
- Java 17 or newer, with `JAVA_HOME` configured for PySpark

Install dependencies in a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the pipeline

Generate the default 100,000 customers, 10,000 products, and 1,000,000 orders, then write them to `data/raw/` as Parquet files:

```bash
python main.py
```

Run PySpark analytics on those Parquet files:

```bash
python run_analytics.py
```

The analytics command writes these Parquet output directories under `data/processed/`:

- `top_customers_by_revenue/`
- `sales_by_category/`
- `monthly_trends/`

Each output contains the Spark-generated Parquet part files. The monthly trend output contains a null growth percentage for the first available month, because there is no prior month to compare.

## Use the analytics class directly

```python
from src.spark_analytics import SalesAnalytics

analytics = SalesAnalytics()
try:
    orders = analytics.load_parquet("data/raw/orders.parquet")
    products = analytics.load_parquet("data/raw/products.parquet")
    top_customers = analytics.top_customers_by_revenue(orders, products, n=10)
    category_sales = analytics.sales_by_category(orders, products)
    monthly = analytics.monthly_trends(orders, products)
finally:
    analytics.stop()
```

`SalesAnalytics` starts Spark in local mode with 4 GB memory, adaptive query execution, and Kryo serialization. [Spark configuration](https://spark.apache.org/docs/latest/configuration) [Spark tuning](https://spark.apache.org/docs/latest/tuning.html)

## Generate pandas DataFrames directly

```python
from src.data_generator import SyntheticDataGenerator

customers, products, orders = SyntheticDataGenerator().generate_all()
```

Customer ages use a normal distribution centered on 35 with standard deviation 12, clipped to ages 18 through 100. The top 20% customer group receives 80% of order assignments (rounded up to whole orders), with Pareto weights distributing those orders within that group.

## Tests

```bash
python -m unittest discover -s tests -v
```
