import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def mock_transaction_df():
    np.random.seed(42)
    n_rows = 150
    start = datetime(2026, 1, 1)

    dates = [start + timedelta(hours=i) for i in range(n_rows)]
    customers = [f"C_{i % 5}" for i in range(n_rows)]
    amounts = [100.0] * n_rows
    # Insert clear outliers
    amounts[10] = 5000.0
    amounts[20] = 7500.0

    return pd.DataFrame({
        "transaction_id": [f"T_{i}" for i in range(n_rows)],
        "timestamp": dates,
        "customer_id": customers,
        "amount": amounts,
        "location": ["TestCity"] * n_rows
    })