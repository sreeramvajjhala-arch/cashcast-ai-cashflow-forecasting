import pandas as pd

from src.evaluate import forecast_metrics
from src.forecast import forecast_average_spending


def test_forecast_metrics_report_common_errors():
    metrics = forecast_metrics(
        pd.Series([100.0, 200.0, 300.0]),
        pd.Series([90.0, 210.0, 270.0]),
    )

    assert metrics["mae"] == 50 / 3
    assert round(metrics["rmse"], 6) == round((1100 / 3) ** 0.5, 6)
    assert round(metrics["mape"], 6) == round((0.1 + 0.05 + 0.1) / 3, 6)


def test_average_forecast_is_fast_and_nonnegative():
    transactions = pd.DataFrame(
        {
            "Date": pd.date_range("2025-01-01", periods=20, freq="D"),
            "Category": ["Food"] * 20,
            "Amount": [100.0] * 20,
            "FlowType": ["Expense"] * 20,
        }
    )

    forecast = forecast_average_spending(transactions, periods=7)

    assert len(forecast) == 7
    assert forecast["yhat"].min() >= 0
