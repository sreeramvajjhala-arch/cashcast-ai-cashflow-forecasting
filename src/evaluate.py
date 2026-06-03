"""Small evaluation helpers for classification and forecasting experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd


def forecast_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Return common regression metrics for aligned forecast observations."""
    aligned = pd.concat([actual.rename("actual"), predicted.rename("predicted")], axis=1).dropna()
    if aligned.empty:
        raise ValueError("Forecast metrics require aligned actual and predicted values.")

    errors = aligned["actual"] - aligned["predicted"]
    nonzero_actual = aligned["actual"].replace(0, np.nan)
    mape = (errors.abs() / nonzero_actual).dropna().mean()
    return {
        "mae": float(errors.abs().mean()),
        "rmse": float(np.sqrt((errors**2).mean())),
        "mape": float(mape) if pd.notna(mape) else float("nan"),
    }

