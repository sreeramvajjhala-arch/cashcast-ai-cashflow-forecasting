"""Explainable cash flow deficit alerts."""

from __future__ import annotations

import pandas as pd


def assess_cash_flow_risk(projection: dict[str, float], forecast: pd.DataFrame) -> dict[str, object]:
    """Return a bounded risk score and forecast-based deficit message."""
    expected_expenses = max(projection["expected_expenses"], 1.0)
    deficit = max(-projection["projected_balance"], 0.0)
    expense_pressure = min(expected_expenses / max(projection["expected_income"], 1.0), 2.0)
    score = round(min(100, deficit / expected_expenses * 70 + max(expense_pressure - 1, 0) * 30))
    level = "High Risk" if score > 60 else "Moderate" if score > 30 else "Safe"
    deficit_date = None

    if deficit and not forecast.empty:
        by_day = forecast.groupby("Date", as_index=False)["yhat"].sum().sort_values("Date")
        buffer = projection["observed_balance"] + projection["expected_income"]
        crossed = by_day[by_day["yhat"].cumsum() > buffer]
        if not crossed.empty:
            deficit_date = pd.Timestamp(crossed.iloc[0]["Date"]).date().isoformat()

    return {
        "score": score,
        "level": level,
        "deficit": float(deficit),
        "deficit_date": deficit_date,
    }

