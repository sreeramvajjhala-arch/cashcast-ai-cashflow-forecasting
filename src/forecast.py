"""Category spending forecasts and cash flow projections."""

from __future__ import annotations

import logging

import pandas as pd

from src.feature_engineering import daily_category_spending

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


def forecast_average_spending(
    transactions: pd.DataFrame,
    periods: int = 30,
    max_categories: int | None = None,
    lookback_days: int = 60,
) -> pd.DataFrame:
    """Forecast spending from recent daily category averages.

    This is intentionally simple and fast, making it useful as an app baseline
    and as a comparison point for Prophet experiments.
    """
    daily = daily_category_spending(transactions)
    if daily.empty:
        return pd.DataFrame(columns=["Date", "Category", "yhat", "yhat_lower", "yhat_upper"])

    last_date = daily["Date"].max()
    recent = daily[daily["Date"] > last_date - pd.Timedelta(days=lookback_days)]
    if max_categories is not None:
        leading = (
            recent.groupby("Category")["Amount"]
            .sum()
            .sort_values(ascending=False)
            .head(max_categories)
            .index
        )
        recent = recent[recent["Category"].isin(leading)]

    baselines = recent.groupby("Category", as_index=False)["Amount"].mean()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=periods, freq="D")
    forecasts: list[pd.DataFrame] = []
    for row in baselines.itertuples(index=False):
        values = pd.DataFrame(
            {
                "Date": future_dates,
                "Category": row.Category,
                "yhat": float(row.Amount),
            }
        )
        weekend_multiplier = 1.15 if str(row.Category) in {"Food", "Shopping", "Travel"} else 1.0
        values.loc[values["Date"].dt.dayofweek >= 5, "yhat"] *= weekend_multiplier
        values["yhat_lower"] = values["yhat"] * 0.85
        values["yhat_upper"] = values["yhat"] * 1.15
        forecasts.append(values)

    if not forecasts:
        return pd.DataFrame(columns=["Date", "Category", "yhat", "yhat_lower", "yhat_upper"])
    return pd.concat(forecasts, ignore_index=True)


def forecast_category_spending(
    transactions: pd.DataFrame,
    periods: int = 30,
    max_categories: int | None = None,
) -> pd.DataFrame:
    """Forecast daily expenses per category with separate Prophet models."""
    daily = daily_category_spending(transactions)
    if max_categories is not None and not daily.empty:
        leading = (
            daily.groupby("Category")["Amount"]
            .sum()
            .sort_values(ascending=False)
            .head(max_categories)
            .index
        )
        daily = daily[daily["Category"].isin(leading)]

    forecasts: list[pd.DataFrame] = []
    for category, category_daily in daily.groupby("Category"):
        if len(category_daily) < 14 or category in {"Income", "Other"}:
            continue
        from prophet import Prophet

        history = category_daily.rename(columns={"Date": "ds", "Amount": "y"})[["ds", "y"]]
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=len(history) >= 300,
            interval_width=0.8,
        )
        model.fit(history)
        future = model.make_future_dataframe(periods=periods, freq="D")
        predicted = model.predict(future).tail(periods)
        predicted["Category"] = category
        forecasts.append(
            predicted[["ds", "Category", "yhat", "yhat_lower", "yhat_upper"]]
            .rename(columns={"ds": "Date"})
            .assign(
                yhat=lambda frame: frame["yhat"].clip(lower=0),
                yhat_lower=lambda frame: frame["yhat_lower"].clip(lower=0),
                yhat_upper=lambda frame: frame["yhat_upper"].clip(lower=0),
            )
        )
    if not forecasts:
        return pd.DataFrame(columns=["Date", "Category", "yhat", "yhat_lower", "yhat_upper"])
    return pd.concat(forecasts, ignore_index=True)


def project_balance(transactions: pd.DataFrame, forecast: pd.DataFrame) -> dict[str, float]:
    """Project balance from observed income minus expenses and expected expenses."""
    if transactions.empty:
        return {
            "expected_income": 0.0,
            "expected_expenses": 0.0,
            "projected_balance": 0.0,
            "observed_balance": 0.0,
        }

    observed_income = transactions.loc[transactions["FlowType"].eq("Income"), "Amount"].sum()
    observed_expenses = transactions.loc[transactions["FlowType"].eq("Expense"), "Amount"].sum()
    elapsed_months = max(
        (transactions["Date"].max().to_period("M") - transactions["Date"].min().to_period("M")).n + 1,
        1,
    )
    expected_income = observed_income / elapsed_months
    expected_expenses = float(forecast["yhat"].sum()) if not forecast.empty else 0.0
    balance = observed_income - observed_expenses + expected_income - expected_expenses
    return {
        "expected_income": float(expected_income),
        "expected_expenses": expected_expenses,
        "projected_balance": float(balance),
        "observed_balance": float(observed_income - observed_expenses),
    }
