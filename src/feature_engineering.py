"""Feature tables derived from normalized transactions."""

from __future__ import annotations

import pandas as pd


def daily_category_spending(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return a dense daily expense series for each category."""
    expenses = transactions[transactions["FlowType"].eq("Expense")].copy()
    if expenses.empty:
        return pd.DataFrame(columns=["Date", "Category", "Amount"])

    daily = expenses.groupby(["Date", "Category"], as_index=False)["Amount"].sum()
    categories = sorted(daily["Category"].dropna().unique())
    dates = pd.date_range(daily["Date"].min(), daily["Date"].max(), freq="D")
    index = pd.MultiIndex.from_product([dates, categories], names=["Date", "Category"])
    return (
        daily.set_index(["Date", "Category"])
        .reindex(index, fill_value=0)
        .reset_index()
        .sort_values(["Category", "Date"])
    )


def date_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Add transparent date signals that can be explored in notebooks."""
    featured = transactions.copy()
    featured["DayOfWeek"] = featured["Date"].dt.day_name()
    featured["IsWeekend"] = featured["Date"].dt.dayofweek >= 5
    featured["Month"] = featured["Date"].dt.month
    featured["MonthDay"] = featured["Date"].dt.day
    return featured

