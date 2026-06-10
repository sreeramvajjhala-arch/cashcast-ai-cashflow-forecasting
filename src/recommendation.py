"""Rule-based insights that remain understandable in a portfolio review."""

from __future__ import annotations

import pandas as pd


def spending_recommendations(transactions: pd.DataFrame) -> list[str]:
    """Compare recent category spending with prior monthly behavior."""
    expenses = transactions[transactions["FlowType"].eq("Expense")].copy()
    if expenses.empty:
        return ["Upload expense transactions to generate recommendations."]

    expenses["Month"] = expenses["Date"].dt.to_period("M")
    monthly = expenses.groupby(["Month", "Category"])["Amount"].sum().unstack(fill_value=0)
    if len(monthly) < 2:
        return ["Collect another month of history before comparing spending behavior."]

    latest = monthly.iloc[-1]
    baseline = monthly.iloc[:-1].mean().replace(0, pd.NA)
    change = ((latest - baseline) / baseline).dropna().sort_values(ascending=False)
    recommendations = []
    for category, ratio in change.head(3).items():
        if ratio >= 0.2:
            recommendations.append(
                f"{category} spending is {ratio:.0%} above its earlier monthly average."
            )
    return recommendations or ["Recent category spending is close to its earlier monthly average."]

