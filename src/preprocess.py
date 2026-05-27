"""Transaction cleanup utilities for uploaded and generated CSV files."""

from __future__ import annotations

import re

import pandas as pd

from src.categories import CATEGORY_KEYWORDS

REQUIRED_COLUMNS = ("Date", "Description", "Amount")
DESCRIPTION_NOISE = re.compile(r"\b(?:upi|neft|imps|pos|txn|pay|ref)\b|[-_/]?\d{3,}", re.I)


def clean_description(value: object) -> str:
    """Return merchant text that is useful for rules and TF-IDF features."""
    if pd.isna(value):
        return ""
    text = str(value or "").strip().lower()
    text = DESCRIPTION_NOISE.sub(" ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _canonical_columns(frame: pd.DataFrame) -> pd.DataFrame:
    lookup = {column.strip().lower(): column for column in frame.columns}
    missing = [column for column in REQUIRED_COLUMNS if column.lower() not in lookup]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    renamed = frame.rename(
        columns={lookup[column.lower()]: column for column in REQUIRED_COLUMNS}
    )
    if "category" in lookup:
        renamed = renamed.rename(columns={lookup["category"]: "Category"})
    return renamed.copy()


def _looks_like_income(description: str, category: str | None) -> bool:
    if str(category or "").strip().lower() == "income":
        return True
    return any(keyword in description for keyword in CATEGORY_KEYWORDS["Income"])


def prepare_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize uploaded transactions and drop rows that cannot be modeled."""
    transactions = _canonical_columns(frame)
    transactions["Date"] = pd.to_datetime(transactions["Date"], errors="coerce")
    raw_amounts = pd.to_numeric(transactions["Amount"], errors="coerce")
    transactions["Description"] = transactions["Description"].map(clean_description)

    category = transactions.get("Category")
    if category is None:
        category = pd.Series(index=transactions.index, dtype="object")
    transactions["Category"] = category.fillna("").astype(str).str.strip()

    valid = (
        transactions["Date"].notna()
        & raw_amounts.notna()
        & transactions["Description"].ne("")
    )
    if not valid.any():
        raise ValueError("CSV does not contain usable Date, Description, and Amount rows.")

    transactions = transactions.loc[valid].copy()
    raw_amounts = raw_amounts.loc[valid]

    transactions["Amount"] = raw_amounts.abs()
    transactions["FlowType"] = [
        "Income" if _looks_like_income(description, label) else "Expense"
        for description, label in zip(transactions["Description"], transactions["Category"])
    ]
    transactions.loc[transactions["FlowType"].eq("Income"), "Category"] = "Income"
    return transactions.sort_values("Date").reset_index(drop=True)

