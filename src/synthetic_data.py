"""Synthetic bank-style transaction generation for baseline experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.categories import EXPENSE_CATEGORIES


@dataclass(frozen=True)
class MerchantProfile:
    category: str
    descriptions: tuple[str, ...]
    low: int
    high: int
    weight: float


PROFILES = (
    MerchantProfile("Food", ("UPI-ZOMATO", "SWIGGYPAY", "Cafe Receipt", "Blinkit Grocer"), 25, 220, 0.58),
    MerchantProfile("Transport", ("Uber Ride", "Ola Cab", "Metro Card", "Fuel Station"), 20, 180, 0.22),
    MerchantProfile("Shopping", ("AMAZONINDIA-PAY", "Flipkart Order", "Myntra Store"), 50, 500, 0.06),
    MerchantProfile("Utilities", ("Electricity Bill", "Broadband Bill", "Mobile Bill", "Water Utility"), 40, 350, 0.04),
    MerchantProfile("Entertainment", ("Netflix Renewal", "Spotify Subscription", "Movie Tickets"), 49, 320, 0.04),
    MerchantProfile("Health", ("Pharmacy Purchase", "Clinic Visit", "Fitness Renewal"), 60, 700, 0.025),
    MerchantProfile("Education", ("Online Course", "Exam Books", "Tuition Payment"), 80, 800, 0.02),
    MerchantProfile("Travel", ("Rail Ticket", "Hotel Booking", "Flight Fare"), 150, 1800, 0.015),
)


def _add_recurring_rows(days: pd.DatetimeIndex, rng: np.random.Generator) -> list[dict]:
    rows: list[dict] = []
    for month_start in days.to_period("M").unique().to_timestamp():
        rows.extend(
            [
                {
                    "Date": month_start + pd.Timedelta(days=0),
                    "Description": "Salary Credit",
                    "Amount": 72000 + int(rng.normal(0, 1800)),
                    "Category": "Income",
                },
                {
                    "Date": month_start + pd.Timedelta(days=2),
                    "Description": "NEFT-HDFC-RENT",
                    "Amount": 18000,
                    "Category": "Rent",
                },
                {
                    "Date": month_start + pd.Timedelta(days=6),
                    "Description": "Broadband Bill",
                    "Amount": 899,
                    "Category": "Utilities",
                },
                {
                    "Date": month_start + pd.Timedelta(days=11),
                    "Description": "Netflix Renewal",
                    "Amount": 649,
                    "Category": "Entertainment",
                },
            ]
        )
    return rows


def generate_transactions(
    start: str = "2025-01-01",
    days: int = 365,
    target_rows: int = 3600,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate transactions with salary, recurring bills, weekends, and spikes."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=days, freq="D")
    rows = _add_recurring_rows(dates, rng)
    probabilities = np.array([profile.weight for profile in PROFILES])
    probabilities = probabilities / probabilities.sum()

    while len(rows) < target_rows:
        date = dates[int(rng.integers(0, len(dates)))]
        profile = PROFILES[int(rng.choice(len(PROFILES), p=probabilities))]
        weekend_factor = 1.25 if date.dayofweek >= 5 and profile.category in {"Food", "Shopping", "Travel"} else 1
        seasonal_factor = 1.35 if date.month in {7, 10, 11} and profile.category in {"Utilities", "Travel", "Shopping"} else 1
        amount = int(rng.integers(profile.low, profile.high + 1) * weekend_factor * seasonal_factor)
        rows.append(
            {
                "Date": date,
                "Description": f"{rng.choice(profile.descriptions)}-{rng.integers(1000, 99999)}",
                "Amount": max(amount, 1),
                "Category": profile.category,
            }
        )

    frame = pd.DataFrame(rows).sort_values(["Date", "Category"]).reset_index(drop=True)
    ordered = frame[["Date", "Description", "Amount", "Category"]]
    assert set(ordered["Category"]).issubset({*EXPENSE_CATEGORIES, "Income"})
    return ordered


def write_synthetic_csv(path: str | Path) -> Path:
    """Write the default synthetic dataset and return its resolved path."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    generate_transactions().to_csv(output, index=False)
    return output.resolve()

