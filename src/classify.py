"""Expense categorization through rules and a trainable NLP pipeline."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.categories import CATEGORY_KEYWORDS


def rule_category(description: str) -> str:
    """Return an interpretable fallback category for known merchant tokens."""
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in description for keyword in keywords):
            return category
    return "Other"


def build_pipeline() -> Pipeline:
    """Build the deployable TF-IDF classifier pipeline."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=800, class_weight="balanced")),
        ]
    )


def train_classifier(transactions: pd.DataFrame) -> tuple[Pipeline, Mapping[str, object]]:
    """Fit a model on labeled non-income rows and return evaluation metrics."""
    labeled = transactions[
        transactions["Category"].ne("") & transactions["Category"].ne("Income")
    ]
    if labeled["Category"].nunique() < 2:
        raise ValueError("Training requires at least two labeled expense categories.")

    stratify = labeled["Category"] if labeled["Category"].value_counts().min() >= 2 else None
    train, test = train_test_split(
        labeled,
        test_size=0.2,
        random_state=11,
        stratify=stratify,
    )
    pipeline = build_pipeline()
    pipeline.fit(train["Description"], train["Category"])
    predictions = pipeline.predict(test["Description"])
    return pipeline, {
        "accuracy": accuracy_score(test["Category"], predictions),
        "report": classification_report(
            test["Category"], predictions, output_dict=True, zero_division=0
        ),
        "test_rows": len(test),
    }


def categorize_transactions(transactions: pd.DataFrame, model: Pipeline | None = None) -> pd.DataFrame:
    """Fill expense categories with the model or merchant rules."""
    categorized = transactions.copy()
    unknown = categorized["FlowType"].eq("Expense") & categorized["Category"].eq("")
    if model is not None and unknown.any():
        categorized.loc[unknown, "Category"] = model.predict(categorized.loc[unknown, "Description"])
    if unknown.any():
        remaining = categorized["FlowType"].eq("Expense") & categorized["Category"].eq("")
        categorized.loc[remaining, "Category"] = categorized.loc[
            remaining, "Description"
        ].map(rule_category)
    return categorized

