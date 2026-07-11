#!/usr/bin/env python3
"""
This module visualizes churn rates per category.
"""
import matplotlib.pyplot as plt


def plot_categorical_vs_churn(df, col):
    """
    Plots churn rate for each category
    in a selected column.

    Args:
        df: The pandas DataFrame
    containing a Churn column.
        col: The categorical column
    to compare with Churn.

    Returns:
        None
    """
    plt.figure(figsize=(12, 8))

    churn_rate = df.groupby(col)["Churn"].apply(
        lambda x: (x == "Yes").mean()
        )
    plt.bar(
        churn_rate.index,
        churn_rate.values,
        width=0.8
        )
    plt.title(f"Churn Rate by {col}")
    plt.ylabel("Churn Rate")
    plt.xticks(rotation=45)
    plt.show()

    return None
