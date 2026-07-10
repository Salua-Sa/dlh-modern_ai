#!/usr/bin/env python3
"""
This module visualizes missing values in a DataFrame
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_missingness(df):
    """
    Visualizes missing values in a DataFrame
usingn a scatter plot.

    Args:
       df: The pandas DataFrame to analyze

    Returns:
       None
    """
    # find missing values positions
    plt.figure(figsize=(12, 8))
    missing_rows, missing_cols = np.where(df.isna())

    # plot missing values
    plt.scatter(missing_rows, missing_cols, marker="|")

    # map y-axis numbers to column names
    plt.yticks(
        ticks=np.arange(
            len(df.columns)), labels=df.columns
        )

    plt.title("Missingness Plot")

    plt.tight_layout()
    plt.show()

    return None
