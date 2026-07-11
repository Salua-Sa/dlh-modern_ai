#!/usr/bin/env python3
"""
This module visualizes the distributions of continuous numerical features.
"""
from xml.etree.ElementInclude import include

from matplotlib import axes
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_continuous_distributions(df, columns_to_plot=None):
    """
    Plots histograms, KDE lines, and boxplots for numeric columns.

    Args:
        df: The pandas DataFrame to analyze.
        columns_to_plot: Optional lists of numeric colummns to plot

    Returns:
        None
    """
    if columns_to_plot is None:
        columns_to_plot = df.select_dtypes(include="number").columns

    n_cols = len(columns_to_plot)
    fig, axes = plt.subplots(n_cols, 2, figsize=(10, 3*n_cols))

    for i, column in enumerate(columns_to_plot):
        data = df[column].dropna()
        axes[i][0].hist(
            data,
            bins=30,
            density=True,
            alpha=0.7,
            edgecolor='black'
            )
        kde = stats.gaussian_kde(data)
        x_values = np.linspace(data.min(), data.max(), 1000)
        axes[i][0].plot(
            x_values,
            kde(x_values),
            color="red",
            linestyle="--")
        axes[i][0].set_title(f"{column} Histogram + KDE")

        if column == "TotalCharges":
            axes[i][0].set_ylim(0, 0.0008)
            axes[i][0].set_yticks([0.0000, 0.0002, 0.0004, 0.0006])
        axes[i][1].boxplot(data, vert=False)
        axes[i][1].set_title(f"{column} Boxplot")
    plt.tight_layout()
    plt.savefig("Task_8.png")
    plt.show()

    return None
