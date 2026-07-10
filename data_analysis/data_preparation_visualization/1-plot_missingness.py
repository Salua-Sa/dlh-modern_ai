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
    plt.figure(figsize=(12, 8))
    missing_rows, missing_cols = np.where(df.isna()) # find missing values positions

    plt.scatter(missing_rows, missing_cols, marker="|") # plot missing values 

    plt.yticks(
        ticks=range(len(df.columns)),
                    labels=df.columns
        ) # map y-axis numbers to column names

    plt.tight_layout()
    plt.show()

    return None
