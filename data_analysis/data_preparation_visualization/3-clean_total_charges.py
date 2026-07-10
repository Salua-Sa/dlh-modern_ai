#!/usr/bin/env python3
"""
This module handles missing values in a DataFrame.
"""


def clean_total_charges(df, method='drop'):
    """
    Handles missing values in the totalCharges column.

    Args:
        df: The pandas DataFrame to modify.
        method: The strategy to use: drop, median, or impute.

    Returns:
        The modified DataFrame.
    """

    df = df.copy()
    if method == 'drop':
        df = df.dropna(subset=['TotalCharges'])
    elif method == 'median':
        df['TotalCharges'] = df['TotalCharges'].fillna(
            df['TotalCharges'].median())
    elif method == "impute":
        df['TotalCharges'] = df['TotalCharges'].fillna(
            df['MonthlyCharges'] * df['tenure'])

    return df
