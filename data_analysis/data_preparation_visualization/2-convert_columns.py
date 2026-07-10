#!/usr/bin/env python3
"""
This module performs type conversion for specific columns in a DataFrame
"""
import pandas as pd


def convert_columns(df):
    """
    Converts selected columns to
the required data types.

    Args:
        df: The pandas DataFrame to modify.

    Returns:
        The modified  DataFrame.
    """

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

    df['SeniorCitizen'] = df['SeniorCitizen'].map({
        0: "No",
        1: "Yes"})

    return df
