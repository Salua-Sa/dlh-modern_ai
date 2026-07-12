#!/usr/bin/env python3
"""
This module standardizes numeric columns.
"""
from sklearn import preprocessing


def scale_numeric(df):
    """
    Standardizes MonthlyCharges and TotalCharges columns.

    Args:
        df: The pandas DataFrame to modify.

    Returns:
        The modified DataFrame.
    """
    columns_to_scale = [
        "MonthlyCharges",
        "TotalCharges"]
    scaler = preprocessing.StandardScaler()

    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    return df
