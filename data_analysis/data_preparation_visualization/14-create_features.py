#!/usr/bin/env python3
"""
This module engineers new features from the datase.
"""
import pandas as pd


def create_features(df):
    """
    Creates NumServices and TenureGroup features.

    Args:
        df: The pandas DataFrame to modify.

    Returns:
        The modified DataFrame.
    """
    services_columns = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
        ]
    df["NumServices"] = 0
    for column in services_columns:
        if column == "InternetService":
            df["NumServices"] += df[column].isin(
                ["DSL", "Fiber optic"]
                ).astype(int)
        else:
            df["NumServices"] += (df[column] == "Yes").astype(int)

    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 60, float("inf")],
        labels=["0-12", "13-24", "25-48", "49-60", "60+"],
        include_lowest=False
        )
    df = df.drop(columns=services_columns + ["tenure"])

    return df
