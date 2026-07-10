#!/usr/bin/env python3
"""
This module removes a column from a DataFrame.
"""


def drop_customerID(df):
    """
    Drops the customerID column from df.

    Args:
        df: The pandas DataFrame cointaining a customerID column.

    Returns:
        The modified DataFrame.
    """
    df = df.drop(columns=['customerID'])

    return df
