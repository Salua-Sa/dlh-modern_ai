#!/usr/bin/env python3
"""
This module removes duplicate rows from a DataFrame.
"""


def remove_duplicates(df):
    """
    Removes duplicated rows a DataFrame.

    Args:
        df: The pandas DataFrame to process.

    Returns:
        The deduplicated DataFrame
    """
    df = df.drop_duplicates()

    return df
