#!/usr/bin/env python3
"""
This module performs Welch's t-tests for
continuous numeric features using scipy.
"""
from scipy import stats


def ttest_numeric(df):
    """
    Computes Welch's t-tests p-values
for numeric columns against Churn

    Args:
        df: The pandas DataFrame
    containing a Churn columns.

    Returns:
        A dictionary mapping numeric feature names to p-values.
    """
    dic_result = {}

    for column in df.columns:
        if df[column].dtype != "object" and column != "Churn":
            churn_yes = df[df["Churn"] == "Yes"][column].dropna()
            churn_no = df[df["Churn"] == "No"][column].dropna()

            test_result = stats.ttest_ind(
                churn_yes,
                churn_no,
                equal_var=False
                )
            p_value = test_result.pvalue

            dic_result[column] = float(p_value)

    return dic_result
