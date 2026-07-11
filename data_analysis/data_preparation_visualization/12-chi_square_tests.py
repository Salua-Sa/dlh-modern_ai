#!/usr/bin/env python3
"""
This module performs chi-square tests for categorical features, using SciPy.
"""
import pandas as pd
from scipy import stats


def chi_square_tests(df):
    """
    Computes chi-square p-values
for categorical columns against Churn

    Args:
        df: The pandas DataFrame
    containing Churn and categorical columns.

    Returns:
        A dictionary mapping feature names to p-values.
    """
    dic_result = {}

    for column in df.columns:
        if df[column].dtype == "object" and column != "Churn":
            table = pd.crosstab(df[column], df["Churn"])

            test_result = stats.chi2_contingency(table)

            chi2 = test_result[0]
            p_value = test_result[1]
            dof = test_result[2]
            expected = test_result[3]

            dic_result[column] = p_value
    return dic_result
