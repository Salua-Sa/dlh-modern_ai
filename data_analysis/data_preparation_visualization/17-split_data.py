#!/usr/bin/env python3
"""
This module splits data into train/test sets.
"""
from sklearn import model_selection


def split_data(df, target='Churn', test_size=0.2, random_state=42):
    """
    Splits a DataFrame into training and testing sets.

    Args:
        df: The pandas DataFrame to modify.
        target: The name of the target column.
        test_size: The proportion of data used for testing.
        random_state: The random seed for reproducibility

    Returns:
        A tuple containing X_train, X_test, y_train, y_test.
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
        )

    return X_train, X_test, y_train, y_test
