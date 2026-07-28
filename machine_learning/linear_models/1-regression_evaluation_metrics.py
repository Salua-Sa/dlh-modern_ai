#!/usr/bin/env python3
"""
This module computes common evaluation metrics
for regression tasks using Scikit-learn.
"""
from sklearn import metrics
import numpy as np


def evaluation_metrics_for_regression(y_true, y_pred):
    """
    Compute common evaluation metrics for regression tasks using Scikit-learn.

    Args:
        y_true: A 1D NumPy array containing the true target values.
        y_pred: A 1D NumPy array containing the predicted target values.

    Returns:
        mA tuple (mse, rmse, mae, r2) where:
        mse: Mean Squared Error – average of the squared
             differences between actual and predicted values.
        rmse: Root Mean Squared Error – square root of the MSE,
              representing error in the original units.
        mae: Mean Absolute Error – average absolute difference
             between actual and predicted values.
        r2: R² Score – indicates how well the model explains the
            variance in the target variable (1 means perfect prediction).
    """
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = metrics.root_mean_squared_error(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    return (mse, rmse, mae, r2)
