#!/usr/bin/env python3
"""
This module creates and returns a Ridge
Regression model using Scikit-learn.
"""
from sklearn import linear_model


def ridge_regression(random_state):
    """
     Creates and returns a Ridge Regression model using Scikit-learn.
     Ridge Regression extends ordinary linear regression
     by adding L2 regularization, which helps stabilize
     the model by shrinking large coefficients.

    Args:
        random_state: An integer used to set the random
        seed for reproducibility.

    Returns:
        model: An untrained Ridge regression model instance.
    """
    model = linear_model.Ridge(random_state=random_state)
    return model
