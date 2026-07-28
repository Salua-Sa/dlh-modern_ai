#!/usr/bin/env python3
"""
This module creates and returns a
Lasso Regression model using Scikit-learn.
"""
from sklearn import linear_model


def lasso_regression(random_state):
    """
     Creates and returns a Lasso Regression model using Scikit-learn.
     Lasso Regression extends ordinary linear regression by adding L1
     regularization, which helps simplify the model by forcing some
     coefficients to zero, enabling automatic feature selection.

    Args:
        random_state: An integer used to set the random
        seed for reproducibility.

    Returns:
        model: An untrained Lasso regression model instance.
    """
    model = linear_model.Lasso(random_state=random_state)
    return model
