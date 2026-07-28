#!/usr/bin/env python3
"""
This module creates a linear regression model using Scikit-learn.
"""
from sklearn import linear_model


def Linear_Regression():
    """
    Create a linear regression model using Scikit-learn,
    which uses ordinary least squares to fit a linear
    model to the data.

    Args:
        None

    Returns:
        model: An untrained LinearRegression instance.
    """
    model = linear_model.LinearRegression()
    return model
