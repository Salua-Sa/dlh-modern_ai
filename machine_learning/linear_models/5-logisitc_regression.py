#!/usr/bin/env python3
"""
This module creates a logistic regression model using Scikit-learn,
which performs binary classification by fitting a logistic function.
"""
from sklearn import linear_model


def Logistic_Regression_Model(random_state):
    """
     Create a logistic regression model using Scikit-learn,
     which performs binary classification by fitting a logistic function.

    Args:
        random_state: An integer used to set the random
                      seed for reproducibility.

    Returns:
        model: An untrained LogisticRegression instance.
    """
    model = linear_model.LogisticRegression(random_state=random_state)
    return model
