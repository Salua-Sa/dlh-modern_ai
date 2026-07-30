#!/usr/bin/env python3
"""
This module creates a Support Vector Machine (SVM)
classifier using Scikit-learn with the specified kernel.
"""
from sklearn import svm


def get_SVM_model(name, random_state):
    """
     Create a Support Vector Machine (SVM) classifier using
     Scikit-learn with the specified kernel.

    Args:
        name: A string indicating the type of model to return.
        Accepted values are:
        - 'linear': returns a SVM model with a linear kernel.
        - 'poly': returns a SVM model with a polynomial kernel.
        - 'rbf': returns a SVM model with a radial basis function (RBF) kernel.
        - random_state: The seed used by the random number
                        generator for reproducibility.

    Returns:
        An untrained instance of SVC.
    """
    if name not in ["linear", "poly", "rbf"]:
        raise TypeError

    scv = svm.SVC(kernel=name, random_state=random_state)
    return scv
