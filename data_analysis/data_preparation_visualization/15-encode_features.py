#!/usr/bin/env python3
"""
This module encodes features for modeling using Scikit-learn.
"""
import pandas as pd
from sklearn import preprocessing


def encode_features(df):
    """
    Encodes selected categorical features.

    Args:
        df: The pandas DataFrame to encode.

    Returns:
        The encoded DataFrame and fitted encoders.
    """
    churn_label_encoded = preprocessing.LabelEncoder()
    df["Churn"] = churn_label_encoded .fit_transform(df["Churn"])

    binary_columns = [
        "Partner",
        "Dependents",
        "PaperlessBilling",
        "SeniorCitizen"
        ]

    ordinal_encoder_binary = preprocessing.OrdinalEncoder(
        categories=[["No", "Yes"]]

    df[binary_columns] = ordinal_encoder_binary.fit_transform(
        df[binary_columns])
    df[binary_columns] = df[binary_columns].astype(int)

    df = pd.get_dummies(
        df,
        columns=["Contract", "PaymentMethod"],
        drop_first=True,
        dtype=int)

    ordinal_encoder_tenure = preprocessing.OrdinalEncoder()

    df[["TenureGroup"]] = ordinal_encoder_tenure.fit_transform(
        df[["TenureGroup"]])

    df[["TenureGroup"]] = df[["TenureGroup"]].astype(int)

    return (df, churn_label_encoded,
            ordinal_encoder_binary, ordinal_encoder_tenure)
