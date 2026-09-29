"""
Scikit-Learn preprocessing transformers and pipeline definitions.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DomainFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Creates enterprise domain features:
    - avg_monthly_ratio: ratio of current monthly fee against total historical fee rate
    - support_intensity: tickets per tenure month
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        # Avoid division by zero
        safe_tenure = np.where(X_out["tenure_months"] == 0, 1, X_out["tenure_months"])
        
        # Historical rate ratio
        historical_rate = X_out["total_charges"] / safe_tenure
        X_out["charge_ratio"] = np.where(historical_rate > 0, X_out["monthly_charges"] / historical_rate, 1.0)
        
        # Support ticket intensity
        X_out["ticket_intensity"] = X_out["support_tickets"] / safe_tenure
        return X_out


def build_preprocessor() -> Pipeline:
    numeric_features = [
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "support_tickets",
        "charge_ratio",
        "ticket_intensity"
    ]
    categorical_features = [
        "contract_type",
        "internet_service",
        "payment_method",
        "paperless_billing"
    ]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    col_transformer = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    full_pipeline = Pipeline(steps=[
        ("domain_features", DomainFeatureExtractor()),
        ("encoder_scaler", col_transformer)
    ])

    return full_pipeline