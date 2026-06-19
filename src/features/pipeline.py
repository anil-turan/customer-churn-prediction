import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_feature_pipeline(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop")


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-specific features from the raw Telco columns.

    This function is the single source of truth for feature creation.
    Both the training notebooks and the serving API import it from here.
    """
    df = df.copy()
    SERVICE_COLS = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    # Group customers by how long they have been with the company
    if "tenure" in df.columns:
        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=[0, 12, 36, np.inf],
            labels=["new", "mid", "long"],
        ).astype(str)
    available = [c for c in SERVICE_COLS if c in df.columns]
    if available:
        df["total_services"] = (df[available] == "Yes").sum(axis=1)
    # Higher cost per service is a strong churn signal (top SHAP feature)
    if "MonthlyCharges" in df.columns and "total_services" in df.columns:
        df["charge_per_service"] = df["MonthlyCharges"] / (df["total_services"] + 1)
    if "OnlineSecurity" in df.columns and "TechSupport" in df.columns:
        df["has_security"] = (
            (df["OnlineSecurity"] == "Yes") | (df["TechSupport"] == "Yes")
        ).astype(int)
    if "Contract" in df.columns:
        df["is_long_contract"] = (df["Contract"] != "Month-to-month").astype(int)
    return df
