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
    df = df.copy()
    # Tenure grupları: yeni / orta / uzun vadeli müşteri
    if "tenure" in df.columns:
        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=[0, 12, 36, np.inf],
            labels=["new", "mid", "long"],
        )
    # Toplam hizmet sayısı
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    available = [c for c in service_cols if c in df.columns]
    if available:
        df["total_services"] = (df[available] == "Yes").sum(axis=1)
    # Aylık ücret / toplam hizmet
    if "MonthlyCharges" in df.columns and "total_services" in df.columns:
        df["charge_per_service"] = df["MonthlyCharges"] / (df["total_services"] + 1)
    return df
