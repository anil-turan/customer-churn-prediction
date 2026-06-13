import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score

from src.features.pipeline import add_derived_features, build_feature_pipeline
from src.models.train import NUM_COLS, CAT_COLS, build_churn_pipeline


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "tenure":          np.random.randint(1, 72, n),
        "MonthlyCharges":  np.random.uniform(20, 120, n),
        "TotalCharges":    np.random.uniform(100, 8000, n),
        "gender":          np.random.choice(["Male", "Female"], n),
        "SeniorCitizen":   np.random.choice([0, 1], n),
        "Partner":         np.random.choice(["Yes", "No"], n),
        "Dependents":      np.random.choice(["Yes", "No"], n),
        "PhoneService":    np.random.choice(["Yes", "No"], n),
        "MultipleLines":   np.random.choice(["Yes", "No", "No phone service"], n),
        "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity":  np.random.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup":    np.random.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection":np.random.choice(["Yes", "No", "No internet service"], n),
        "TechSupport":     np.random.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV":     np.random.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], n),
        "Contract":        np.random.choice(["Month-to-month", "One year", "Two year"], n),
        "PaperlessBilling":np.random.choice(["Yes", "No"], n),
        "PaymentMethod":   np.random.choice(["Electronic check", "Mailed check",
                                              "Bank transfer", "Credit card"], n),
        "Churn":           np.random.choice([0, 1], n, p=[0.73, 0.27]),
    })


def test_derived_features_created(sample_df):
    df = add_derived_features(sample_df)
    assert "tenure_group" in df.columns
    assert "total_services" in df.columns
    assert "charge_per_service" in df.columns


def test_no_data_leakage(sample_df):
    """Train AUC > Test AUC by a small margin — pipeline must not leak."""
    df = add_derived_features(sample_df)
    X, y = df.drop("Churn", axis=1), df["Churn"]
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    pipeline = build_churn_pipeline()
    pipeline.fit(X_train, y_train)

    train_auc = roc_auc_score(y_train, pipeline.predict_proba(X_train)[:, 1])
    test_auc  = roc_auc_score(y_test,  pipeline.predict_proba(X_test)[:, 1])

    # Overfitting gap must be small — large gap signals leakage
    assert train_auc - test_auc < 0.30, f"Overfit gap too large: {train_auc - test_auc:.3f}"


def test_beats_dummy_baseline(sample_df):
    """Model must outperform a majority-class dummy on AUC-ROC."""
    df = add_derived_features(sample_df)
    X, y = df.drop("Churn", axis=1), df["Churn"]
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_auc = roc_auc_score(y_test, dummy.predict_proba(X_test)[:, 1])

    pipeline = build_churn_pipeline()
    pipeline.fit(X_train, y_train)
    model_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])

    assert model_auc > dummy_auc, f"Model ({model_auc:.3f}) did not beat dummy ({dummy_auc:.3f})"
