import mlflow
import mlflow.sklearn
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline

from src.features.pipeline import build_feature_pipeline, add_derived_features

SCORERS = {
    "roc_auc":  make_scorer(roc_auc_score, needs_proba=True),
    "avg_prec": make_scorer(average_precision_score, needs_proba=True),
}

NUM_COLS = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "total_services", "charge_per_service",
]
CAT_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
    "tenure_group",
]


def build_churn_pipeline(scale_pos_weight: float = 10.0) -> Pipeline:
    preprocessor = build_feature_pipeline(NUM_COLS, CAT_COLS)
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("prep", preprocessor), ("model", model)])


def cross_validate_pipeline(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = cross_validate(
        pipeline, X, y, cv=cv, scoring=SCORERS, return_train_score=True
    )
    summary = {}
    for metric in SCORERS:
        test_scores = results[f"test_{metric}"]
        summary[metric] = {
            "mean":        float(test_scores.mean()),
            "std":         float(test_scores.std()),
            "overfit_gap": float(results[f"train_{metric}"].mean() - test_scores.mean()),
        }
    return summary


def train_and_log(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    run_name: str = "xgb_churn_baseline",
) -> dict:
    with mlflow.start_run(run_name=run_name):
        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = {
            "roc_auc":  roc_auc_score(y_test, proba),
            "avg_prec": average_precision_score(y_test, proba),
        }
        mlflow.log_params(pipeline.named_steps["model"].get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, "model")
    return metrics
