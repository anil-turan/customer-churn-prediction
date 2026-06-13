import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


def full_report(y_true: pd.Series, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc":    roc_auc_score(y_true, y_proba),
        "avg_prec":   average_precision_score(y_true, y_proba),
        "brier":      brier_score_loss(y_true, y_proba),
    }


def plot_evaluation_suite(y_true: pd.Series, y_proba: np.ndarray, save_dir: str = "reports/figures"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    RocCurveDisplay.from_predictions(y_true, y_proba, ax=axes[0])
    axes[0].set_title("ROC Curve")

    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=axes[1])
    axes[1].set_title("Precision-Recall Curve")

    # Calibration curve
    fraction_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10)
    axes[2].plot(mean_pred, fraction_pos, marker="o", label="Model")
    axes[2].plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    axes[2].set_xlabel("Mean predicted probability")
    axes[2].set_ylabel("Fraction of positives")
    axes[2].set_title("Calibration Curve")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/evaluation_suite.png", dpi=150)
    plt.close()


def plot_shap_summary(pipeline, X_test: pd.DataFrame, save_dir: str = "reports/figures"):
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["prep"]
    X_transformed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
