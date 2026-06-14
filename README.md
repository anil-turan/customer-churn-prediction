# Customer Churn Prediction

End-to-end machine learning pipeline that predicts which telecom customers are likely to cancel their subscription.

**Dataset:** IBM Telco Customer Churn — 7,043 customers, 21 features  
**Best model:** LightGBM · AUC-PR 0.636 · AUC-ROC 0.829  
**Business impact:** cost-optimal threshold reduces misclassification cost by **43%** vs default (£37k → £65k on test set)  
**Stack:** Python 3.11 · scikit-learn · LightGBM · XGBoost · SHAP · MLflow · FastAPI

---

## Project Structure

```
customer-churn-prediction/
├── src/
│   ├── features/pipeline.py     # feature engineering functions
│   ├── models/train.py          # pipeline builder + MLflow logging
│   ├── evaluation/metrics.py    # ROC/PR/SHAP plot helpers
│   └── serving/app.py           # FastAPI prediction endpoint
├── notebooks/
│   ├── 01_eda.ipynb             # exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb  # CV, SHAP, MLflow, model export
│   └── 04_cost_sensitive.ipynb  # business cost matrix, threshold optimisation
├── tests/
│   └── test_pipeline.py         # leak test + dummy baseline test
├── data/
│   ├── raw/                     # original CSV (not committed)
│   └── processed/               # train/test splits (not committed)
├── saved_models/                # trained pipeline pickle
├── reports/figures/             # all EDA and evaluation plots
├── Dockerfile
├── pyproject.toml
└── .github/workflows/ci.yml
```

---

## Results

| Model | AUC-PR | AUC-ROC | Overfit gap |
|-------|--------|---------|-------------|
| Dummy baseline | 0.265 | 0.500 | — |
| XGBoost | 0.631 | 0.826 | 0.04 |
| **LightGBM** | **0.639** | **0.833** | 0.03 |

> AUC-PR is the primary metric because the dataset is imbalanced (26.5% churn rate). Accuracy would be misleading here.

**Test set (LightGBM, threshold = 0.5):**
- Correctly identified 266 out of 374 churners (71% recall)
- 230 false positives out of 1,035 non-churners (22% false alarm rate)
- Best F1 threshold: ~0.40

### Cost-Sensitive Analysis

In churn prediction, a missed churner (false negative) is far more expensive than a false alarm (false positive). The default 0.5 threshold treats both errors equally — this section corrects that.

**Cost assumptions (illustrative):**
- False Negative (missed churner): £500 — lost customer lifetime value
- False Positive (false alarm): £50 — wasted retention campaign spend
- FN/FP cost ratio: **10x**

**Threshold comparison (test set, 1,409 customers):**

| Threshold | Churners caught | Missed | Total cost |
|-----------|----------------|--------|------------|
| Default (0.50) | 266 / 374 | 108 | £65,500 |
| Best F1 (0.40) | 288 / 374 | 86 | £57,350 |
| **Cost-optimal (0.05)** | **363 / 374** | **11** | **£37,150** |

**Projected annual saving** (100k customers): **~£2M** vs default threshold.

Two levers are applied in combination:
1. **Threshold optimisation** — shift the decision boundary to reflect the asymmetric cost
2. **Cost-sensitive retraining** — assign higher sample weights to churners during training (weight = FN/FP ratio = 10)

A sensitivity analysis shows the optimal threshold ranges from 0.46 (low cost ratio) to 0.01 (high cost ratio) — confirming that the right threshold must always be derived from actual business costs, not set arbitrarily.

---

### Top SHAP Features

| Rank | Feature | Direction |
|------|---------|-----------|
| 1 | `tenure` | longer tenure → lower churn risk |
| 2 | `is_long_contract` | 2-year contracts strongly reduce churn |
| 3 | `charge_per_service` | high cost per service → higher churn |
| 4 | `Contract_Two year` | confirms contract signal |
| 5 | `MonthlyCharges` | high charges → higher churn |

---

## Key EDA Findings

| Segment | Churn Rate |
|---------|-----------|
| Month-to-month contract | **42.7%** |
| First 12 months (new customers) | **47.7%** |
| Electronic check payment | **45.3%** |
| Fiber optic internet | **41.9%** |
| Two-year contract | 2.8% |
| 37+ months tenure | 11.9% |

---

## Quickstart

**1. Install dependencies**
```bash
pip install -e ".[dev]"
```

**2. Download data**
```bash
kaggle datasets download -d blastchar/telco-customer-churn \
  -p data/raw --unzip
```

**3. Run notebooks in order**
```
notebooks/01_eda.ipynb
notebooks/02_feature_engineering.ipynb
notebooks/03_model_training.ipynb
notebooks/04_cost_sensitive.ipynb
```

**4. Run tests**
```bash
pytest tests/ -v --cov=src
```

**5. Start the API**
```bash
uvicorn src.serving.app:app --reload
```

---

## API

### `GET /health`
```json
{"status": "ok", "model": "LightGBM churn pipeline"}
```

### `POST /predict`

**Request body (example — high-risk customer):**
```json
{
  "tenure": 2,
  "MonthlyCharges": 95.5,
  "TotalCharges": 191.0,
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check"
}
```

**Response:**
```json
{
  "churn_probability": 0.9102,
  "churn_prediction": true,
  "risk_tier": "high"
}
```

`risk_tier` values: `"low"` (< 0.4) · `"medium"` (0.4–0.7) · `"high"` (≥ 0.7)

---

## Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

---

## Run with Docker

Interactive API docs available at `http://localhost:8000/docs` after starting the server.

---

## Technical Notes

- **No data leakage:** all transformers are fit on training data only and applied to test data via sklearn `Pipeline`
- **Imbalanced classes:** handled with `scale_pos_weight` in both XGBoost and LightGBM
- **Cost-sensitive learning:** churners assigned sample weight = FN/FP cost ratio (10x) during retraining
- **Threshold optimisation:** decision boundary set by minimising total business cost, not F1
- **Explainability:** SHAP `TreeExplainer` provides both global feature importance and per-customer waterfall explanations
- **Experiment tracking:** all runs logged to MLflow (metrics, params, model artifacts, figures)
- **CI:** GitHub Actions runs `ruff`, `black`, and `pytest` on every push to `main`
