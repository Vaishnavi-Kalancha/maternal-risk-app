import pandas as pd
import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, accuracy_score, make_scorer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# -------------------------------
# Generate Example Dataset with Feature Names
# -------------------------------
X_np, y = make_classification(
    n_samples=1000,
    n_features=24,
    n_informative=12,
    n_redundant=6,
    n_classes=2,
    random_state=42
)

# Convert to DataFrame to retain feature names
feature_names = [f"feature_{i}" for i in range(X_np.shape[1])]
X = pd.DataFrame(X_np, columns=feature_names)

# -------------------------------
# Create Cross-Validation Strategy
# -------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Scorers
f1 = make_scorer(f1_score)
accuracy = make_scorer(accuracy_score)

# -------------------------------
# XGBoost Classifier
# -------------------------------
xgb_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42,
    verbosity=0
)

xgb_f1_scores = cross_val_score(xgb_model, X, y, cv=skf, scoring=f1)
xgb_acc_scores = cross_val_score(xgb_model, X, y, cv=skf, scoring=accuracy)

print("===== XGBoost =====")
print(f"F1 Score (Mean): {xgb_f1_scores.mean():.4f}")
print(f"Accuracy (Mean): {xgb_acc_scores.mean():.4f}")

# -------------------------------
# LightGBM Classifier
# -------------------------------
lgbm_model = LGBMClassifier(
    objective='binary',
    random_state=42,
    verbose=-1,
    n_jobs=-1
)

lgbm_f1_scores = cross_val_score(lgbm_model, X, y, cv=skf, scoring=f1)
lgbm_acc_scores = cross_val_score(lgbm_model, X, y, cv=skf, scoring=accuracy)

print("\n===== LightGBM =====")
print(f"F1 Score (Mean): {lgbm_f1_scores.mean():.4f}")
print(f"Accuracy (Mean): {lgbm_acc_scores.mean():.4f}")
