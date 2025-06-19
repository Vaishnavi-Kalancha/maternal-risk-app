
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score
)
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# === Load and preprocess ===
df = pd.read_excel(r"C:\Users\VAISHNAVI KALANCHA\Desktop\mini_project\Book2 (1).xlsx", sheet_name="Sheet1")
df.columns = df.iloc[0]
df = df[1:].reset_index(drop=True)
df.columns.name = None

df.rename(columns={
    'ANCC REGISTER': 'Name',
    'Unnamed: 1': 'Age',
    'Unnamed: 2': 'Gravida',
    'Unnamed: 3': 'TetanusDose',
    'Unnamed: 4': 'GestationalAge',
    'Unnamed: 5': 'Weight',
    'Unnamed: 6': 'Height',
    'Unnamed: 7': 'BloodPressure',
    'Unnamed: 8': 'Anemia',
    'Unnamed: 9': 'Jaundice',
    'Unnamed: 10': 'FetalPosition',
    'Unnamed: 11': 'FetalMovement',
    'Unnamed: 12': 'FetalHeartbeat',
    'Unnamed: 13': 'UrineAlbumin',
    'Unnamed: 14': 'UrineSugar',
    'Unnamed: 15': 'VDRL',
    'Unnamed: 16': 'HepatitisB',
    'Unnamed: 17': 'HighRisk'
}, inplace=True)

df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Weight'] = df['Weight'].str.replace(' kg', '', regex=False).astype(float)
df['Height'] = df['Height'].str.replace("''", '', regex=False).astype(float)
df['GestationalAge'] = df['GestationalAge'].str.extract(r'(\d+)').astype(float)
df['FetalHeartbeat'] = df['FetalHeartbeat'].str.replace('m', '', regex=False).astype(float)
df['HighRisk'] = df['HighRisk'].map({'Yes': 1, 'No': 0})
df = df.dropna(subset=['HighRisk'])

# Show class imbalance
print("\n=== Class Distribution ===")
print(df['HighRisk'].value_counts())

# One-hot encoding
cat_cols = [
    'Gravida', 'TetanusDose', 'BloodPressure', 'Anemia', 'Jaundice',
    'FetalPosition', 'FetalMovement', 'UrineAlbumin', 'UrineSugar',
    'VDRL', 'HepatitisB'
]
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
df = df.drop(columns=['Name'], errors='ignore')
df = df.dropna()

X = df.drop(columns=['HighRisk'])
y = df['HighRisk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

def evaluate_model(name, model):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)

    print(f"\n=== {name} ===")
    print(classification_report(y_test, preds))
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"F1 Score: {f1_score(y_test, preds):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, proba):.4f}")

# === Models ===

# 1. Random Forest with class weights
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

# 2. LightGBM with imbalance flag
lgb_model = lgb.LGBMClassifier(n_estimators=100, is_unbalance=True, random_state=42)

# 3. HistGradientBoosting with balanced classes
hgb = HistGradientBoostingClassifier(max_iter=100, class_weight='balanced', random_state=42)

# === Evaluate All ===
evaluate_model("Random Forest (Balanced)", rf)
evaluate_model("LightGBM (is_unbalance=True)", lgb_model)
evaluate_model("HistGradientBoosting (Balanced)", hgb)
