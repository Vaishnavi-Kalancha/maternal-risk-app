import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, roc_auc_score
)
from imblearn.over_sampling import SMOTE

# === Load dataset ===
df = pd.read_excel("Book2 (1).xlsx", sheet_name="Sheet1")
df.columns = df.iloc[0]
df = df[1:].reset_index(drop=True)
df.columns.name = None

# Rename columns
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

# Convert numeric fields
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Weight'] = df['Weight'].str.replace(' kg', '', regex=False).astype(float)
df['Height'] = df['Height'].str.replace("''", '', regex=False).astype(float)
df['GestationalAge'] = df['GestationalAge'].str.extract(r'(\d+)').astype(float)
df['FetalHeartbeat'] = df['FetalHeartbeat'].str.replace('m', '', regex=False).astype(float)
df['HighRisk'] = df['HighRisk'].map({'Yes': 1, 'No': 0})

df = df.drop(columns=['Name'], errors='ignore')
df = df.dropna(subset=['HighRisk'])
df = pd.get_dummies(df, columns=[
    'Gravida', 'TetanusDose', 'BloodPressure', 'Anemia', 'Jaundice',
    'FetalPosition', 'FetalMovement', 'UrineAlbumin', 'UrineSugar',
    'VDRL', 'HepatitisB'
], drop_first=True)

df = df.dropna()

# === Train/Test Split ===
X = df.drop(columns=['HighRisk'])
y = df['HighRisk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# === Apply SMOTE ===
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# === Train Random Forest ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_bal, y_train_bal)

# === Evaluate ===
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n=== Evaluation Report ===")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

# === Save model and feature order ===
joblib.dump(model, "model.pkl")
joblib.dump(list(X.columns), "feature_order.pkl")
print("✅ Model and feature order saved.")
