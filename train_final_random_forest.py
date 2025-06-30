# === train_final_random_forest.py ===
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE

# Fix deprecated np.bool usage
np.bool = bool

# Load and preprocess dataset
df = pd.read_excel("Book2 (2).xlsx", sheet_name="Sheet1", header=1)

# Rename Bengali to English
df.rename(columns={
    'Name': 'Name',
    'Age': 'Age',
    'Gravida': 'Gravida',
    'TiTi Tika': 'TetanusDose',
    'গর্ভকাল': 'GestationalAge',
    'ওজন': 'Weight',
    'উচ্চতা': 'Height',
    'রক্ত চাপ': 'BloodPressure',
    'রক্তস্বল্পতা': 'Anemia',
    'জন্ডিস': 'Jaundice',
    'গর্ভস্হ শিশু অবস্থান': 'FetalPosition',
    'গর্ভস্হ শিশু নাড়াচাড়া': 'FetalMovement',
    'গর্ভস্হ শিশু হৃৎস্পন্দন': 'FetalHeartbeat',
    'প্রসাব পরিক্ষা এলবুমিন': 'UrineAlbumin',
    'প্রসাব পরিক্ষা সুগার': 'UrineSugar',
    'VDRL': 'VDRL',
    'HRsAG': 'HepatitisB',
    'ঝুকিপূর্ণ গর্ভ': 'HighRisk'
}, inplace=True)

# Drop Name column
df.drop(columns=['Name'], inplace=True, errors='ignore')

# Clean and convert numeric fields
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Weight'] = df['Weight'].astype(str).str.replace(' kg', '', regex=False).astype(float)
df['Height'] = df['Height'].astype(str).str.replace("''", '', regex=False).astype(float)
df['GestationalAge'] = df['GestationalAge'].astype(str).str.extract(r'(\d+)').astype(float)
df['FetalHeartbeat'] = df['FetalHeartbeat'].astype(str).str.replace('m', '', regex=False).astype(float)
df['HighRisk'] = df['HighRisk'].map({'Yes': 1, 'No': 0})

# Drop rows with missing target
df.dropna(subset=['HighRisk'], inplace=True)

# Define all expected categories for one-hot features
cat_features = {
    'Gravida': ['1st', '2nd', '3rd'],
    'TetanusDose': ['2nd', '3rd'],
    'BloodPressure': ["100/60", "100/65", "100/70", "110/55", "110/60", "110/65", "110/80", "120/60", "80/60", "90/60"],
    'Anemia': ['Minimal'],
    'Jaundice': ['Minimal'],
    'FetalPosition': ['Normal'],
    'FetalMovement': ['Normal'],
    'UrineAlbumin': ['Minimal', 'Medium'],
    'UrineSugar': ['Yes'],
    'VDRL': ['Positive'],
    'HepatitisB': ['Positive', 'Negative']
}

# Apply one-hot encoding with expected categories
df = pd.get_dummies(df, columns=cat_features.keys())

# Ensure all possible one-hot columns exist
template = pd.DataFrame(columns=[])
for feature, values in cat_features.items():
    for val in values:
        col = f"{feature}_{val}"
        if col not in df.columns:
            df[col] = 0

# Drop rows with missing values
df.dropna(inplace=True)

# Train/Test Split
X = df.drop(columns=['HighRisk'])
y = df['HighRisk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_bal, y_train_bal)

# Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

# Save model
joblib.dump(model, "model.pkl")
joblib.dump(X.columns.tolist(), "feature_order.pkl")
print("\n✅ Model and feature order saved.")
