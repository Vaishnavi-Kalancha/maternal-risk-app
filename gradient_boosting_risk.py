import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier  # You can switch to HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, RocCurveDisplay
)

# === Load and preprocess dataset ===
df = pd.read_excel(r"C:\Users\VAISHNAVI KALANCHA\Desktop\mini_project\Book2 (1).xlsx", sheet_name="Sheet1")

# Use first row as header
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

# Clean and convert numeric columns
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Weight'] = df['Weight'].str.replace(' kg', '', regex=False).astype(float)
df['Height'] = df['Height'].str.replace("''", '', regex=False).astype(float)
df['GestationalAge'] = df['GestationalAge'].str.extract(r'(\d+)').astype(float)
df['FetalHeartbeat'] = df['FetalHeartbeat'].str.replace('m', '', regex=False).astype(float)
df['HighRisk'] = df['HighRisk'].map({'Yes': 1, 'No': 0})

# Drop rows with missing target
df = df.dropna(subset=['HighRisk'])

# 🔍 Class balance check
print("\n=== Class Distribution ===")
print(df['HighRisk'].value_counts())

# One-hot encoding
categorical_cols = [
    'Gravida', 'TetanusDose', 'BloodPressure', 'Anemia', 'Jaundice',
    'FetalPosition', 'FetalMovement', 'UrineAlbumin', 'UrineSugar',
    'VDRL', 'HepatitisB'
]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Drop name column and any remaining NaNs
df = df.drop(columns=['Name'], errors='ignore')
df = df.dropna()

# === Train/test split ===
X = df.drop(columns=['HighRisk'])
y = df['HighRisk']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# === Model training ===
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# === Evaluation ===
preds = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]

print("\n=== Classification Report ===")
print(classification_report(y_test, preds))

print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_test, preds)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low Risk', 'High Risk']).plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# === ROC Curve ===
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
plt.title(f"ROC Curve (AUC = {roc_auc:.2f})")
plt.show()

# === Cross-validation ===
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1_macro')
acc_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

print("\n=== Cross-Validation ===")
print("F1 Macro Mean:", np.mean(f1_scores))
print("Accuracy Mean:", np.mean(acc_scores))

# === SHAP Explainability ===
print("\n=== SHAP Explainability ===")
X_shap = X_train.astype(np.float64)
X_test_shap = X_test.astype(np.float64)

explainer = shap.Explainer(model, X_shap)
shap_values = explainer(X_test_shap)

plt.figure(figsize=(12, 6))
shap.summary_plot(shap_values, X_test_shap, show=False)
plt.tight_layout()
plt.savefig("shap_summary_plot.png", dpi=300)
plt.show()

plt.figure(figsize=(10, 5))
shap.plots.bar(shap_values, show=False)
plt.tight_layout()
plt.savefig("shap_bar_plot.png", dpi=300)
plt.show()

# === Save model and feature order for Streamlit ===
joblib.dump(model, "model.pkl")
joblib.dump(list(X.columns), "feature_order.pkl")
print("✅ Model and feature order saved.")

