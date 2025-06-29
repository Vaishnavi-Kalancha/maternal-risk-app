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

# === Load dataset ===
df = pd.read_excel("Book2 (1).xlsx", sheet_name="Sheet1")
df.columns = df.iloc[0]
df = df[1:].reset_index(drop=True)
df.columns.name = None

# === Rename columns ===
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

# === Clean and convert columns ===
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Weight'] = df['Weight'].str.replace(' kg', '', regex=False).astype(float)
df['Height'] = df['Height'].str.replace("''", '', regex=False).astype(float)
df['GestationalAge'] = df['GestationalAge'].str.extract(r'(\d+)').astype(float)
df['FetalHeartbeat'] = df['FetalHeartbeat'].str.replace('m', '', regex=False).astype(float)
df['HighRisk'] = df['HighRisk'].map({'Yes': 1, 'No': 0})

# Drop unusable and NaNs
df = df.drop(columns=['Name'], errors='ignore')
df = df.dropna(subset=['HighRisk'])

# One-hot encode all relevant columns
df = pd.get_dummies(df, columns=[
    'Gravida', 'TetanusDose', 'BloodPressure', 'Anemia', 'Jaundice',
    'FetalPosition', 'FetalMovement', 'UrineAlbumin', 'UrineSugar',
    'VDRL', 'HepatitisB'
], drop_first=False)  # Keep all to prevent mismatch

# Drop rows with remaining NaNs
df = df.dropna()

# === Train/Test Split ===
X = df.drop(columns=['HighRisk'])
y = df['HighRisk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# === Balance classes ===
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# === Train Random Forest ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_bal, y_train_bal)

# === Evaluation ===
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

# === Save model and feature order ===
joblib.dump(model, "model.pkl")
joblib.dump(X.columns.tolist(), "feature_order.pkl")
print("✅ Model and feature order saved.")

# === SHAP Summary Plot ===
import numpy as np
np.bool = bool  # Quick fix for deprecated np.bool usage in SHAP

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values[1], X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("shap_summary_bar.png", dpi=300)
plt.close()

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Risk", "High Risk"], yticklabels=["No Risk", "High Risk"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()

# === ROC Curve ===
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="darkorange")
plt.plot([0, 1], [0, 1], linestyle="--", color="navy")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=300)
plt.close()

print("✅ All plots and model files generated.")
