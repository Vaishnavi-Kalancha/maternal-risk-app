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
df = pd.read_excel("Book2 (2).xlsx", sheet_name="Sheet1", header=1)

# Show original column names to debug
print("Excel column headers:", df.columns.tolist())

# === Rename Bengali columns to English ===
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

# === Clean and convert columns ===
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Weight'] = df['Weight'].astype(str).str.replace(' kg', '', regex=False).astype(float)
df['Height'] = df['Height'].astype(str).str.replace("''", '', regex=False).astype(float)
df['GestationalAge'] = df['GestationalAge'].astype(str).str.extract(r'(\d+)').astype(float)
df['FetalHeartbeat'] = df['FetalHeartbeat'].astype(str).str.replace('m', '', regex=False).astype(float)
df['HighRisk'] = df['HighRisk'].map({'Yes': 1, 'No': 0})

# Drop unusable and NaNs
df = df.drop(columns=['Name'], errors='ignore')
df = df.dropna(subset=['HighRisk'])

# === One-hot encode categorical columns ===
df = pd.get_dummies(df, columns=[
    'Gravida', 'TetanusDose', 'BloodPressure', 'Anemia', 'Jaundice',
    'FetalPosition', 'FetalMovement', 'UrineAlbumin', 'UrineSugar',
    'VDRL', 'HepatitisB'
], drop_first=False)

# Final cleanup
df = df.dropna()

# === Split features and target ===
X = df.drop(columns=['HighRisk'])
y = df['HighRisk']

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# === Balance classes with SMOTE ===
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
np.bool = bool  # Patch deprecated type for SHAP

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
print("Training samples after SMOTE:", X_train_bal.shape[0])
print("Class distribution after SMOTE:\n", y_train_bal.value_counts())
