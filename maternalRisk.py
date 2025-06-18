
# highRisk.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, RocCurveDisplay
)

# === Load and Preprocess Dataset ===
df = pd.read_excel(r"C:\Users\VAISHNAVI KALANCHA\Desktop\mini_project\Book2 (1).xlsx", sheet_name="Sheet1")

# Set the first row as headers
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

# Clean and convert data
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Weight'] = df['Weight'].str.replace(' kg', '', regex=False).astype(float)
df['Height'] = df['Height'].str.replace("''", '', regex=False).astype(float)
df['GestationalAge'] = df['GestationalAge'].str.extract(r'(\d+)').astype(float)
df['FetalHeartbeat'] = df['FetalHeartbeat'].str.replace('m', '', regex=False).astype(float)
df['HighRisk'] = df['HighRisk'].map({'Yes': 1, 'No': 0})

# Drop rows with missing target
df = df.dropna(subset=['HighRisk'])

# One-hot encode categorical columns
categorical_cols = [
    'Gravida', 'TetanusDose', 'BloodPressure', 'Anemia', 'Jaundice',
    'FetalPosition', 'FetalMovement', 'UrineAlbumin', 'UrineSugar',
    'VDRL', 'HepatitisB'
]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Drop identifier column
df = df.drop(columns=['Name'])

# Drop remaining missing values
df = df.dropna()

# Split features and target
X = df.drop(columns=['HighRisk'])
y = df['HighRisk']

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train Model ===
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# === Evaluate ===
preds = model.predict(X_test)
print("\n=== Classification Report ===")
print(classification_report(y_test, preds))

# === Cross Validation ===
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1_macro')
acc_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

print("\n=== Cross-Validation Results ===")
print("F1 Scores:", f1_scores)
print("Mean F1 Score:", np.mean(f1_scores))
print("Accuracy Scores:", acc_scores)
print("Mean Accuracy:", np.mean(acc_scores))

# === Confusion Matrix ===
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# === ROC Curve ===
y_probs = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
plt.title(f"ROC Curve (AUC = {roc_auc:.2f})")
plt.show()

# === Predict a Single Case ===
print("\n=== Single Case Prediction ===")

# Create a zero-filled DataFrame for the single case
new_case = pd.DataFrame(data=np.zeros((1, len(X.columns))), columns=X.columns)

# Define your example input (must match one-hot columns!)
example_input = {
    'Age': 26,
    'GestationalAge': 28,
    'Weight': 60,
    'Height': 155,
    'FetalHeartbeat': 140,

    'Gravida_2nd': 1,
    'Gravida_3rd': 0,

    'TetanusDose_2nd': 1,
    'TetanusDose_3rd': 0,

    'BloodPressure_100/70': 1,
    'BloodPressure_100/60': 0,
    'BloodPressure_100/65': 0,
    'BloodPressure_110/55': 0,
    'BloodPressure_110/60': 0,
    'BloodPressure_110/65': 0,
    'BloodPressure_110/80': 0,
    'BloodPressure_120/60': 0,
    'BloodPressure_80/60': 0,
    'BloodPressure_90/60': 0,

    'Anemia_Minimal': 1,
    'Jaundice_Minimal': 1,
    'FetalPosition_Normal': 1,

    'UrineAlbumin_Medium': 0,
    'UrineAlbumin_Minimal': 1,

    'UrineSugar_Yes': 0,
    'VDRL_Positive': 0,
    'HepatitisB_Positive': 0
}

# Fill in the values for the new case
for col, val in example_input.items():
    if col in new_case.columns:
        new_case.at[0, col] = val
    else:
        print(f"Warning: Column '{col}' not found in features.")

# Predict the risk
predicted_class = model.predict(new_case)[0]
predicted_proba = model.predict_proba(new_case)[0]

print(f"\nPredicted Risk Class: {'High Risk' if predicted_class == 1 else 'Low Risk'}")
print(f"Probability of High Risk: {predicted_proba[1]:.2f}")
print(f"Probability of Low Risk: {predicted_proba[0]:.2f}")
