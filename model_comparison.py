
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

# NEW IMPORTS
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# === Load and Preprocess Dataset ===
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

categorical_cols = [
    'Gravida', 'TetanusDose', 'BloodPressure', 'Anemia', 'Jaundice',
    'FetalPosition', 'FetalMovement', 'UrineAlbumin', 'UrineSugar',
    'VDRL', 'HepatitisB'
]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
df = df.drop(columns=['Name'])
df = df.dropna()

X = df.drop(columns=['HighRisk'])
y = df['HighRisk']

# === Standardize ===
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# === Models ===
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(),
    "SVM": SVC(probability=True)
}

results = []

# === Split Ratios ===
split_ratios = [(0.8, 0.2), (0.7, 0.3), (0.6, 0.4)]

for train_size, test_size in split_ratios:
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, train_size=train_size, random_state=42, stratify=y)

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]  # Only for binary

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, y_proba)

        results.append({
            "Model": model_name,
            "Train-Test Split": f"{int(train_size*100)}-{int(test_size*100)}",
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC AUC": roc_auc
        })

        # === Confusion Matrix only at 70-30 split ===
        if train_size == 0.7:
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
            plt.title(f"Confusion Matrix - {model_name} (70-30 Split)")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            plt.show()

# === Results DataFrame ===
results_df = pd.DataFrame(results)

# === Show Results for Each Split ===
splits = results_df["Train-Test Split"].unique()

for split in splits:
    split_df = results_df[results_df["Train-Test Split"] == split]

    print(f"\n=== Performance Metrics for Train-Test Split {split} ===\n")
    print(split_df[[ "Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC" ]].to_string(index=False))

    # === Accuracy Bar Plot ===
    plt.figure(figsize=(10, 6))
    sns.barplot(data=split_df, x="Model", y="Accuracy", palette="Set2", hue="Model", legend=False)
    plt.title(f"Accuracy Comparison - Train-Test Split {split}", fontsize=14)
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()
