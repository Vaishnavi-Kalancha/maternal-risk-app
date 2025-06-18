import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# === Load model and feature order ===
model = joblib.load("model.pkl")
feature_order = joblib.load("feature_order.pkl")

st.set_page_config(page_title="Maternal Risk Predictor", layout="centered")
st.title("🤰 Maternal Health Risk Predictor")
st.write("Enter the patient’s clinical details below to predict the risk level.")

# === User Inputs ===
age = st.number_input("Age", min_value=15, max_value=50, value=25)
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=120.0, value=60.0)
height = st.number_input("Height (cm)", min_value=130.0, max_value=180.0, value=155.0)
gest_age = st.slider("Gestational Age (weeks)", 0, 40, 28)
fhr = st.slider("Fetal Heart Rate (bpm)", 100, 180, 140)

anemia_min = st.checkbox("Anemia Minimal")
urine_sugar = st.checkbox("Urine Sugar Present")
jaundice_min = st.checkbox("Jaundice Minimal")
hepatitis_b = st.checkbox("Hepatitis B Positive")
vdrl_pos = st.checkbox("VDRL Positive")
fetal_pos_normal = st.checkbox("Fetal Position Normal")

gravida = st.selectbox("Gravida", ["1st", "2nd", "3rd"])
tetanus = st.selectbox("Tetanus Dose", ["1st", "2nd", "3rd"])
bp = st.selectbox("Blood Pressure", [
    "100/60", "100/65", "100/70", "110/55", "110/60", "110/65",
    "110/80", "120/60", "80/60", "90/60"
])
urine_albumin = st.selectbox("Urine Albumin", ["None", "Minimal", "Medium"])

# === Prepare Input DataFrame ===
input_df = pd.DataFrame([[0]*len(feature_order)], columns=feature_order)

input_df.at[0, 'Age'] = age
input_df.at[0, 'Weight'] = weight
input_df.at[0, 'Height'] = height
input_df.at[0, 'GestationalAge'] = gest_age
input_df.at[0, 'FetalHeartbeat'] = fhr

def set_if_exists(df, col):
    if col in df.columns:
        df.at[0, col] = 1

# Set categorical/checks
if anemia_min: set_if_exists(input_df, 'Anemia_Minimal')
if urine_sugar: set_if_exists(input_df, 'UrineSugar_Yes')
if jaundice_min: set_if_exists(input_df, 'Jaundice_Minimal')
if hepatitis_b: set_if_exists(input_df, 'HepatitisB_Positive')
if vdrl_pos: set_if_exists(input_df, 'VDRL_Positive')
if fetal_pos_normal: set_if_exists(input_df, 'FetalPosition_Normal')

set_if_exists(input_df, f'Gravida_{gravida}')
set_if_exists(input_df, f'TetanusDose_{tetanus}')
set_if_exists(input_df, f'BloodPressure_{bp}')
if urine_albumin != "None":
    set_if_exists(input_df, f'UrineAlbumin_{urine_albumin}')

# === Predict and Explain ===
if st.button("Predict Risk"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if probability < 0.3:
        risk_level = "✅ Low Risk"
    elif probability < 0.7:
        risk_level = "⚠️ Moderate Risk"
    else:
        risk_level = "🛑 High Risk"

    st.subheader(f"Prediction: {risk_level}")
    st.write(f"**Probability of High Risk:** {probability:.2%}")

    # SHAP explanation
    st.subheader("🔍 Feature Contribution (SHAP Explanation)")
    explainer = shap.Explainer(model, input_df)
    shap_values = explainer(input_df)

    # Text-based top features
    shap_series = pd.Series(shap_values[0].values, index=input_df.columns)
    shap_series = shap_series.sort_values(key=abs, ascending=False)

    st.markdown("### Top Factors Influencing This Prediction:")
    for feature, value in shap_series.head(5).items():
        direction = "increased" if value > 0 else "decreased"
        emoji = "🔺" if value > 0 else "🔻"
        st.write(f"{emoji} **{feature}** — {direction} the risk")

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.bar(shap_values[0], show=False)
    st.pyplot(fig)
