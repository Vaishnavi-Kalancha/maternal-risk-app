import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("model.pkl")

st.set_page_config(page_title="Maternal Risk Predictor", layout="centered")
st.title("🤰 Maternal Health Risk Predictor")
st.write("Enter the patient’s clinical details below to predict the risk level.")

# === Collect inputs from user ===
age = st.number_input("Age", min_value=15, max_value=50, value=25)
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=120.0, value=60.0)
height = st.number_input("Height (cm)", min_value=130.0, max_value=180.0, value=155.0)
gest_age = st.slider("Gestational Age (weeks)", 0, 40, 28)
fhr = st.slider("Fetal Heart Rate (bpm)", 100, 180, 140)

# Optional categorical fields (checkbox/selectbox)
anemia_min = st.checkbox("Anemia Minimal")
urine_sugar = st.checkbox("Urine Sugar Present")
jaundice_min = st.checkbox("Jaundice Minimal")
hepatitis_b = st.checkbox("Hepatitis B Positive")
vdrl_pos = st.checkbox("VDRL Positive")
fetal_pos_normal = st.checkbox("Fetal Position Normal")

# Select dropdowns
gravida = st.selectbox("Gravida", ["1st", "2nd", "3rd"])
tetanus = st.selectbox("Tetanus Dose", ["1st", "2nd", "3rd"])
bp = st.selectbox("Blood Pressure", [
    "100/60", "100/65", "100/70", "110/55", "110/60", "110/65",
    "110/80", "120/60", "80/60", "90/60"
])
urine_albumin = st.selectbox("Urine Albumin", ["None", "Minimal", "Medium"])

# === Build full feature set expected by the model ===
feature_list = [
    'Age', 'Weight', 'Height', 'GestationalAge', 'FetalHeartbeat',
    'Anemia_Minimal', 'BloodPressure_100/60', 'BloodPressure_100/65',
    'BloodPressure_100/70', 'BloodPressure_110/55', 'BloodPressure_110/60',
    'BloodPressure_110/65', 'BloodPressure_110/80', 'BloodPressure_120/60',
    'BloodPressure_80/60', 'BloodPressure_90/60', 'FetalPosition_Normal',
    'Gravida_2nd', 'Gravida_3rd', 'HepatitisB_Positive',
    'Jaundice_Minimal', 'TetanusDose_2nd', 'TetanusDose_3rd',
    'UrineAlbumin_Medium', 'UrineAlbumin_Minimal', 'UrineSugar_Yes',
    'VDRL_Positive'
]

# Start with all zeros
input_df = pd.DataFrame(data=np.zeros((1, len(feature_list))), columns=feature_list)

# Set user inputs
input_df.at[0, 'Age'] = age
input_df.at[0, 'Weight'] = weight
input_df.at[0, 'Height'] = height
input_df.at[0, 'GestationalAge'] = gest_age
input_df.at[0, 'FetalHeartbeat'] = fhr

# Set checkboxes
input_df.at[0, 'Anemia_Minimal'] = 1 if anemia_min else 0
input_df.at[0, 'UrineSugar_Yes'] = 1 if urine_sugar else 0
input_df.at[0, 'Jaundice_Minimal'] = 1 if jaundice_min else 0
input_df.at[0, 'HepatitisB_Positive'] = 1 if hepatitis_b else 0
input_df.at[0, 'VDRL_Positive'] = 1 if vdrl_pos else 0
input_df.at[0, 'FetalPosition_Normal'] = 1 if fetal_pos_normal else 0

# Set select options
input_df.at[0, f'Gravida_{gravida}'] = 1 if f'Gravida_{gravida}' in input_df.columns else 0
input_df.at[0, f'TetanusDose_{tetanus}'] = 1 if f'TetanusDose_{tetanus}' in input_df.columns else 0
input_df.at[0, f'BloodPressure_{bp}'] = 1 if f'BloodPressure_{bp}' in input_df.columns else 0
if urine_albumin != "None":
    input_df.at[0, f'UrineAlbumin_{urine_albumin}'] = 1

# === Predict and show result ===
if st.button("Predict Risk"):
    prediction = model.predict(input_df)[0]
    risk = "🛑 High Risk" if prediction == 1 else "✅ Low Risk"
    st.success(f"Prediction: {risk}")
