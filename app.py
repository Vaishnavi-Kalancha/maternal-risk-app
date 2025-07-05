import streamlit as st 
import pandas as pd
import numpy as np
import joblib

# Load trained model and expected feature order
model = joblib.load("model.pkl")
feature_order = joblib.load("feature_order.pkl")

st.set_page_config(page_title="Maternal Health Risk Predictor", layout="centered")

# --- CSS Styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    background: #f6f8fc;
    color: #2d3436;
}
h1 { text-align: center; color: #2c3e50; margin-bottom: 0.5em; }
input, select, textarea { border-radius: 6px !important; }
.stButton>button {
    background-color: #6c5ce7; color: white;
    border-radius: 8px; padding: 0.4em 1.5em;
    font-weight: 600; margin: 1em 0;
}
.card {
    background-color: white;
    padding: 1.5em;
    border-radius: 12px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    margin-top: 1.5em;
}
.result-label { font-size: 1.4em; margin-bottom: 0.5em; }
.risk-high { color: #c0392b; font-weight: bold; }
.risk-low { color: #27ae60; font-weight: bold; }
.card hr {
    margin: 1em 0;
    border: none;
    border-top: 1px solid #e0e0e0;
}
.card h4 { margin-bottom: 0.5em; color: #2d3436; }
.card p, .card div { margin: 0.2em 0; }
</style>
""", unsafe_allow_html=True)

st.title("🤰 Maternal Health Risk Predictor")

with st.form("risk_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 15, 50, 25)
        weight = st.number_input("Weight (kg)", 30.0, 120.0, 60.0)
        height = st.number_input("Height (cm)", 130.0, 180.0, 155.0)
        gest_age = st.slider("Gestational Age (weeks)", 0, 40, 28)
        bp = st.selectbox("Blood Pressure", [
            "100/60", "100/65", "100/70", "110/55", "110/60", "110/65",
            "110/80", "120/60", "80/60", "90/60"
        ])
        urine_albumin = st.selectbox("Urine Albumin", ["None", "Minimal", "Medium"])

    with col2:
        fhr = st.slider("Fetal Heart Rate (bpm)", 100, 180, 140)
        gravida = st.selectbox("Gravida", ["1st", "2nd", "3rd"])
        tetanus = st.selectbox("Tetanus Dose", ["2nd", "3rd"])
        anemia_min = st.checkbox("Anemia Minimal")
        urine_sugar = st.checkbox("Urine Sugar Present")
        jaundice_min = st.checkbox("Jaundice Minimal")
        hepatitis_b = st.selectbox("Hepatitis B", ["Positive", "Negative"])
        vdrl_pos = st.checkbox("VDRL Positive")
        fetal_pos_normal = st.checkbox("Fetal Position Normal")

    submit = st.form_submit_button("Predict Risk")

# --- Prediction Logic ---
if submit:
    input_df = pd.DataFrame([[0]*len(feature_order)], columns=feature_order)
    input_df.at[0, 'Age'] = age
    input_df.at[0, 'Weight'] = weight
    input_df.at[0, 'Height'] = height
    input_df.at[0, 'GestationalAge'] = gest_age
    input_df.at[0, 'FetalHeartbeat'] = fhr

    def set_feature(col_name):
        if col_name in input_df.columns:
            input_df.at[0, col_name] = 1

    set_feature(f'Gravida_{gravida}')

    for td in ['2nd', '3rd']:
        col = f'TetanusDose_{td}'
        if col in input_df.columns:
            input_df.at[0, col] = 0
    set_feature(f'TetanusDose_{tetanus}')

    set_feature(f'BloodPressure_{bp}')
    if urine_albumin != "None":
        set_feature(f'UrineAlbumin_{urine_albumin}')
    if anemia_min:
        set_feature('Anemia_Minimal')
    if urine_sugar:
        set_feature('UrineSugar_Yes')
    if jaundice_min:
        set_feature('Jaundice_Minimal')
    if hepatitis_b == "Positive":
        set_feature('HepatitisB_Positive')
    else:
        set_feature('HepatitisB_Negative')
    if vdrl_pos:
        set_feature('VDRL_Positive')
    if fetal_pos_normal:
        set_feature('FetalPosition_Normal')

    # Predict
    prediction = model.predict(input_df)[0]

    # Show result
    if prediction == 1:
        st.markdown(f"""
        <div class="card">
            <div class="risk-high result-label">🛑 High Risk</div>
            <div>This pregnancy is at high risk and needs urgent clinical attention.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="card">
            <div class="risk-low result-label">✅ Low Risk</div>
            <div>No significant risk factors detected in this pregnancy.</div>
        </div>
        """, unsafe_allow_html=True)
