import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# Load model and feature order
model = joblib.load("model.pkl")
feature_order = joblib.load("feature_order.pkl")

st.set_page_config(page_title="Maternal Risk Predictor", layout="centered")

# --- CSS Styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    background: #f6f8fc;
    color: #2d3436;
}

h1 {
    text-align: center;
    color: #2c3e50;
    margin-bottom: 0.5em;
}

input, select, textarea {
    border-radius: 6px !important;
}

.stButton>button {
    background-color: #6c5ce7;
    color: white;
    border-radius: 8px;
    padding: 0.4em 1.5em;
    font-weight: 600;
    margin: 1em 0;
}

.card {
    background-color: white;
    padding: 1.5em;
    border-radius: 12px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    margin-top: 1.5em;
}

.result-label {
    font-size: 1.4em;
    margin-bottom: 0.5em;
}

.risk-high {
    color: #c0392b;
    font-weight: bold;
}

.risk-moderate {
    color: #d35400;
    font-weight: bold;
}

.risk-low {
    color: #27ae60;
    font-weight: bold;
}

.card hr {
    margin: 1em 0;
    border: none;
    border-top: 1px solid #e0e0e0;
}

.card h4 {
    margin-bottom: 0.5em;
    color: #2d3436;
}

.card p, .card div {
    margin: 0.2em 0;
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("🤰 Maternal Health Risk Predictor")

# --- Form Layout ---
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
        tetanus = st.selectbox("Tetanus Dose", ["1st", "2nd", "3rd"])
        anemia_min = st.checkbox("Anemia Minimal")
        urine_sugar = st.checkbox("Urine Sugar Present")
        jaundice_min = st.checkbox("Jaundice Minimal")
        hepatitis_b = st.checkbox("Hepatitis B Positive")
        vdrl_pos = st.checkbox("VDRL Positive")
        fetal_pos_normal = st.checkbox("Fetal Position Normal")

    submit = st.form_submit_button("Predict Risk")

# --- Prediction Logic ---
if submit:
    # Build input DataFrame
    input_df = pd.DataFrame([[0]*len(feature_order)], columns=feature_order)
    input_df.at[0, 'Age'] = age
    input_df.at[0, 'Weight'] = weight
    input_df.at[0, 'Height'] = height
    input_df.at[0, 'GestationalAge'] = gest_age
    input_df.at[0, 'FetalHeartbeat'] = fhr

    def set_feature(col_name):
        if col_name in input_df.columns:
            input_df.at[0, col_name] = 1

    for cond, name in [
        (anemia_min, 'Anemia_Minimal'),
        (urine_sugar, 'UrineSugar_Yes'),
        (jaundice_min, 'Jaundice_Minimal'),
        (hepatitis_b, 'HepatitisB_Positive'),
        (vdrl_pos, 'VDRL_Positive'),
        (fetal_pos_normal, 'FetalPosition_Normal')
    ]:
        if cond: set_feature(name)

    set_feature(f'Gravida_{gravida}')
    set_feature(f'TetanusDose_{tetanus}')
    set_feature(f'BloodPressure_{bp}')
    if urine_albumin != "None":
        set_feature(f'UrineAlbumin_{urine_albumin}')

    # Prediction
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    # Risk Label
    if prob < 0.3:
        label = "✅ Low Risk"
        style = "risk-low"
    elif prob < 0.7:
        label = "⚠️ Moderate Risk"
        style = "risk-moderate"
    else:
        label = "🛑 High Risk"
        style = "risk-high"

    # --- Show Results in a Single Card ---
    st.markdown(f"""
    <div class="card">
        <div class="{style} result-label">{label}</div>
        <div><strong>Probability of High Risk:</strong> {prob:.2%}</div>
        <hr>
        <h4>📋 Top Factors Influencing This Prediction:</h4>
    """, unsafe_allow_html=True)

    try:
        explainer = shap.Explainer(model, input_df)
        shap_values = explainer(input_df)
        shap_array = shap_values.values[0] if shap_values.values.ndim == 2 else shap_values.values[0][:, 1]
        shap_series = pd.Series(shap_array, index=input_df.columns).sort_values(key=np.abs, ascending=False)

        factors_html = "<ul style='padding-left: 1.2em;'>"
        for feature, value in shap_series.head(5).items():
            direction = "increased" if value > 0 else "decreased"
            emoji = "🔺" if value > 0 else "🔻"
            factors_html += f"<li>{emoji} <strong>{feature}</strong> — {direction} the risk</li>"
        factors_html += "</ul>"
        st.markdown(factors_html, unsafe_allow_html=True)

    except Exception as e:
        st.warning("Could not explain this prediction.")
        st.exception(e)

    st.markdown("</div>", unsafe_allow_html=True)

