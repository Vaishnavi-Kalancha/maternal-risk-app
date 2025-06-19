import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# === CONFIGURATION ===
st.set_page_config(page_title="Maternal Risk Predictor", layout="centered")

# === CUSTOM CSS STYLING ===
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stButton > button {
            background-color: #FF4B4B;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 20px;
        }
        .stButton > button:hover {
            background-color: #ff3333;
        }
        h1, h2, h3 {
            color: #d6336c;
        }
    </style>
""", unsafe_allow_html=True)

# === SIDEBAR ===
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3220/3220310.png", width=120)
    st.markdown("### 🩺 About")
    st.write("This app uses machine learning to predict maternal health risk.")
    st.markdown("**Author**: Vaishnavi Kalancha")
    st.markdown("[🔗 GitHub](https://github.com/Vaishnavi-Kalancha/maternal-risk-app)")

# === TITLE ===
st.title("🤰 Maternal Health Risk Predictor")
st.subheader("📝 Enter the patient’s clinical details below")

# === LOAD MODEL AND FEATURES ===
model = joblib.load("model.pkl")
feature_order = joblib.load("feature_order.pkl")

# === INPUT COLUMNS ===
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 15, 50, 25)
    weight = st.number_input("Weight (kg)", 30.0, 120.0, 60.0)
    height = st.number_input("Height (cm)", 130.0, 180.0, 155.0)
    gest_age = st.slider("Gestational Age (weeks)", 0, 40, 28)
with col2:
    fhr = st.slider("Fetal Heart Rate (bpm)", 100, 180, 140)
    gravida = st.selectbox("Gravida", ["1st", "2nd", "3rd"])
    tetanus = st.selectbox("Tetanus Dose", ["1st", "2nd", "3rd"])
    bp = st.selectbox("Blood Pressure", [
        "100/60", "100/65", "100/70", "110/55", "110/60", "110/65",
        "110/80", "120/60", "80/60", "90/60"
    ])

# === EXPANDER FOR TESTS ===
with st.expander("🧪 Additional Test Results"):
    anemia_min = st.checkbox("Anemia Minimal")
    urine_sugar = st.checkbox("Urine Sugar Present")
    jaundice_min = st.checkbox("Jaundice Minimal")
    hepatitis_b = st.checkbox("Hepatitis B Positive")
    vdrl_pos = st.checkbox("VDRL Positive")
    fetal_pos_normal = st.checkbox("Fetal Position Normal")
    urine_albumin = st.selectbox("Urine Albumin", ["None", "Minimal", "Medium"])

# === BUILD INPUT DF ===
input_df = pd.DataFrame([[0]*len(feature_order)], columns=feature_order)
input_df.at[0, 'Age'] = age
input_df.at[0, 'Weight'] = weight
input_df.at[0, 'Height'] = height
input_df.at[0, 'GestationalAge'] = gest_age
input_df.at[0, 'FetalHeartbeat'] = fhr

def set_if_exists(df, col):
    if col in df.columns:
        df.at[0, col] = 1

# One-hot mappings
set_if_exists(input_df, f'Gravida_{gravida}')
set_if_exists(input_df, f'TetanusDose_{tetanus}')
set_if_exists(input_df, f'BloodPressure_{bp}')
if urine_albumin != "None":
    set_if_exists(input_df, f'UrineAlbumin_{urine_albumin}')
if anemia_min: set_if_exists(input_df, 'Anemia_Minimal')
if urine_sugar: set_if_exists(input_df, 'UrineSugar_Yes')
if jaundice_min: set_if_exists(input_df, 'Jaundice_Minimal')
if hepatitis_b: set_if_exists(input_df, 'HepatitisB_Positive')
if vdrl_pos: set_if_exists(input_df, 'VDRL_Positive')
if fetal_pos_normal: set_if_exists(input_df, 'FetalPosition_Normal')

# === PREDICTION ===
if st.button("🔍 Predict Risk"):
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

    # === SHAP Explanation ===
    st.subheader("📋 Key Factors Influencing This Prediction")

    try:
        explainer = shap.Explainer(model, input_df)
        shap_values = explainer(input_df)

        shap_array = shap_values.values[0] if hasattr(shap_values, "values") else shap_values[0].values
        shap_series = pd.Series(shap_array, index=input_df.columns).sort_values(key=np.abs, ascending=False)

        top_factors = shap_series.head(5)
        if top_factors.abs().max() == 0:
            st.info("No major features significantly influenced this prediction.")
        else:
            for feature, value in top_factors.items():
                direction = "increased" if value > 0 else "decreased"
                emoji = "🔺" if value > 0 else "🔻"
                st.write(f"{emoji} **{feature}** — {direction} the risk")
    except Exception as e:
        st.error("Could not generate explanation.")
        st.exception(e)

# === FOOTER ===
st.markdown("""
<hr style="margin-top: 2rem;">
<center>
    <small style="color:gray">
        Made with ❤️ by Vaishnavi | Department of CSE | 2025<br>
        For maternal healthcare awareness & safety
    </small>
</center>
""", unsafe_allow_html=True)
