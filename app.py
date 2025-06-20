import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# === Load model and feature order ===
model = joblib.load("model.pkl")
feature_order = joblib.load("feature_order.pkl")

st.set_page_config(page_title="Maternal Risk Predictor", layout="wide")

# === Custom Styling ===
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #f5f7fa, #c3cfe2);
    }
    .reportview-container {
        background: transparent;
    }
    .main-container {
        background-color: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        max-width: 1000px;
        margin: auto;
    }
    .stButton > button {
        background-color: #6c5ce7;
        color: white;
        font-weight: bold;
    }
    .risk-box {
        padding: 1em;
        border-radius: 10px;
        margin-top: 1em;
        font-size: 18px;
    }
    .low-risk {
        background-color: #dff9fb;
        color: #00b894;
    }
    .moderate-risk {
        background-color: #ffeaa7;
        color: #d35400;
    }
    .high-risk {
        background-color: #fab1a0;
        color: #c0392b;
    }
    </style>
""", unsafe_allow_html=True)

# === App Layout ===
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.title("🤰 Maternal Health Risk Predictor")
st.markdown("### 📝 Enter the patient’s clinical details below")

# === Sidebar About Section ===
with st.sidebar:
    st.image("https://img.icons8.com/ios-filled/100/steam-bun.png", width=100)
    st.markdown("## ℹ️ About")
    st.markdown("""
    This app uses a trained machine learning model to **predict maternal health risk** based on clinical indicators.

    **Author**: Vaishnavi Kalancha  
    [📂 GitHub Repo](https://github.com/Vaishnavi-Kalancha/maternal-risk-app)
    """)

# === Input Fields ===
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=15, max_value=50, value=25)
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

# === Prepare Input Data ===
input_df = pd.DataFrame([[0]*len(feature_order)], columns=feature_order)
input_df.at[0, 'Age'] = age
input_df.at[0, 'Weight'] = weight
input_df.at[0, 'Height'] = height
input_df.at[0, 'GestationalAge'] = gest_age
input_df.at[0, 'FetalHeartbeat'] = fhr

def set_if_exists(df, col):
    if col in df.columns:
        df.at[0, col] = 1

for cond, col_name in [
    (anemia_min, 'Anemia_Minimal'), (urine_sugar, 'UrineSugar_Yes'),
    (jaundice_min, 'Jaundice_Minimal'), (hepatitis_b, 'HepatitisB_Positive'),
    (vdrl_pos, 'VDRL_Positive'), (fetal_pos_normal, 'FetalPosition_Normal')
]:
    if cond: set_if_exists(input_df, col_name)

set_if_exists(input_df, f'Gravida_{gravida}')
set_if_exists(input_df, f'TetanusDose_{tetanus}')
set_if_exists(input_df, f'BloodPressure_{bp}')
if urine_albumin != "None":
    set_if_exists(input_df, f'UrineAlbumin_{urine_albumin}')

# === Prediction ===
if st.button("Predict Risk"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if probability < 0.3:
        risk = "✅ Low Risk"
        risk_class = "low-risk"
    elif probability < 0.7:
        risk = "⚠️ Moderate Risk"
        risk_class = "moderate-risk"
    else:
        risk = "🛑 High Risk"
        risk_class = "high-risk"

    st.markdown(f'<div class="risk-box {risk_class}">Prediction: {risk}</div>', unsafe_allow_html=True)
    st.write(f"**Probability of High Risk:** {probability:.2%}")

    # SHAP Explanation
    st.subheader("📋 Key Factors Influencing This Prediction")
    try:
        explainer = shap.Explainer(model, input_df)
        shap_values = explainer(input_df)

        if shap_values.values.ndim == 3:
            shap_array = shap_values.values[0][:, 1]
        else:
            shap_array = shap_values.values[0]

        shap_series = pd.Series(shap_array, index=input_df.columns).sort_values(key=np.abs, ascending=False)
        for feature, value in shap_series.head(5).items():
            direction = "increased" if value > 0 else "decreased"
            emoji = "🔺" if value > 0 else "🔻"
            st.write(f"{emoji} **{feature}** — {direction} the risk")
    except Exception as e:
        st.error("Could not generate explanation.")
        st.exception(e)

# === Info Section ===
with st.expander("ℹ️ How does this app work?"):
    st.markdown("""
    This app uses a **Random Forest** model trained on clinical data  
    to assess maternal risk levels based on features like:
    - **Age**
    - **Blood Pressure**
    - **Gestational Age**
    - **Anemia, Jaundice, Hepatitis**  
    The model uses **SHAP** explainability to help understand feature influence.
    """)

st.markdown("</div>", unsafe_allow_html=True)
