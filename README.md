# 🤰 Maternal Health Risk Prediction Web App

This is a machine learning web application that predicts whether a pregnant individual is at **Low**, **Moderate**, or **High Risk** based on clinical data collected during antenatal care. The model is trained using a **Random Forest Classifier**, and the application is built and deployed with **Streamlit**.

🔗 **Live Demo**: [Click here to try the app](https://maternal-risk-app-fequ2kxspfmfvgkzf67wkk.streamlit.app/)

---

## 🚀 Features

- Predicts maternal risk as **Low**, **Moderate**, or **High**
- Interactive form to input clinical characteristics
- Trained on real-world antenatal care data
- **SHAP explainability** to highlight key contributing factors
- Clean and responsive UI with custom CSS
- Fully deployed online with **Streamlit Cloud**

---

## 📷 Screenshot

*(You can add a screenshot like `shap_summary_bar.png` or the app UI if desired)*

---

## 🧠 Machine Learning Model

- Algorithm: `RandomForestClassifier`
- Dataset: Preprocessed and balanced with **SMOTE**
- Evaluation (70:30 split):
  - Accuracy: **94.67%**
  - F1 Score: **95.96%**
  - ROC AUC: **99.20%**
- Explainability: Integrated SHAP plots per prediction

---

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| **Python** | Core Programming Language |
| **scikit-learn** | Machine Learning Algorithms |
| **SHAP** | Explainable AI |
| **Pandas, NumPy** | Data Processing |
| **Streamlit** | Web UI Framework |
| **Joblib** | Model Serialization |
| **Git + GitHub** | Version Control & Hosting |

---

## 📦 How to Run Locally

```bash
git clone https://github.com/Vaishnavi-Kalancha/maternal-risk-app.git
cd maternal-risk-app
pip install -r requirements.txt
streamlit run app.py
