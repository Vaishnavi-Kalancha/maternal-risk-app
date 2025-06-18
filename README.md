# 🤰 Maternal Health Risk Prediction Web App

This is a machine learning web application that predicts whether a pregnant individual is at **high risk** or **low risk** based on clinical data collected during antenatal care. The model is trained using **Gradient Boosting Classifier**, and the application is built and deployed with **Streamlit**.

🔗 **Live Demo**: [Click here to try the app](https://maternal-risk-app-fequ2kxspfmfvgkzf67wkk.streamlit.app/)

---

## 🚀 Features

- Predicts maternal risk as **High Risk** or **Low Risk**
- Interactive form to input patient details
- Trained on real-world antenatal care data
- Uses **SHAP** for model explainability
- Deployed online with **Streamlit Cloud**

---

## 📷 Screenshot

*(Add screenshot of your app if you like)*

---

## 🧠 Machine Learning Model

- Algorithm: `GradientBoostingClassifier`
- Evaluation:
  - Accuracy: ~96%
  - F1 Score: ~0.97
- Explainability: SHAP Summary + Bar plots

---

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| **Python** | Programming Language |
| **scikit-learn** | Machine Learning |
| **SHAP** | Model Explainability |
| **Pandas / NumPy** | Data Handling |
| **Streamlit** | Web UI |
| **Git + GitHub** | Version Control & Hosting |

---

## 📦 How to Run Locally

```bash
git clone https://github.com/Vaishnavi-Kalancha/maternal-risk-app.git
cd maternal-risk-app
pip install -r requirements.txt
streamlit run app.py
