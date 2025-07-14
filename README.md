
 Maternal Risk Prediction System (ML-MRPS)
This project is a lightweight, web-based maternal health risk prediction system built using machine learning. It assists frontline health workers in resource-constrained environments to assess whether a pregnancy is High Risk or Low Risk, based on antenatal clinical data.
Features
Binary Risk Classification: Low Risk / High Risk

Trained with Random Forest (best model after comparison)

Class imbalance handled with SMOTE

Full preprocessing pipeline: cleaning, encoding, standardization

Deployed via Streamlit (local or cloud)

Supports legacy clinical registers from rural health records

Dataset
The data was collected from antenatal care (ANC) registers in rural Bangladesh. It includes fields like:

Age, Weight, Height, Gestational Age

Fetal Heartbeat, Fetal Movement

Anemia, Jaundice

Urine Albumin, Urine Sugar, Tetanus Dose

And more...

Technologies Used
Python

scikit-learn (ML)

pandas / numpy (Data)

SMOTE (Imbalance handling)

Streamlit (Web App)

Joblib (Model serialization)

Git + GitHub (Version control)

Model Training
Run the training script:

bash
Copy
Edit
python train_final_random_forest.py
This will:

Preprocess and balance the dataset

Train a Random Forest model

Save the model and feature order as model.pkl and feature_order.pkl

Launch the Web App
Run the following to start the app:

bash
Copy
Edit
streamlit run app.py
Features:

User-friendly clinical input form

Displays Low Risk or High Risk as output

Designed for use by ASHA workers, nurses, or midwives

Repository Structure
bash
Copy
Edit
├── app.py                      # Streamlit frontend
├── train_final_random_forest.py # Training script
├── model.pkl                   # Trained Random Forest model
├── feature_order.pkl           # Column order for predictions
├── README.md                   # You're reading it!
Future Enhancements
Mobile-first UI for rural health workers

Real-time data entry from IoT or wearables

Alert system for high-risk cases

Regional data expansion to improve generalizability

Author
Kalancha Vaishnavi
I M Tech II Semester | 2025
Dept. of CSE