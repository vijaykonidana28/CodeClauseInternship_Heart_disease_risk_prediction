# ❤️ Heart Disease Risk Assessment App

This Streamlit web application helps assess the risk of heart disease based on user input using a machine learning model (Random Forest Classifier). Users can upload their dataset, view data, and receive a prediction on heart disease risk.

## 🔍 Project Overview

- **Model Used**: Random Forest Classifier
- **Language**: Python
- **Framework**: Streamlit
- **Purpose**: Predict the risk of heart disease based on medical attributes

## 📁 Features

- Upload your custom dataset (CSV)
- View dataset preview
- Enter patient data via sidebar widgets
- Get prediction (Low Risk / High Risk)
- Built-in data preprocessing (One-Hot Encoding for categorical variables)

##  📊 Technologies Used
	•	Python
	•	Streamlit
	•	Scikit-learn
	•	Pandas
	•	RandomForestClassifier

🧠 Model Training

The model is trained on a dataset (heart.csv) with various health indicators. Categorical features are handled with one-hot encoding.


## 🛠️ Setup Instructions

### 1. Clone the Repository

git clone https://github.com/your-username/CodeClauseInternship_Heart_disease_risk_prediction.git
cd CodeClauseInternship_Heart_disease_risk_prediction

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


🧾 Requirements

Make sure you have Python installed (preferably 3.9+). Required libraries include:
	•	streamlit
	•	pandas
	•	scikit-learn
