import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ----------- SECTION 3: Streamlit UI ----------- #
st.set_page_config(page_title="❤️ Heart Disease Risk Assessment", layout="centered")
st.title("❤️ Heart Disease Risk Assessment")
st.markdown("Enter your health details below:")
# ----------- SECTION 1: Train Model and Save if Not Exists ----------- #
@st.cache_resource
def train_and_save_model():
    if not os.path.exists("model.pkl") or not os.path.exists("features.pkl"):
        st.info("Training model...")

        # Sample dataset (You should upload your actual dataset)
        df = pd.read_csv("heart.csv")  # Make sure this file exists in your project

        # Preprocessing
        df = pd.get_dummies(df, drop_first=True)
        X = df.drop("HeartDisease", axis=1)
        y = df["HeartDisease"]

        # Train model
        model = RandomForestClassifier()
        model.fit(X, y)

        # Save model and features
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open("features.pkl", "wb") as f:
            pickle.dump(X.columns.tolist(), f)

train_and_save_model()

# ----------- SECTION 2: Load Model and Feature Columns ----------- #
model = pickle.load(open("model.pkl", "rb"))
feature_names = pickle.load(open("features.pkl", "rb"))



# ----------- SECTION 4: User Input ----------- #
def get_user_input():
    age = st.slider("Age", 20, 80, 45)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
    trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
    chol = st.slider("Cholesterol", 100, 400, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
    restecg = st.selectbox("Rest ECG", ["Normal", "ST", "LVH"])
    thalach = st.slider("Max Heart Rate Achieved", 60, 200, 150)
    exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])

    data = {
        "Age": age,
        "Sex": 1 if sex == "Male" else 0,
        "ChestPainType": cp,
        "RestingBP": trestbps,
        "Cholesterol": chol,
        "FastingBS": 1 if fbs == "Yes" else 0,
        "RestingECG": restecg,
        "MaxHR": thalach,
        "ExerciseAngina": 1 if exang == "Yes" else 0,
    }
    return pd.DataFrame([data])

input_df = get_user_input()

# ----------- SECTION 5: Preprocess Input to Match Training ----------- #
input_df = pd.get_dummies(input_df)

# Align with training columns
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# ----------- SECTION 6: Predict and Display Result ----------- #
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease!")
    else:
        st.success("✅ Low Risk of Heart Disease.")