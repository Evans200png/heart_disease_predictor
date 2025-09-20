import streamlit as st
import numpy as np
import joblib

# Load model and scaler

model = joblib.load("heart_rf_model.pkl")
scaler = joblib.load("heart_scaler.pkl")

st.title("❤️ Heart Disease Prediction App")
st.write("Fill in the details below to check the likelihood of heart disease.")

# Input fields (13 features collected)

age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", ("Male", "Female"))
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol (chol)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])

# Resting ECG (mapped to numbers)

restecg_display = st.selectbox("Resting ECG (restecg)", ["Normal", "LV Hypertrophy", "ST-T Abnormality"])
restecg_map = {"Normal": 0, "LV Hypertrophy": 1, "ST-T Abnormality": 2}
restecg = restecg_map[restecg_display]

thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=60, max_value=220, value=150)

# Exang (mapped to numbers)

exang_display = st.selectbox("Exercise Induced Angina (exang)", ["False", "True"])
exang_map = {"False": 0, "True": 1}
exang = exang_map[exang_display]

oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

# Slope (mapped to numbers)

slope_display = st.selectbox("Slope of Peak Exercise ST (slope)", ["Flat", "Upsloping", "Downsloping"])
slope_map = {"Flat": 0, "Upsloping": 1, "Downsloping": 2}
slope = slope_map[slope_display]

ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4])

# Thal (mapped to numbers)

thal_display = st.selectbox("Thalassemia (thal)", ["Normal", "Reversable Defect", "Fixed Defect"])
thal_map = {"Normal": 0, "Reversable Defect": 1, "Fixed Defect": 2}
thal = thal_map[thal_display]

# Convert categorical values

sex = 1 if sex == "Male" else 0

# Collect features (13 provided by user)

user_features = np.array([
	age, sex, cp, trestbps, chol, fbs, restecg, thalach,
	exang, oldpeak, slope, ca, thal
])

# Pad features to match scaler's expected shape (30 features)

if user_features.shape[0] < scaler.n_features_in_:
	user_features = np.pad(
		user_features,
		(0, scaler.n_features_in_ - user_features.shape[0]),
		'constant'
	)

# Reshape for scaler

user_features = user_features.reshape(1, -1)

# Scale and predict

features_scaled = scaler.transform(user_features)

if st.button("Predict"):
	prediction = model.predict(features_scaled)
	if prediction[0] == 1:
		st.error("⚠️ The patient is at risk of heart disease.")
	else:
		st.success("✅ The patient is not at risk of heart disease.")
