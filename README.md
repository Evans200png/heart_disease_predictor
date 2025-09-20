# ❤️ Heart Disease Prediction App

A machine learning project that predicts the risk of heart disease based on patient medical data.
The model is trained on the **UCI Heart Disease dataset** and deployed as an interactive web app using **Streamlit**.

---

## 🚀 Features

* Input patient details such as:

  * Age, Sex
  * Chest pain type (cp)
  * Resting blood pressure (trestbps)
  * Cholesterol (chol)
  * Fasting blood sugar (fbs)
  * Resting ECG results (restecg)
  * Maximum heart rate achieved (thalach)
  * Exercise induced angina (exang)
  * ST depression induced by exercise (oldpeak)
  * Slope of peak exercise ST segment (slope)
  * Number of major vessels colored by fluoroscopy (ca)
  * Thalassemia status (thal)

* Predicts whether the patient is **at risk of heart disease**.

* Uses a trained **RandomForestClassifier** (saved as `model.pkl`).

* Input features are scaled with a **StandardScaler** (`scaler.pkl`).

* Simple, responsive UI built with Streamlit.

---

## 📂 Project Structure

```
heart_disease_predictor/
│── app.py               # Streamlit app
│── model.pkl            # Trained ML model
│── scaler.pkl           # Trained scaler
│── requirements.txt     # Dependencies
│── README.md            # Project documentation
└── data/                # (Optional) dataset or preprocessing files
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/heart_disease_predictor.git
cd heart_disease_predictor
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it:

* Windows:

  ```bash
  venv\Scripts\activate
  ```
* Linux / Mac:

  ```bash
  source venv/bin/activate
  ```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the App

Once setup is complete, run:

```bash
streamlit run app.py
```

This will launch the app in your browser at **[http://localhost:8501/](http://localhost:8501/)**.

---

## 🧠 Model Training (Optional)

If you want to retrain the model:

1. Load the dataset (e.g., UCI Heart Disease dataset).
2. Preprocess features:

   * Encode categorical variables (`cp`, `restecg`, `slope`, `thal`, etc.).
   * Scale numerical features using `StandardScaler`.
3. Train a classifier (RandomForest, DecisionTree, Logistic Regression, etc.).
4. Save the trained model and scaler:

   ```python
   import joblib
   joblib.dump(model, "model.pkl")
   joblib.dump(scaler, "scaler.pkl")
   ```

---

## 🖥️ Example Inputs

* Age: `55`
* Sex: `Male`
* cp: `3` (typical angina)
* trestbps: `140`
* chol: `250`
* fbs: `1` (true)
* restecg: `2` (ST-T abnormality)
* thalach: `120`
* exang: `1` (true)
* oldpeak: `2.3`
* slope: `0` (flat)
* ca: `2`
* thal: `2` (reversible defect)

👉 Expected outcome: **High risk of heart disease**

---

## 📦 Requirements

* Python 3.9+
* Streamlit
* scikit-learn
* numpy
* joblib

(Install via `pip install -r requirements.txt`)

---

## 📸 Demo Screenshot

*(Check it out here: ['https://evans200png-heart-disease-predictor-app-ttija1.streamlit.app/'])*

---

## 🤝 Contributing

Pull requests are welcome! If you’d like to improve the model, add new features, or enhance the UI, feel free to fork and submit changes.

---

## Author

Evanson Kimani 💼Linkedin ('https://ke.linkedin.com/in/evanson-kimani-2770b4331')  ✉️Email ('kimanievansonk@gmail.com')

---

