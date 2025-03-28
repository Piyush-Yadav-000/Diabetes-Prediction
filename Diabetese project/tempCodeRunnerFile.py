from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

app = Flask(__name__)

# Load and train the model
df = pd.read_csv('diabetes.csv')
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

def classify_diabetes(glucose, insulin, age, bmi):
    if glucose < 140:
        return "No Diabetes", "Low", 0
    elif glucose <90:
        return "Low Glucose", "low", 0
    elif 140 <= glucose < 180:
        return "Prediabetes", "Moderate", 365
    elif glucose >= 180:
        if insulin < 50 and age < 30:
            return "Type 1 Diabetes", "High", 90
        elif bmi > 25 and age >= 30:
            return "Type 2 Diabetes", "High", 180
        elif age < 50:
            return "Gestational Diabetes", "High", 120
    return "Diabetes (Unknown Type)", "High", 180

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = {
        'Pregnancies': float(request.form['pregnancies']),
        'Glucose': float(request.form['glucose']),
        'BloodPressure': float(request.form['bloodpressure']),
        'SkinThickness': float(request.form['skinthickness']),
        'Insulin': float(request.form['insulin']),
        'BMI': float(request.form['bmi']),
        'DiabetesPedigreeFunction': float(request.form['dpf']),
        'Age': float(request.form['age'])
    }
    
    input_df = pd.DataFrame([features])
    prediction = model.predict(input_df)[0]
    
    diabetes_type, risk_level, days_to_diabetes = classify_diabetes(
        features['Glucose'], features['Insulin'], features['Age'], features['BMI']
    )
    
    return render_template('result.html', 
                         prediction=prediction,
                         diabetes_type=diabetes_type,
                         risk_level=risk_level,
                         days_to_diabetes=days_to_diabetes)

if __name__ == '__main__':
    app.run(debug=True)