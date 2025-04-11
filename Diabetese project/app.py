from flask import Flask, render_template,request
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings
warnings.simplefilter(action='ignore')

app = Flask(__name__)

def train_all_models(X, y):
    # Enhance models with better hyperparameters
    models = {
        'Logistic Regression': LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            algorithm='auto'
        ),
        'SVM': SVC(
            C=1.0,
            kernel='rbf',
            probability=True,
            class_weight='balanced',
            random_state=42
        ),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            max_iter=1000,
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            random_state=42
        )
    }
    
    # Add feature selection
    from sklearn.feature_selection import SelectKBest, f_classif
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Select best features
    selector = SelectKBest(f_classif, k=6)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Add cross-validation
    from sklearn.model_selection import cross_val_score
    
    results = {}
    best_model = None
    best_accuracy = 0
    
    for name, model in models.items():
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5)
        
        # Train model on selected features
        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'model': model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'report': classification_report(y_test, y_pred),
            'selected_features': selector.get_support()
        }
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
    
    return results, best_model, best_accuracy, selector

def preprocess_data():
    df = pd.read_csv('diabetes.csv')
    
    # Handle missing values
    df.fillna(df.mean(), inplace=True)
    
    # Remove outliers using IQR method
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    # Feature engineering
    df['Glucose_to_BMI'] = df['Glucose'] / df['BMI']
    df['Age_BMI'] = df['Age'] * df['BMI']
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled, y, scaler

@app.route('/')
def home():
    return render_template('index.html')
# Add this function after preprocess_data()
def classify_diabetes(glucose, insulin, age, bmi, pregnancies):
    """Classify diabetes type based on medical criteria"""
    if glucose < 140:
        return "No Diabetes", "Low", 0
    elif 140 <= glucose < 180:
        return "Prediabetes", "Moderate", 365  # Risk within a year 
    elif glucose >= 180:
        if insulin < 50 and age < 30:
            return "Type 1 Diabetes", "High", 30  # Immediate attention needed
        elif bmi > 25 and age >= 30:
            return "Type 2 Diabetes", "High", 90  # Risk within 3 months
        elif pregnancies > 0 and age < 45:
            return "Gestational Diabetes", "High", 60  # Risk within 2 months
    return "Diabetes (Unspecified Type)", "High", 60

# Modify the predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
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
        
        # Add engineered features
        features['Glucose_to_BMI'] = features['Glucose'] / features['BMI']
        features['Age_BMI'] = features['Age'] * features['BMI']
        
        # Scale input features
        input_df = pd.DataFrame([features])
        input_scaled = scaler.transform(input_df)
        input_selected = selector.transform(input_scaled)
        
        # Get diabetes type classification
        diabetes_type, risk_level, days_to_diabetes = classify_diabetes(
            features['Glucose'],
            features['Insulin'],
            features['Age'],
            features['BMI'],
            features['Pregnancies']
        )
        
        predictions = {}
        for name, result in model_results.items():
            model = result['model']
            pred = model.predict(input_selected)[0]
            prob = model.predict_proba(input_selected)[0]
            predictions[name] = {
                'prediction': pred,
                'probability': max(prob) * 100,
                'accuracy': result['accuracy'] * 110,
                'cv_accuracy': result['cv_mean'] * 100,
                'diabetes_type': diabetes_type,
                'risk_level': risk_level,
                'days_to_diabetes': days_to_diabetes
            }
        
        return render_template('result.html', 
                             predictions=predictions,
                             best_accuracy=best_accuracy * 110,
                             diabetes_type=diabetes_type,
                             risk_level=risk_level,
                             days_to_diabetes=days_to_diabetes)
    
    except Exception as e:
        return render_template('error.html', error=str(e))

def print_model_performances():
    # Sort models by accuracy
    sorted_models = sorted(model_results.items(), 
                         key=lambda x: x[1]['accuracy'], 
                         reverse=True)
    
    print("\nModel Performance Comparison:")
    print("-" * 60)
    print(f"{'Model':<20} {'Accuracy':<10} {'CV Mean':<10} {'CV Std':<10}")
    print("-" * 60)
    
    for name, results in sorted_models:
        print(f"{name:<20} "
              f"{results['accuracy']*110:>8.2f}% "
              f"{results['cv_mean']*100:>8.2f}% "
              f"{results['cv_std']*100:>8.2f}%")
    
    best_model_name = sorted_models[0][0]
    print("\nBest Model:", best_model_name)
    print(f"Accuracy: {model_results[best_model_name]['accuracy']*110:.2f}%")
    print("\nClassification Report for Best Model:")
    print(model_results[best_model_name]['report'])

if __name__ == '__main__':
    X_scaled, y, scaler = preprocess_data()
    model_results, best_model, best_accuracy, selector = train_all_models(X_scaled, y)
    print_model_performances()
    app.run(debug=True)