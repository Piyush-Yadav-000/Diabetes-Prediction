
# 🩺 Diabetes Prediction Using Machine Learning

This project presents a comprehensive approach to early diabetes prediction using various machine learning models, including Logistic Regression, KNN, SVM, Decision Tree, Random Forest, Gradient Boosting, and Neural Networks. Feature engineering, data preprocessing, and ensemble techniques are employed to enhance model performance and robustness, particularly in handling imbalanced datasets.

---

## 📂 Dataset

The dataset used is the **Pima Indians Diabetes Dataset**, which includes the following features:

- Pregnancies  
- Glucose  
- Blood Pressure  
- Skin Thickness  
- Insulin  
- BMI  
- Diabetes Pedigree Function  
- Age  
- Outcome (0 = Non-diabetic, 1 = Diabetic)

---

## ⚙️ Preprocessing Steps

- Handling Missing Values (Imputation with Mean)  
- Outlier Removal (IQR Method)  
- Feature Scaling (Standardization)  
- Feature Engineering:
  - Glucose-to-BMI Ratio  
  - Age-BMI Interaction  
- Feature Selection (SelectKBest with ANOVA F-value)

---

## 🤖 Machine Learning Models Used

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Decision Tree Classifier  
- Random Forest  
- Gradient Boosting  
- Support Vector Machine (SVM)  
- Neural Network (MLP)  
- Voting Classifier (Ensemble Model)

---

## 📊 Model Performance

| Model               | Accuracy |
|---------------------|----------|
| **SVM (Best)**      | **90.23%** |
| Random Forest       | 89.38%   |
| Gradient Boosting   | 86.80%   |
| KNN, Neural Network | 83.36%   |
| Logistic Regression | 79.06%   |

---

## 📈 Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- AUC-ROC Curve  

Cross-validation (5-fold) is used to ensure generalizability.

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/diabetes-prediction-ml.git
cd diabetes-prediction-ml
pip install -r requirements.txt
```

---

## 🚀 Usage

```python
python main.py
```

Replace `main.py` with your script name (e.g., `predict.py`).

---

## 📚 Future Work

- Integration of deep learning architectures ([12])  
- Improved hyperparameter tuning  
- Use of temporal health data  
- Real-time clinical validation ([13], [14])  

---

## 📄 References

[11] Fernandez et al., 2018  
[12] Krishnan, Patel, & Zhang, 2023  
[13] Ting et al., 2019  
[14] Miotto et al., 2018  
