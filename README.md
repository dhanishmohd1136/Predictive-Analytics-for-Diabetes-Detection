# Predictive Analytics for Diabetes Detection

A complete end-to-end Machine Learning project for predicting diabetes using the Pima Indians Diabetes Dataset.  
This project covers the full ML pipeline including data preprocessing, exploratory data analysis (EDA), feature engineering, model comparison, hyperparameter tuning, evaluation, prediction, and Streamlit deployment.

---

# Project Overview

The objective of this project is to build a robust machine learning system capable of predicting whether a patient is diabetic based on diagnostic medical measurements.

The project demonstrates:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Machine learning model comparison
- Hyperparameter tuning
- Model evaluation
- Deployment using Streamlit

---

# Problem Statement

Diabetes is a chronic disease that affects millions of people worldwide. Early prediction can help in preventive healthcare and timely treatment.

This project aims to classify patients into:

- `0 → Non-Diabetic`
- `1 → Diabetic`

using clinical measurements such as:
- Glucose
- BMI
- Age
- Insulin
- Blood Pressure
- Pregnancies

---

# Dataset Information

Dataset Used:
- Pima Indians Diabetes Dataset

Features:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

Target:
- Outcome

Dataset Shape:
- Rows: 768
- Columns: 9

---

# Project Structure

```text
Predictive-Analytics-for-Diabetes-Detection/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── diabetes.csv
│   └── processed/
│
├── models/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_models.ipynb
│   ├── 06_hyperparameter_tuning.ipynb
│   ├── 07_model_evaluation.ipynb
│   └── 08_prediction.ipynb
│
├── reports/
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

---

# Machine Learning Workflow

## 1. Data Understanding
- Dataset inspection
- Shape and structure analysis
- Data type analysis
- Statistical summaries

---

## 2. Data Cleaning
- Handling invalid zero values
- Missing value treatment
- Duplicate checking
- Data validation

---

## 3. Exploratory Data Analysis (EDA)
- Target distribution analysis
- Correlation analysis
- Outlier detection
- Distribution analysis
- Feature relationship analysis
- Skewness and kurtosis analysis

---

## 4. Feature Engineering
- Train-test split
- Feature scaling
- Data transformation
- Saving processed datasets

---

## 5. Model Comparison

Models compared:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree

Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Cross Validation Score

---

## 6. Hyperparameter Tuning

Techniques Used:
- GridSearchCV
- Stratified K-Fold Cross Validation

Parameters Tuned:
- SVM → kernel, C, gamma
- KNN → n_neighbors, metric, weights
- Decision Tree → max_depth, criterion
- Logistic Regression → C, penalty

---

## 7. Final Model Evaluation
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Feature Importance
- Final Metrics Analysis

---

## 8. Prediction System
- Single prediction
- Multiple predictions
- Probability estimation
- Risk interpretation

---

## 9. Streamlit Deployment

Interactive web application for diabetes prediction.

Features:
- User-friendly UI
- Real-time prediction
- Probability estimation
- Risk-level interpretation

---

# Models Used

| Model | Type |
|------|------|
| Logistic Regression | Linear Classification |
| KNN | Distance-Based Learning |
| SVM | Margin-Based Learning |
| Decision Tree | Tree-Based Learning |

---

# Technologies Used

## Programming Language
- Python

## Libraries
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib
- streamlit

---

# Installation

## Clone Repository

```bash
git clone https://github.com/dhanishmohd1136/Predictive-Analytics-for-Diabetes-Detection.git
```

---

## Navigate to Project Directory

```bash
cd Predictive-Analytics-for-Diabetes-Detection
```

---

## Create Virtual Environment

```bash
python3 -m venv venv
```

---

## Activate Virtual Environment

### Linux / Mac

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Streamlit App

```bash
streamlit run app/streamlit_app.py
```

---

# Evaluation Metrics

The project uses:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

Medical prediction prioritizes:
- Recall
- F1 Score

to reduce false negatives.

---

# Key Insights

- Dataset contains nonlinear relationships.
- Feature scaling significantly improved KNN and SVM.
- Decision Tree captured nonlinear patterns effectively.
- Logistic Regression showed strong ROC-AUC performance.
- Hyperparameter tuning improved model generalization.

---

# Future Improvements

Potential enhancements:
- XGBoost
- LightGBM
- CatBoost
- SHAP Explainability
- Docker Deployment
- FastAPI Integration
- Cloud Deployment
- MLflow Tracking

---

# Deployment

The project is deployable using:
- Streamlit Community Cloud
- Render
- Hugging Face Spaces

---

# Repository

Repository Link:
https://github.com/dhanishmohd1136/Predictive-Analytics-for-Diabetes-Detection

---

# Author

Muhammed Dhanish K

M.Sc. Applied Statistics and Data Analytics  
AI/ML Engineer | Data Scientist | Statistician

GitHub:
https://github.com/dhanishmohd1136
