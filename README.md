# Predictive Analytics for Diabetes Detection

A complete end-to-end Machine Learning project for predicting diabetes using the Pima Indians Diabetes Dataset.  
This project covers the complete ML lifecycle including data preprocessing, exploratory data analysis, feature engineering, model comparison, hyperparameter tuning, evaluation, prediction, and deployment using Streamlit.

---

# Live Streamlit Application

Deployed Application:

https://predictive-analytics-for-diabetes-detection-gtrn8rceatdnhmr4r6.streamlit.app/

---

# GitHub Repository

Repository Link:

https://github.com/dhanishmohd1136/Predictive-Analytics-for-Diabetes-Detection

---

# Project Overview

The objective of this project is to build a robust machine learning system capable of predicting whether a patient is diabetic based on medical measurements.

This project demonstrates:

- End-to-end machine learning workflow
- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model comparison
- Hyperparameter tuning
- Final model evaluation
- Interactive deployment using Streamlit

---

# Problem Statement

Diabetes is one of the most common chronic diseases worldwide. Early prediction can help in preventive healthcare and timely treatment.

The goal of this project is to classify patients into:

- `0 → Non-Diabetic`
- `1 → Diabetic`

using diagnostic medical measurements such as:
- Glucose
- BMI
- Insulin
- Blood Pressure
- Age
- Pregnancies

---

# Dataset Information

Dataset:
- Pima Indians Diabetes Dataset

Dataset Shape:
- Rows: 768
- Columns: 9

Target Variable:
- Outcome

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

---

# Machine Learning Workflow

## 1. Data Understanding

Performed:
- Dataset inspection
- Shape analysis
- Data type analysis
- Summary statistics
- Missing value analysis
- Duplicate checking

Notebook:
```text
01_data_understanding.ipynb
```

---

## 2. Data Cleaning

Performed:
- Invalid zero-value handling
- Missing value treatment
- Data validation
- Data consistency checking

Notebook:
```text
02_data_cleaning.ipynb
```

---

## 3. Exploratory Data Analysis (EDA)

Performed:
- Target distribution analysis
- Correlation analysis
- Outlier detection
- Distribution analysis
- Skewness and kurtosis analysis
- Feature relationship analysis
- BMI category analysis
- Age-group analysis

Notebook:
```text
03_exploratory_data_analysis.ipynb
```

Generated Visualizations:
- Target distribution
- Correlation heatmap
- Boxplots
- Histograms
- Countplots
- Feature distributions

---

## 4. Feature Engineering

Performed:
- Train-test split
- Feature scaling
- Data transformation
- Processed dataset creation

Notebook:
```text
04_feature_engineering.ipynb
```

Generated Files:
```text
X_train.csv
X_test.csv
X_train_scaled.csv
X_test_scaled.csv
y_train.csv
y_test.csv
scaler.pkl
```

---

## 5. Model Comparison

Models Compared:
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

Notebook:
```text
05_model_comparison.ipynb
```

---

## 6. Hyperparameter Tuning

Technique Used:
- GridSearchCV
- Stratified K-Fold Cross Validation

Parameters Tuned:

### Logistic Regression
- C
- penalty

### KNN
- n_neighbors
- metric
- weights

### SVM
- kernel
- C
- gamma

### Decision Tree
- criterion
- max_depth
- min_samples_split
- min_samples_leaf

Notebook:
```text
06_hyperparameter_tuning.ipynb
```

---

## 7. Final Model Evaluation

Performed:
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Feature Importance
- Classification Report
- Final Metric Analysis

Notebook:
```text
07_final_model_evaluation.ipynb
```

---

## 8. Prediction System

Performed:
- Single prediction
- Multiple predictions
- Probability prediction
- Risk-level interpretation

Notebook:
```text
08_prediction.ipynb
```

---

## 9. Streamlit Deployment

Developed:
- Interactive web application
- Real-time diabetes prediction system
- User-friendly UI

Features:
- Medical input form
- Real-time prediction
- Probability estimation
- Risk categorization

Deployment:
- Streamlit Community Cloud

Main Application File:
```text
app/streamlit_app.py
```

---

# Project Structure

```text
Predictive-Analytics-for-Diabetes-Detection/
│
├── app/
│   └── streamlit_app.py
│
├── config/
│   ├── config.yaml
│   └── model_params.yaml
│
├── data/
│   ├── raw/
│   │   └── diabetes.csv
│   │
│   └── processed/
│       ├── cleaned_data.csv
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── X_train_scaled.csv
│       ├── X_test_scaled.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── models/
│   ├── logistic_regression.pkl
│   ├── knn.pkl
│   ├── svm.pkl
│   ├── decision_tree.pkl
│   ├── logistic_regression_tuned.pkl
│   ├── knn_tuned.pkl
│   ├── svm_tuned.pkl
│   ├── decision_tree_tuned.pkl
│   ├── best_model.pkl
│   ├── best_tuned_model.pkl
│   ├── scaler.pkl
│   └── best_tuned_model_name.txt
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_comparison.ipynb
│   ├── 06_hyperparameter_tuning.ipynb
│   ├── 07_final_model_evaluation.ipynb
│   └── 08_prediction.ipynb
│
├── reports/
│   ├── figures/
│   │   ├── target_distribution.png
│   │   ├── correlation_heatmap.png
│   │   ├── model_f1_comparison.png
│   │   ├── model_roc_auc_comparison.png
│   │   ├── confusion_matrix_best_model.png
│   │   ├── final_roc_curve.png
│   │   └── final_precision_recall_curve.png
│   │
│   ├── model_metrics.csv
│   ├── cross_validation_results.csv
│   ├── hyperparameter_results.csv
│   ├── final_model_metrics.csv
│   ├── final_predictions.csv
│   └── final_model_report.txt
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── hyperparameter_tuning.py
│   ├── prediction.py
│   └── utils.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_training.py
│   └── test_prediction.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Models Used

| Model | Type |
|------|------|
| Logistic Regression | Linear Classification |
| KNN | Distance-Based Learning |
| SVM | Margin-Based Classification |
| Decision Tree | Tree-Based Classification |

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
- streamlit
- joblib

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
- Feature scaling improved KNN and SVM performance.
- Decision Tree captured nonlinear patterns effectively.
- Logistic Regression showed strong ROC-AUC performance.
- Hyperparameter tuning improved model generalization.

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

# Run Streamlit Application

```bash
streamlit run app/streamlit_app.py
```

---

# Future Improvements

Potential future enhancements:
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- SHAP Explainability
- Docker Deployment
- FastAPI Integration
- Cloud Deployment
- MLflow Tracking

---

# Deployment Platforms

The project can be deployed using:
- Streamlit Community Cloud
- Render
- Hugging Face Spaces
- AWS
- Azure

---

# Author

Muhammed Dhanish K

M.Sc. Applied Statistics and Data Analytics

AI/ML Engineer | Data Scientist | Statistician

GitHub:
https://github.com/dhanishmohd1136

---