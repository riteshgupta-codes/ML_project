
# 🎓 End-to-End Student Performance Prediction System

An end-to-end Machine Learning project that predicts **Student Math Scores** using a complete ML pipeline architecture including:

* Data Ingestion
* Data Transformation
* Model Training with Hyperparameter Tuning
* Model Evaluation
* Prediction Pipeline
* Exception Handling
* Logging System
* Flask Deployment Ready

---

## 🚀 Project Overview

This project predicts student math scores based on:

* Gender
* Race / Ethnicity
* Parental Level of Education
* Lunch Type
* Test Preparation Course
* Reading Score
* Writing Score

The system follows a production-style ML architecture using modular components.

---

## 🏗️ Project Architecture

```
src/
│
├── components/
│   ├── data_ingestion.py
│   ├── data_transformation.py
│   ├── model_trainer.py
│
├── pipeline/
│   └── predict_pipeline.py
│
├── exception.py
├── logger.py
├── utils.py
│
artifacts/
│   ├── train.csv
│   ├── test.csv
│   ├── model.pkl
│   ├── preprocessor.pkl
│
app.py
requirements.txt
README.md
```

---

## ⚙️ Machine Learning Workflow

### 1️⃣ Data Ingestion

* Reads dataset
* Performs train-test split
* Stores raw and processed data in `artifacts/`

### 2️⃣ Data Transformation

* Numerical Pipeline:

  * Median Imputation
  * Standard Scaling
* Categorical Pipeline:

  * Most Frequent Imputation
  * One Hot Encoding
  * Scaling
* Saves preprocessor as `preprocessor.pkl`

### 3️⃣ Model Training

Models evaluated using GridSearchCV:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor
* AdaBoost Regressor
* XGBoost Regressor
* CatBoost Regressor

Best model selected based on **R² Score**.

Saved as:

```
artifacts/model.pkl
```

---

## 📊 Evaluation Metric

* R² Score
* GridSearchCV (Hyperparameter tuning)
* Train/Test validation

---

## 🔮 Prediction Pipeline

The prediction system:

1. Loads saved model
2. Loads preprocessing object
3. Applies same transformation
4. Returns predicted math score

This ensures **training-serving consistency**.

---

## 🧱 Engineering Features

✔ Modular Architecture
✔ Production-style pipeline
✔ Custom Exception Handling
✔ Logging System
✔ Artifact management
✔ Hyperparameter tuning
✔ Clean separation of concerns

---

## ⚙️ Installation & Running Locally

### 1️⃣ Clone Repository

```bash
git clone https://github.com/riteshgupta-codes/ML_project.git
cd ML_project
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

(Windows)

```
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run Training Pipeline

```bash
python src/components/data_ingestion.py
```

This will:

* Ingest data
* Transform data
* Train best model
* Save artifacts

---

## 🌐 Running Flask App

```bash
python app.py
```

Production server:

```bash
gunicorn app:app
```

---

## 🧠 Key Learning Outcomes

* End-to-End ML Pipeline Design
* Model Selection & Hyperparameter Tuning
* Production-ready Architecture
* Feature Engineering Pipelines
* Exception & Logging Handling
* Deployment Preparation

---

## 🔥 Future Improvements

* Dockerization
* CI/CD Integration
* Model Monitoring
* Cloud Deployment (AWS/GCP)
* REST API versioning
* MLflow Integration

---

## 👨‍💻 Author

**Ritesh Gupta**
Aspiring AI Engineer
GitHub: [https://github.com/riteshgupta-codes]

---


