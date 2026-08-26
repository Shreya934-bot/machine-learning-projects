<div align="center">

# 🔥 Boiler Intelligence

### AI-Powered Predictive Maintenance System for Industrial Boilers

**Predict failure risk. Estimate remaining useful time. Enable proactive maintenance.**

[![Live App](https://img.shields.io/badge/🚀_Live_App-Try_Boiler_Intelligence-success?style=for-the-badge)](https://boiler-intelligence.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45.1-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit Learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br>

## 🚀 [Launch the Live Application →](https://boiler-intelligence.streamlit.app/)

**Developed by [Shreya Verma](https://github.com/Shreya934-bot)**

*ML Engineer • AI Enthusiast • Developer*

</div>

---

## 📌 Overview

**Boiler Intelligence** is an end-to-end machine learning application designed to support **predictive maintenance for industrial boiler systems**.

Instead of relying entirely on reactive or fixed-schedule maintenance, the system analyzes operational sensor data and uses trained machine learning models to answer two critical maintenance questions:

> **⚠️ Is the boiler likely to fail within the next 7 days?**

and

> **⏳ How many days remain before a potential failure?**

The application transforms raw boiler operating data into actionable maintenance intelligence through a complete machine learning workflow consisting of:

- Data validation and preprocessing
- Failure-risk classification
- Remaining-time regression
- Risk categorization
- Batch prediction
- Live single-record prediction
- Interactive analytics
- Maintenance prioritization
- Downloadable prediction reports

The final solution is deployed as an interactive web application using **Streamlit Community Cloud**.

---

# 🌐 Live Application

<div align="center">

### 🔥 Experience Boiler Intelligence in Action

## [🚀 OPEN LIVE APP](https://boiler-intelligence.streamlit.app/)

**https://boiler-intelligence.streamlit.app/**

</div>

The deployed application provides five major modules:

| Module | Purpose |
|---|---|
| 🏠 **Overview** | System overview and prediction workflow |
| 🔮 **Batch Prediction** | Upload and analyze multiple boiler records |
| ⚡ **Single Prediction** | Perform real-time prediction from manual sensor inputs |
| 📊 **Analytics** | Explore risk distributions and prediction insights |
| 🤖 **Model Insights** | Understand the ML pipeline and feature contract |

---

# 🎯 Problem Statement

Industrial boilers operate under continuously changing conditions involving:

- Temperature
- Pressure
- Vibration
- Fuel consumption
- Water levels
- Ambient conditions
- Operating hours since maintenance

Unexpected equipment failures can result in:

- ⚠️ Unplanned downtime
- 💰 High maintenance costs
- 📉 Production losses
- 🏭 Operational disruptions
- 🔧 Emergency repairs
- 🛡️ Potential safety concerns

Traditional maintenance strategies are often based on either:

### Reactive Maintenance

> Fix the equipment after it fails.

This can result in expensive unplanned downtime.

### Preventive Maintenance

> Service equipment at predefined intervals.

This can lead to unnecessary maintenance because components may still be healthy.

### Predictive Maintenance

> Use historical and operational data to predict equipment failure before it occurs.

This project focuses on the third approach.

**Boiler Intelligence uses machine learning to help identify risk earlier and support proactive maintenance decisions.**

---

# 🧠 Machine Learning Approach

The project uses a **dual-model prediction pipeline**.

Instead of solving the problem using only one prediction, two different machine learning tasks are performed.

---

## 1️⃣ Failure Classification

The classification model predicts whether a boiler is expected to fail within a defined time window.

### Prediction Target

```text
failure_within_7_days
````

### Output

```text
Yes / No
```

When available from the model, the application also displays:

```text
Failure Probability (%)
```

This provides an additional measure for prioritizing maintenance records.

---

## 2️⃣ Days-to-Failure Regression

The regression model estimates the remaining number of days before a potential failure.

### Prediction Target

```text
days_to_failure
```

### Output

```text
Predicted Days to Failure
```

The application ensures negative predictions are not displayed by applying:

```python
max(predicted_days, 0)
```

---

## 🔗 Combined Prediction Pipeline

The two predictions are combined to provide a more complete operational view.

```text
                  BOILER SENSOR DATA
                           │
                           ▼
                ┌─────────────────────┐
                │ Data Validation     │
                │ & Feature Selection │
                └──────────┬──────────┘
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
        ┌────────────────┐   ┌──────────────────┐
        │ Classification │   │ Feature Scaling  │
        │     Model      │   └────────┬─────────┘
        └───────┬────────┘            │
                │                     ▼
                ▼             ┌──────────────────┐
        Failure Probability   │ Regression Model │
        Fail within 7 days    └────────┬─────────┘
                                      │
                                      ▼
                           Predicted Days to Failure
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
                   Risk Assessment         Maintenance Priority
```

---

# 🔥 Risk Intelligence System

The application converts the regression prediction into three operational risk categories.

| Risk Level    | Predicted Days to Failure | Operational Interpretation       |
| ------------- | ------------------------- | -------------------------------- |
| 🔴 **High**   | ≤ 3 days                  | Immediate review recommended     |
| 🟠 **Medium** | > 3 and ≤ 7 days          | Maintenance planning recommended |
| 🟢 **Low**    | > 7 days                  | Continue routine monitoring      |

This helps transform a raw regression value into a more understandable maintenance decision.

For example:

```text
Prediction: 1.8 days
→ 🔴 HIGH RISK
→ Immediate attention
```

```text
Prediction: 5.2 days
→ 🟠 MEDIUM RISK
→ Schedule maintenance planning
```

```text
Prediction: 14.6 days
→ 🟢 LOW RISK
→ Continue routine monitoring
```

---

# 📊 Input Features

The deployed models expect the following seven boiler operating features.

|  # | Feature                   | Description                             |
| -: | ------------------------- | --------------------------------------- |
|  1 | `temperature_C`           | Boiler operating temperature in Celsius |
|  2 | `pressure_psi`            | Boiler operating pressure               |
|  3 | `vibration_level`         | Equipment vibration measurement         |
|  4 | `run_hours_since_service` | Operating hours since the last service  |
|  5 | `fuel_flow_rate`          | Fuel consumption or flow rate           |
|  6 | `water_level_percent`     | Boiler water level percentage           |
|  7 | `ambient_temp_C`          | Ambient environmental temperature       |

These features form the model input contract and are validated before prediction.

---

# ✨ Application Features

## 🏠 1. Overview Dashboard

The landing page provides a high-level view of the predictive maintenance system.

### Highlights

* Industrial AI dashboard interface
* Prediction workflow visualization
* Input feature information
* ML system overview
* Risk tier explanation
* Latest prediction run information
* Developer branding

---

## 🔮 2. Batch Prediction

Users can upload a CSV containing multiple boiler sensor records.

The application automatically performs:

```text
CSV Upload
    ↓
Encoding Detection
    ↓
Column Validation
    ↓
Numeric Data Validation
    ↓
Feature Selection
    ↓
Classification Prediction
    ↓
Feature Scaling
    ↓
Regression Prediction
    ↓
Risk Categorization
    ↓
Maintenance Prioritization
```

### Robust CSV Handling

The application attempts to handle common file encodings, including:

```text
UTF-8 with BOM
UTF-8
Windows-1252
Latin-1
```

This helps avoid common dataset upload problems caused by encoding differences.

### Validation Checks

The application checks for:

* Missing required columns
* Missing values
* Non-numeric values
* Empty datasets
* Unexpected encoding issues

Optional target columns from training-style datasets are safely ignored:

```text
days_to_failure
failure_within_7_days
```

This means the app can focus on the actual sensor features required for inference.

---

## ⚡ 3. Single Prediction

The Single Prediction module simulates a live sensor assessment.

Users can manually enter values for all seven features:

```text
Temperature
Pressure
Vibration
Run Hours Since Service
Fuel Flow Rate
Water Level
Ambient Temperature
```

The application then produces:

* 🔴🟠🟢 Risk Level
* ⏳ Estimated Days Remaining
* 📈 Failure Probability
* ⚠️ Failure-within-7-days prediction

This creates an interactive workflow for evaluating an individual boiler operating condition.

---

## 📊 4. Analytics Dashboard

The analytics module provides interactive visualizations for the latest prediction run.

### Available Insights

#### Risk Composition

Visualizes how the uploaded records are distributed across:

* High Risk
* Medium Risk
* Low Risk

#### Days-to-Failure Distribution

Shows the distribution of predicted remaining time across the analyzed dataset.

#### Sensor Relationship Explorer

Users can interactively select sensor variables for:

* X-axis
* Y-axis

The visualization is colored according to risk level, making it easier to explore relationships between operating conditions and predicted risk.

#### Maintenance Queue

Records are prioritized using predicted:

1. Days to failure
2. Failure probability

This creates an actionable list for maintenance review.

---

## 🤖 5. Model Insights

The Model Insights page explains the technical prediction pipeline used by the application.

It provides visibility into:

* Classification model
* Regression model
* Feature preprocessing
* Input feature contract
* Risk policy
* Model metadata

The system uses:

### Classification Pipeline

```text
Boiler Features
      ↓
Random Forest Classifier
      ↓
Failure Within 7 Days
+
Failure Probability
```

### Regression Pipeline

```text
Boiler Features
      ↓
StandardScaler
      ↓
Random Forest Regressor
      ↓
Predicted Days to Failure
```

---

# 🏗️ System Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                        USER                                  │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                   STREAMLIT WEB APPLICATION                   │
│                                                              │
│  Overview │ Batch Prediction │ Single Prediction │ Analytics │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                    DATA PROCESSING LAYER                     │
│                                                              │
│  CSV Reading                                                 │
│  Encoding Handling                                           │
│  Column Validation                                           │
│  Numeric Conversion                                          │
│  Feature Selection                                           │
└────────────────────────────┬─────────────────────────────────┘
                             │
                  ┌──────────┴──────────┐
                  ▼                     ▼
┌─────────────────────────┐   ┌──────────────────────────────┐
│ Random Forest           │   │ StandardScaler               │
│ Classifier              │   │              ↓               │
│                         │   │ Random Forest Regressor      │
└────────────┬────────────┘   └───────────────┬──────────────┘
             │                                │
             ▼                                ▼
    Failure Classification            Days-to-Failure Estimate
             │                                │
             └───────────────┬────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                    RISK INTELLIGENCE LAYER                   │
│                                                              │
│      🔴 High        🟠 Medium         🟢 Low                  │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                    ACTIONABLE OUTPUTS                        │
│                                                              │
│  KPI Cards • Charts • Prediction Table • Maintenance Queue   │
│  Downloadable CSV Report                                     │
└──────────────────────────────────────────────────────────────┘
```

---

# 🗂️ Project Structure

```text
industrial-boiler-predictive-maintenance/
│
├── app/
│   └── app.py
│
├── data/
│   └── realistic_boiler_dataset_v2_clean.csv
│
├── models/
│   ├── rf_classifier_model.pkl
│   ├── rf_regressor_model.pkl
│   └── scaler_regression_model.pkl
│
├── notebook/
│   └── [model development and experimentation files]
│
├── requirements.txt
├── README.md
└── .gitattributes
```

---

# ⚙️ Technology Stack

## Programming

* **Python 3.11**

## Machine Learning

* **scikit-learn**
* Random Forest Classification
* Random Forest Regression

## Data Processing

* **Pandas**
* **NumPy**

## Model Serialization

* **Joblib**

## Web Application

* **Streamlit**

## Visualization

* **Plotly**

## Deployment

* **Streamlit Community Cloud**

---

# 📦 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Shreya934-bot/industrial-boiler-predictive-maintenance.git
```

Navigate into the project:

```bash
cd industrial-boiler-predictive-maintenance
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The project uses pinned package versions for reproducibility and model compatibility.

---

## 4. Run the Application

From the project root:

```bash
streamlit run app/app.py
```

The application will open in your browser.

---

# 📋 Requirements

```text
streamlit==1.45.1
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.6.1
joblib==1.5.1
plotly==6.0.1
```

> **Important:** The scikit-learn version is pinned to maintain compatibility with the serialized trained models.

---

# 🧪 How to Use

## Batch Prediction

### Step 1

Open:

```text
🔮 Batch Prediction
```

### Step 2

Upload a CSV containing the required sensor features:

```text
temperature_C
pressure_psi
vibration_level
run_hours_since_service
fuel_flow_rate
water_level_percent
ambient_temp_C
```

### Step 3

The application validates the dataset and runs the trained models.

### Step 4

Review:

* Total records analyzed
* High-risk records
* Medium-risk records
* Low-risk records
* Average predicted days to failure
* Risk distribution
* Maintenance priority queue

### Step 5

Download the complete prediction report.

---

## Single Prediction

Navigate to:

```text
⚡ Single Prediction
```

Enter the seven sensor values manually.

Click:

```text
⚡ Run live prediction
```

The application returns an immediate maintenance assessment.

---

# 📈 Example Prediction Output

A typical prediction record can contain:

| Record | Failure Within 7 Days | Failure Probability | Predicted Days | Risk      |
| -----: | --------------------- | ------------------: | -------------: | --------- |
|      1 | Yes                   |               87.4% |           2.31 | 🔴 High   |
|      2 | Yes                   |               64.8% |           5.76 | 🟠 Medium |
|      3 | No                    |               18.2% |          16.42 | 🟢 Low    |

*The values above are illustrative examples.*

---

# 🚀 Deployment

The application is deployed using **Streamlit Community Cloud**.

### Deployment Configuration

```text
Repository:
Shreya934-bot/industrial-boiler-predictive-maintenance

Branch:
main

Main file:
app/app.py

Python:
3.11
```

### Live URL

## 🔥 [https://boiler-intelligence.streamlit.app/](https://boiler-intelligence.streamlit.app/)

For deployment, the repository includes:

* Streamlit application
* Serialized ML models
* Pinned dependencies
* Compatible Python environment configuration

---

# 🛡️ Reliability and Validation

The application includes several safeguards before running predictions.

### Data Validation

The system validates:

```text
✓ Required columns
✓ Numeric values
✓ Missing values
✓ Empty datasets
✓ Common file encodings
```

### Model Loading

The trained ML objects are loaded once using Streamlit resource caching:

```python
@st.cache_resource
```

This reduces unnecessary model reloads during application reruns.

### Prediction Safety

Regression predictions are constrained to non-negative values:

```python
np.maximum(prediction, 0)
```

---

# 🔮 Future Improvements

Potential future enhancements include:

### 🧠 Advanced Machine Learning

* XGBoost model comparison
* Ensemble learning
* Hyperparameter optimization
* Cross-validation reporting
* Automated model comparison dashboard
* Feature importance visualizations
* SHAP explainability

### 📡 Industrial Data Integration

* Real-time IoT sensor ingestion
* REST API integration
* MQTT streaming
* Automated prediction pipelines

### 🔔 Maintenance Intelligence

* Email alerts
* High-risk notifications
* Automated maintenance recommendations
* Maintenance scheduling integration

### 🗄️ Data Infrastructure

* Historical prediction database
* PostgreSQL integration
* Prediction history tracking
* Model monitoring

### ☁️ Production Deployment

* Docker containerization
* CI/CD pipeline
* Cloud deployment
* API endpoints
* Authentication and user management

---

# 🎓 Key Learning Outcomes

This project demonstrates practical experience with:

* End-to-end machine learning workflows
* Predictive maintenance systems
* Classification and regression
* Industrial sensor data
* Model serialization and loading
* Feature preprocessing
* Dependency management
* Model version compatibility
* Interactive dashboard development
* Data validation
* Plotly visualization
* Streamlit deployment
* Cloud deployment
* Production-oriented ML application design

---

# 👩‍💻 Developer

<div align="center">

## Shreya Verma

### Machine Learning Engineer • AI & Data Enthusiast • Developer

[![GitHub](https://img.shields.io/badge/GitHub-Shreya934--bot-181717?style=for-the-badge\&logo=github)](https://github.com/Shreya934-bot)

### 🔥 Boiler Intelligence

**From industrial sensor data to actionable maintenance intelligence.**

</div>

---

# ⭐ Support the Project

If you found this project interesting or useful:

* ⭐ Star the repository
* 🚀 Try the live application
* 🐛 Report issues or improvements
* 🔗 Share the project

---

<div align="center">

## 🚀 [Try Boiler Intelligence Live](https://boiler-intelligence.streamlit.app/)

<br>

**Designed & Developed by Shreya Verma**

*Building intelligent systems with Machine Learning.*

</div>
```

