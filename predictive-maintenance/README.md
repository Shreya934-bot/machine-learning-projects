# 🔧 Predictive Maintenance of Boiler System

## 📌 Overview

This project focuses on predicting potential failures in industrial boiler systems using machine learning.
It combines classification and regression models to estimate both **whether a failure will occur within 7 days** and **how many days remain before failure**.

The goal is to enable **proactive maintenance**, reduce downtime, and improve operational safety.

---

## 🏭 Real-World Context

This project is inspired by a real-world industrial scenario from a **thermal power plant (Rosa)**, where boiler systems are critical infrastructure.

Failures in such systems can lead to:

* ⚠️ Unexpected downtime
* 💸 High maintenance costs
* 🔥 Safety risks

This project simulates a **predictive maintenance system** using historical sensor data to anticipate failures before they occur.

---

## ❗ Problem Statement

Traditional maintenance strategies are:

* **Reactive** → Fix after failure
* **Scheduled** → Maintenance at fixed intervals

Both approaches are inefficient.

This project aims to build a **data-driven predictive system** that:

* Identifies failures in advance
* Estimates remaining useful life
* Supports smarter maintenance decisions

---

## 🚀 Key Features

* 🔍 **Failure Prediction (Classification)**
  Predicts whether a system will fail within the next 7 days

* ⏳ **Time-to-Failure Estimation (Regression)**
  Predicts number of days remaining before failure

* ⚠️ **Risk Categorization**

  * High (≤ 3 days)
  * Medium (≤ 7 days)
  * Low (> 7 days)

* 📊 **Interactive Visualization**

  * Risk distribution chart

* 📥 **CSV Upload & Download**

  * Upload sensor data
  * Download predictions

---

## ⚙️ Machine Learning Pipeline

1. **Data Collection**

   * Industrial boiler sensor dataset

2. **Data Preprocessing**

   * Handling missing values
   * Removing invalid inputs
   * Feature scaling

3. **Feature Engineering**

   * Preparing input features for model performance

4. **Model Building**

   * Random Forest Classifier → Failure prediction
   * Random Forest Regressor → Days to failure

5. **Model Evaluation**

   * Classification → Accuracy, F1-score
   * Regression → Mean Absolute Error

6. **Deployment**

   * Streamlit web application for real-time usage

---

## 🖥️ Streamlit Application

An interactive web app allows users to:

* Upload boiler sensor data
* Get instant predictions
* Visualize risk distribution
* Download results

### ▶️ Run locally:

```bash
streamlit run app/app.py
```

---

## 📁 Project Structure

```bash
predictive-maintenance/
│
├── notebook/
│   └── boiler_predictive_maintenance.ipynb
│
├── data/
│   └── boiler_dataset.xlsx
│
├── models/
│   ├── rf_classifier_model.pkl
│   ├── rf_regressor_model.pkl
│   └── scaler_regression_model.pkl
│
├── app/
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

## 📊 Sample Output

* Will Fail in 7 Days → Yes / No
* Predicted Days to Failure
* Risk Level (High / Medium / Low)

---

## 📈 Key Insights

* High-risk systems typically fail within **3 days**
* Clean data significantly improves prediction accuracy
* Combining classification + regression provides better decision support

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Streamlit
* Altair

---

## 🎯 Use Case

This project demonstrates how machine learning can be applied in industrial systems to:

* Reduce unexpected failures
* Optimize maintenance schedules
* Improve system reliability

---

## 🔮 Future Improvements

* Deploy the app online
* Integrate real-time sensor data
* Experiment with advanced models (XGBoost, Deep Learning)

---

🚀 Built as part of my machine learning portfolio with a focus on solving real-world industrial problems.
