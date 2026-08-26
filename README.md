# 🔥 Boiler Intelligence
### AI-Powered Predictive Maintenance for Industrial Boiler Systems

<p align="center">
  <strong>Predict failure risk. Estimate time-to-failure. Enable proactive maintenance.</strong>
</p>

<p align="center">
  <a href="https://boiler-intelligence.streamlit.app/">🚀 Live Application</a> •
  <a href="https://github.com/Shreya934-bot/industrial-boiler-predictive-maintenance">📂 Source Code</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Predictive%20Maintenance-00A67E?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Live-success?style=for-the-badge" />
</p>

---

## 🚀 Live Demo

### 🔗 [Open Boiler Intelligence →](https://boiler-intelligence.streamlit.app/)

Boiler Intelligence is a live, interactive machine-learning application built to transform industrial boiler sensor data into actionable maintenance intelligence.

**What the application can do:**

- 🏠 **Overview** — Command-center view of the predictive maintenance system
- 🤖 **Batch Prediction** — Upload sensor data and generate predictions at scale
- ⚡ **Single Prediction** — Enter sensor values manually for instant inference
- 📊 **Analytics** — Explore prediction and risk patterns through interactive visualizations
- 🧠 **Model Insights** — Inspect the machine-learning pipeline and model information
- 📥 **Export Results** — Download generated prediction outputs for further analysis

---

# 🎯 The Problem

Industrial boiler failures can lead to:

> ⚠️ Unexpected downtime  
> 💸 Expensive emergency maintenance  
> 🔥 Operational and safety risks  
> 📉 Reduced equipment reliability

Traditional maintenance strategies often operate in one of two ways:

| Strategy | Limitation |
|---|---|
| **Reactive Maintenance** | Action is taken only after a failure occurs |
| **Scheduled Maintenance** | Components may be serviced even when maintenance is unnecessary |

### The smarter alternative: Predictive Maintenance

Instead of waiting for equipment to fail, **Boiler Intelligence analyzes operational sensor data to identify risk earlier and estimate the remaining time before failure.**

This helps shift maintenance from:

```text
FAILURE → REPAIR
```

to:

```text
SENSOR DATA → ML ANALYSIS → EARLY WARNING → PROACTIVE ACTION
```

---

# 🧠 Solution Overview

Boiler Intelligence combines **classification and regression** into a single predictive maintenance workflow.

### 1️⃣ Failure Risk Prediction

The classification pipeline estimates whether the boiler is likely to experience failure within the defined prediction horizon.

### 2️⃣ Remaining Time Estimation

The regression pipeline estimates the number of days remaining before a potential failure.

### 3️⃣ Risk Intelligence

Predicted time-to-failure is converted into an actionable maintenance risk tier:

| 🔴 Risk | Estimated Time Before Failure |
|---|---|
| **High** | ≤ 3 days |
| **Medium** | > 3 and ≤ 7 days |
| **Low** | > 7 days |

The result is not just a prediction — it is a maintenance-oriented decision signal.

---

# ⚙️ Machine Learning Workflow

```text
                         ┌──────────────────────┐
                         │   Boiler Sensor Data │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Data Validation &    │
                         │ Preprocessing        │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Feature Preparation  │
                         │ + Scaling            │
                         └──────────┬───────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
          ┌──────────────────────┐      ┌──────────────────────┐
          │ Classification Model │      │ Regression Model     │
          │ Failure Risk         │      │ Days to Failure      │
          └──────────┬───────────┘      └──────────┬───────────┘
                     └──────────────┬──────────────┘
                                    ▼
                         ┌──────────────────────┐
                         │ Risk Categorization  │
                         │ High / Medium / Low  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Maintenance Decision │
                         │ Support & Insights   │
                         └──────────────────────┘
```

---

# ✨ Application Highlights

## 🖥️ Industrial Command-Center UI

The application is designed as a modern predictive-maintenance dashboard rather than a basic ML form.

Features include:

- Premium dark industrial interface
- Branded **Boiler Intelligence** command-center experience
- Clear system status indicators
- Dedicated navigation for different prediction workflows
- Responsive KPI-style information cards
- Interactive analytics and visualization components

## 📂 Batch Prediction

Upload a compatible CSV file containing boiler sensor readings.

The application:

1. Validates the uploaded dataset
2. Checks required model features
3. Handles supported numeric input
4. Runs predictions through the trained ML pipeline
5. Generates failure-risk and time-to-failure outputs
6. Assigns a maintenance risk tier
7. Presents results for analysis and export

## ⚡ Single Prediction

For one-off scenarios, sensor values can be entered manually to receive immediate predictive insights.

Useful for:

- Engineering experiments
- Scenario analysis
- Manual equipment checks
- Demonstrating model inference

## 📊 Analytics

Visual outputs help transform raw model predictions into patterns that are easier to interpret for maintenance planning.

## 🧠 Model Insights

The application exposes a dedicated area for understanding the predictive system and its role in the overall maintenance workflow.

---

# 📁 Project Structure

```text
industrial-boiler-predictive-maintenance/
│
├── app/
│   └── app.py                         # Main Streamlit application
│
├── data/
│   ├── realistic_boiler_dataset...    # Boiler sensor datasets
│   └── realistic_boiler_dataset_v2_clean.csv
│
├── models/
│   ├── rf_classifier_model.pkl        # Failure classification model
│   ├── rf_regressor_model.pkl         # Time-to-failure regression model
│   └── scaler_regression_model.pkl    # Feature preprocessing scaler
│
├── notebook/
│   └── boiler_predictive_maintenance.ipynb
│
├── requirements.txt                   # Python dependencies
└── README.md
```

> **Note:** Keep the `models/` directory available when running the application locally because the deployed inference pipeline depends on the serialized model artifacts.

---

# 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **Language** | Python |
| **Web Application** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn |
| **Model Persistence** | Joblib |
| **Visualization** | Plotly |
| **Deployment** | Streamlit Community Cloud |
| **Version Control** | Git & GitHub |

---

# 💻 Run Locally

## 1. Clone the repository

```bash
git clone https://github.com/Shreya934-bot/industrial-boiler-predictive-maintenance.git
cd industrial-boiler-predictive-maintenance
```

## 2. Create a virtual environment

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

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the application

```bash
streamlit run app/app.py
```

The application will start locally and can be opened through the Streamlit URL shown in your terminal.

---

# 📥 Expected Input

The application expects boiler sensor data with the feature structure used by the trained model pipeline.

Before inference, uploaded data should:

- Match the expected feature names
- Contain numeric sensor values where required
- Avoid corrupted or incompatible encodings
- Use CSV format for the batch-upload workflow

The application performs validation and reports problematic columns or rows when the input cannot be processed safely.

---

# 📤 Prediction Output

Each successful prediction workflow provides maintenance-oriented information such as:

```text
Failure Prediction
        ↓
Estimated Days Before Failure
        ↓
Maintenance Risk Level
        ↓
Actionable Predictive Insight
```

### Example interpretation

| Model Output | Example Meaning |
|---|---|
| **Failure Risk** | Indicates whether the system is predicted to be at risk |
| **Days to Failure** | Estimated remaining time before failure |
| **High Risk** | Immediate maintenance attention recommended |
| **Medium Risk** | Schedule inspection or maintenance soon |
| **Low Risk** | Continue monitoring under normal conditions |

> Predictions are decision-support outputs generated from the trained model and should not replace established industrial safety procedures or expert engineering judgment.

---

# 🔬 Project Background

This project was developed around a real-world industrial predictive-maintenance use case involving boiler systems, where early identification of abnormal operating conditions can support proactive maintenance planning.

The core idea is simple:

> **Use historical sensor patterns to estimate what may happen before a costly failure occurs.**

By combining failure classification with time-to-failure estimation, the system provides a broader maintenance picture than a binary prediction alone.

---

# 📈 Why Classification + Regression?

A classifier can answer:

> **"Is a failure likely?"**

But maintenance teams also need to know:

> **"How soon could it happen?"**

Boiler Intelligence therefore combines two complementary prediction tasks:

```text
Classification
      +
Regression
      ↓
More useful maintenance intelligence
```

This approach enables risk prioritization based on both **failure likelihood** and **estimated remaining time**.

---

# 🚀 Deployment

The project is deployed as a live Streamlit application.

### 🌐 Live Application

👉 **[boiler-intelligence.streamlit.app](https://boiler-intelligence.streamlit.app/)**

The deployment uses the repository's Streamlit application entry point and dependency configuration to run the ML inference experience online.

---

# 🧪 Future Improvements

The project can be extended with:

- [ ] Real-time IoT sensor streaming
- [ ] Live equipment health monitoring
- [ ] Automated maintenance alerts
- [ ] Model confidence and uncertainty visualization
- [ ] SHAP-based model explainability
- [ ] Historical prediction storage
- [ ] Authentication and role-based dashboards
- [ ] REST API integration
- [ ] Docker containerization
- [ ] CI/CD automation
- [ ] Model drift detection
- [ ] Cloud database integration
- [ ] Advanced ensemble experimentation
- [ ] Deep-learning-based time-series models

---

# 📸 Screenshots

> Add screenshots of the deployed application here to make the repository even stronger.

Suggested screenshots:

- 🏠 Command Center / Overview
- 🤖 Batch Prediction
- ⚡ Single Prediction
- 📊 Analytics Dashboard
- 🧠 Model Insights
- 📈 Prediction Results

Example:

```md
![Boiler Intelligence Dashboard](assets/dashboard.png)
```

---

# 👩‍💻 Developer

## **Shreya Verma**

**ML Engineer & Developer**

Built with a focus on applying machine learning to a practical industrial predictive-maintenance problem — from model development and preprocessing to an interactive user interface and live cloud deployment.

<p>
  <a href="https://github.com/Shreya934-bot">GitHub</a> •
  <a href="https://github.com/Shreya934-bot/industrial-boiler-predictive-maintenance">Project Repository</a> •
  <a href="https://boiler-intelligence.streamlit.app/">Live Application</a>
</p>

---

<p align="center">
  <strong>🔥 Boiler Intelligence</strong><br/>
  <em>Predict earlier. Maintain smarter. Keep critical systems running.</em>
</p>

<p align="center">
  ⭐ If you found this project interesting, consider starring the repository.
</p>
