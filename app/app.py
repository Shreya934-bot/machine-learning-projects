import joblib
import streamlit as st
import pandas as pd
import altair as alt

# Load models
rf_classifier_model = joblib.load("../models/rf_classifier_model.pkl")
rf_regressor_model = joblib.load("../models/rf_regressor_model.pkl")
scaler = joblib.load("../models/scaler_regression_model.pkl")

# Categorize risk
def categorize_risk(days):
    if days <= 3:
        return "High"
    elif days <= 7:
        return "Medium"
    else:
        return "Low"


# Page configuration
st.set_page_config(page_title="Boiler Failure Predictor", page_icon="🔥", layout="centered")

# Custom background using CSS
st.markdown(
    """
    <style>
    .main {
        background-color: #2C3E50;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# App title and description
st.title("🔧 Predictive Maintenance - Boiler System")
st.markdown("Upload boiler sensor data to predict failure risks and plan maintenance proactively.")
st.markdown("---")

# File uploader
uploaded_file = st.file_uploader("📤 Upload your boiler sensor CSV file", type="csv")

# If file is uploaded
if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)

        # Drop target columns if present
        data = data.drop(columns=["days_to_failure", "failure_within_7_days"], errors="ignore")

        # Classifier prediction
        fail_prediction = rf_classifier_model.predict(data)
        fail_yes_no = ["Yes" if val == 1 else "No" for val in fail_prediction]

        # Regression prediction
        data_scaled = scaler.transform(data)
        days_prediction = rf_regressor_model.predict(data_scaled)
        risks = [categorize_risk(d) for d in days_prediction]

        # Create result DataFrame
        result_df = pd.DataFrame({
            "Will Fail in 7 Days?": fail_yes_no,
            "Predicted Days to Failure": days_prediction,
            "Predicted Risk Level": risks
        })

        st.markdown("---")
        st.subheader("🔍 Prediction Results:")

        # Highlight risk level with color
        def highlight_risk(val):
            if val == "High":
                return "background-color: #FF6B6B"
            elif val == "Medium":
                return "background-color: #FFD93D"
            elif val == "Low":
                return "background-color: #6BCB77"
            return ""

        styled_df = result_df.style.applymap(highlight_risk, subset=["Predicted Risk Level"])
        st.dataframe(styled_df, use_container_width=True)

        # Create bar chart for risk level distribution
        risk_counts = pd.DataFrame(result_df["Predicted Risk Level"].value_counts()).reset_index()
        risk_counts.columns = ["Risk Level", "Count"]

        chart = alt.Chart(risk_counts).mark_bar().encode(
            x=alt.X("Risk Level", sort=["High", "Medium", "Low"]),
            y="Count",
            color=alt.Color("Risk Level", scale=alt.Scale(domain=["High", "Medium", "Low"],
                                                          range=["#FF6B6B", "#FFD93D", "#6BCB77"]))
        ).properties(
            title="Risk Level Distribution",
            width=500,
            height=300
        )

        st.markdown("### 📊 Risk Level Graph:")
        st.altair_chart(chart, use_container_width=True)

        # Download CSV button
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="boiler_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
