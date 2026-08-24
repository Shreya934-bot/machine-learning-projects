from pathlib import Path
import io
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# BRAND IDENTITY
# ============================================================
APP_NAME = "Boiler Intelligence"
APP_TAGLINE = "Predictive Maintenance Command Center"
DEVELOPER_NAME = "Shreya Verma"
DEVELOPER_ROLE = "ML Engineer & Developer"
APP_VERSION = "v2.0"
GITHUB_URL = "https://github.com/Shreya934-bot"



# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Boiler Intelligence | Predictive Maintenance",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS + MODEL CONTRACT
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

FEATURE_COLUMNS = [
    "temperature_C",
    "pressure_psi",
    "vibration_level",
    "run_hours_since_service",
    "fuel_flow_rate",
    "water_level_percent",
    "ambient_temp_C",
]

TARGET_COLUMNS = [
    "days_to_failure",
    "failure_within_7_days",
]

FEATURE_LABELS = {
    "temperature_C": "Temperature (°C)",
    "pressure_psi": "Pressure (PSI)",
    "vibration_level": "Vibration Level",
    "run_hours_since_service": "Run Hours Since Service",
    "fuel_flow_rate": "Fuel Flow Rate",
    "water_level_percent": "Water Level (%)",
    "ambient_temp_C": "Ambient Temperature (°C)",
}

RISK_ORDER = ["High", "Medium", "Low"]
RISK_COLORS = {
    "High": "#ef4444",
    "Medium": "#f59e0b",
    "Low": "#10b981",
}


# ============================================================
# CUSTOM STYLING
# ============================================================
st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at 5% 0%, rgba(37, 99, 235, 0.12), transparent 28%),
                radial-gradient(circle at 95% 5%, rgba(16, 185, 129, 0.08), transparent 24%),
                #0b1120;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.14);
        }

        [data-testid="stSidebar"] * {
            color: #e5e7eb;
        }

        .hero {
            padding: 2.2rem 2.2rem 1.8rem 2.2rem;
            border: 1px solid rgba(96, 165, 250, 0.22);
            border-radius: 22px;
            background:
                linear-gradient(135deg, rgba(30, 41, 59, 0.96), rgba(15, 23, 42, 0.88));
            box-shadow: 0 18px 60px rgba(0, 0, 0, 0.20);
            margin-bottom: 1.2rem;
        }

        .hero-kicker {
            color: #60a5fa;
            font-size: 0.86rem;
            font-weight: 700;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }

        .hero h1 {
            color: #f8fafc;
            margin: 0;
            font-size: clamp(2rem, 5vw, 3.8rem);
            line-height: 1.05;
        }

        .hero p {
            color: #cbd5e1;
            font-size: 1.05rem;
            max-width: 850px;
            margin-top: 0.9rem;
            margin-bottom: 0;
        }

        .section-title {
            color: #f8fafc;
            font-size: 1.35rem;
            font-weight: 700;
            margin-top: 0.8rem;
            margin-bottom: 0.35rem;
        }

        .section-subtitle {
            color: #94a3b8;
            margin-bottom: 1rem;
        }

        .metric-card {
            background: linear-gradient(145deg, rgba(30, 41, 59, 0.92), rgba(15, 23, 42, 0.96));
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            min-height: 112px;
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.14);
        }

        .metric-label {
            color: #94a3b8;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .metric-value {
            color: #f8fafc;
            font-size: 2rem;
            font-weight: 800;
            margin-top: 0.3rem;
        }

        .metric-note {
            color: #cbd5e1;
            font-size: 0.78rem;
            margin-top: 0.2rem;
        }

        .risk-banner {
            border-radius: 18px;
            padding: 1rem 1.2rem;
            border: 1px solid rgba(148, 163, 184, 0.16);
            margin: 0.8rem 0 1rem 0;
        }

        .small-card {
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 16px;
            background: rgba(15, 23, 42, 0.62);
            padding: 1rem;
            min-height: 135px;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
        }

        .brand-strip {
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:1rem;
            margin-bottom:1rem;
            padding:.7rem 1rem;
            border:1px solid rgba(96,165,250,.16);
            border-radius:14px;
            background:rgba(15,23,42,.48);
        }

        .brand-mark {
            display:flex;
            align-items:center;
            gap:.7rem;
        }

        .brand-orb {
            width:34px;
            height:34px;
            border-radius:50%;
            display:grid;
            place-items:center;
            background:linear-gradient(135deg,#2563eb,#14b8a6);
            box-shadow:0 8px 24px rgba(37,99,235,.25);
            font-size:1.05rem;
        }

        .brand-name {
            color:#f8fafc;
            font-weight:800;
            font-size:.95rem;
        }

        .brand-meta {
            color:#64748b;
            font-size:.75rem;
            margin-top:.08rem;
        }

        .version-pill {
            color:#93c5fd;
            background:rgba(37,99,235,.12);
            border:1px solid rgba(96,165,250,.22);
            border-radius:999px;
            padding:.32rem .65rem;
            font-size:.72rem;
            font-weight:700;
            white-space:nowrap;
        }

        .developer-card {
            border:1px solid rgba(96,165,250,.16);
            border-radius:18px;
            padding:1rem;
            background:linear-gradient(145deg,rgba(30,41,59,.92),rgba(15,23,42,.95));
            margin:.4rem 0 .9rem 0;
        }

        .developer-name {
            color:#f8fafc;
            font-size:1rem;
            font-weight:800;
        }

        .developer-role {
            color:#94a3b8;
            font-size:.78rem;
            margin-top:.15rem;
        }

        .footer {
            margin-top:2.2rem;
            padding:1.5rem 1rem 1rem 1rem;
            border-top:1px solid rgba(148,163,184,.12);
            color:#64748b;
            text-align:center;
            font-size:.78rem;
        }

        .footer strong { color:#cbd5e1; }
        .footer a { color:#60a5fa; text-decoration:none; font-weight:700; }
        .footer-dot { color:#334155; padding:0 .35rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource(show_spinner="Loading trained machine-learning models...")
def load_models():
    files = {
        "Classifier": MODEL_DIR / "rf_classifier_model.pkl",
        "Regressor": MODEL_DIR / "rf_regressor_model.pkl",
        "Scaler": MODEL_DIR / "scaler_regression_model.pkl",
    }

    missing = [name for name, path in files.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model file(s): " + ", ".join(missing)
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        classifier = joblib.load(files["Classifier"])
        regressor = joblib.load(files["Regressor"])
        scaler = joblib.load(files["Scaler"])

    return classifier, regressor, scaler


# ============================================================
# DATA + PREDICTION HELPERS
# ============================================================
def categorize_risk(days):
    if days <= 3:
        return "High"
    if days <= 7:
        return "Medium"
    return "Low"


def risk_icon(risk):
    return {"High": "🔴", "Medium": "🟠", "Low": "🟢"}.get(risk, "⚪")


def read_uploaded_file(uploaded_file):
    """Read CSV robustly across common UTF-8/Windows encodings."""
    raw = uploaded_file.getvalue()
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except Exception as error:
            last_error = error

    raise ValueError(
        "The uploaded file could not be read as a valid CSV. "
        "Save it as CSV UTF-8 and try again."
    ) from last_error


def clean_and_prepare_features(data):
    data = data.copy()
    data.columns = (
        data.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    data = data.drop(columns=TARGET_COLUMNS, errors="ignore")

    missing = [c for c in FEATURE_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(
            "Missing required column(s): " + ", ".join(missing)
        )

    features = data[FEATURE_COLUMNS].copy()

    for column in FEATURE_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="coerce")

    invalid_mask = features.isna().any(axis=1)
    if invalid_mask.any():
        invalid_columns = [
            c for c in FEATURE_COLUMNS if features[c].isna().any()
        ]
        bad_rows = (features.index[invalid_mask] + 1).tolist()
        preview = ", ".join(map(str, bad_rows[:12]))

        raise ValueError(
            "Missing or non-numeric values found in: "
            + ", ".join(invalid_columns)
            + f". Problematic row number(s): {preview}"
        )

    if features.empty:
        raise ValueError("The uploaded CSV contains no data rows.")

    return features


def predict_records(classifier, regressor, scaler, features):
    classifier_input = features.copy()

    failure_prediction = classifier.predict(classifier_input)

    probability = None
    if hasattr(classifier, "predict_proba"):
        probability = classifier.predict_proba(classifier_input)
        if probability.ndim == 2 and probability.shape[1] > 1:
            classes = list(getattr(classifier, "classes_", []))
            if 1 in classes:
                failure_probability = probability[:, classes.index(1)] * 100
            else:
                failure_probability = probability[:, -1] * 100
        else:
            failure_probability = probability.ravel() * 100
    else:
        failure_probability = np.full(len(features), np.nan)

    scaled_features = scaler.transform(features)
    days_prediction = np.maximum(regressor.predict(scaled_features), 0)
    days_prediction = np.round(days_prediction, 2)

    result = pd.DataFrame(
        {
            "Will Fail Within 7 Days": np.where(
                np.asarray(failure_prediction).astype(int) == 1,
                "Yes",
                "No",
            ),
            "Failure Probability (%)": np.round(failure_probability, 1),
            "Predicted Days to Failure": days_prediction,
        }
    )
    result["Risk Level"] = result["Predicted Days to Failure"].apply(
        categorize_risk
    )

    return result


def combined_prediction_table(features, results):
    output = pd.concat(
        [
            features.reset_index(drop=True),
            results.reset_index(drop=True),
        ],
        axis=1,
    )
    output.insert(0, "Record", np.arange(1, len(output) + 1))
    return output


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title, subtitle, kicker="INDUSTRIAL AI OPERATIONS"):
    st.markdown(
        f"""
        <div class="brand-strip">
            <div class="brand-mark">
                <div class="brand-orb">🔥</div>
                <div>
                    <div class="brand-name">{APP_NAME}</div>
                    <div class="brand-meta">Designed & developed by {DEVELOPER_NAME}</div>
                </div>
            </div>
            <div class="version-pill">{APP_VERSION}</div>
        </div>
        <div class="hero">
            <div class="hero-kicker">{kicker} · {APP_TAGLINE}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def store_prediction(features, results):
    st.session_state["features"] = features
    st.session_state["results"] = results
    st.session_state["combined"] = combined_prediction_table(
        features, results
    )


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        f"""
        <div class="developer-card">
            <div style="font-size:1.45rem;">🔥</div>
            <div class="developer-name">{APP_NAME}</div>
            <div class="developer-role">{APP_TAGLINE}</div>
            <div style="height:.65rem;"></div>
            <div style="color:#60a5fa;font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;">Developer</div>
            <div style="color:#e2e8f0;font-weight:700;margin-top:.18rem;">{DEVELOPER_NAME}</div>
            <div style="color:#94a3b8;font-size:.76rem;margin-top:.12rem;">{DEVELOPER_ROLE}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "🔮 Batch Prediction",
            "⚡ Single Prediction",
            "📊 Analytics",
            "🤖 Model Insights",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### System Status")
    st.success("Prediction engine ready")
    st.caption("2 ML models + 1 preprocessing scaler")

    if "combined" in st.session_state:
        st.info(
            f"Latest batch: {len(st.session_state['combined']):,} record(s)"
        )

    st.divider()
    st.caption(
        "Built around your trained boiler failure classification "
        "and regression models."
    )


# ============================================================
# OVERVIEW
# ============================================================
if page == "🏠 Overview":
    render_hero(
        "Predict failures before they become downtime.",
        "Analyze boiler sensor data with your trained machine-learning "
        "models to estimate failure risk and remaining days before failure.",
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Input Sensors", "7", "Validated model features")
    with c2:
        metric_card("ML Models", "2", "Classification + regression")
    with c3:
        metric_card("Risk Tiers", "3", "High · Medium · Low")
    with c4:
        metric_card(
            "Latest Run",
            (
                f"{len(st.session_state['combined']):,}"
                if "combined" in st.session_state
                else "—"
            ),
            "Records analyzed",
        )

    st.markdown("<div class='section-title'>How the system works</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>A simple workflow from raw sensor data to maintenance insight.</div>",
        unsafe_allow_html=True,
    )

    workflow_cols = st.columns(4)
    workflow = [
        ("01", "Upload", "Upload your boiler sensor CSV."),
        ("02", "Validate", "Check columns, encoding, and numeric values."),
        ("03", "Predict", "Run classification and regression models."),
        ("04", "Act", "Prioritize maintenance from risk insights."),
    ]
    for col, (step, title, text) in zip(workflow_cols, workflow):
        with col:
            st.markdown(
                f"""
                <div class="small-card">
                    <div class="hero-kicker">STEP {step}</div>
                    <div style="font-size:1.15rem;font-weight:700;color:#f8fafc;">{title}</div>
                    <div style="color:#94a3b8;margin-top:.55rem;">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📋 Required sensor columns")
    required_df = pd.DataFrame(
        {
            "CSV Column": FEATURE_COLUMNS,
            "Description": [FEATURE_LABELS[c] for c in FEATURE_COLUMNS],
        }
    )
    st.dataframe(required_df, use_container_width=True, hide_index=True)

    with st.expander("Why are these columns required?"):
        st.write(
            "These are the seven input features expected by the trained "
            "models. The column names must match exactly. If you upload "
            "the original labelled training-style dataset, the target "
            "columns are automatically ignored during prediction."
        )


# ============================================================
# BATCH PREDICTION
# ============================================================
elif page == "🔮 Batch Prediction":
    render_hero(
        "Batch prediction workspace",
        "Upload a CSV containing one or more boiler sensor records. "
        "The app validates the data, runs both trained models, and "
        "builds an operational risk report.",
        kicker="BATCH INFERENCE",
    )

    with st.expander("📋 Required CSV format", expanded=False):
        st.code(", ".join(FEATURE_COLUMNS), language=None)
        st.caption(
            "Optional columns `days_to_failure` and "
            "`failure_within_7_days` are allowed and ignored."
        )

    uploaded_file = st.file_uploader(
        "Upload boiler sensor CSV",
        type=["csv"],
        help=(
            "UTF-8 CSV is recommended. Common Windows encodings are "
            "also tried automatically."
        ),
    )

    if uploaded_file is None:
        st.info("Upload a CSV to start a new prediction run.")
    else:
        try:
            with st.spinner("Validating data and running predictive models..."):
                raw_data = read_uploaded_file(uploaded_file)
                features = clean_and_prepare_features(raw_data)
                classifier, regressor, scaler = load_models()
                results = predict_records(
                    classifier, regressor, scaler, features
                )
                store_prediction(features, results)

            st.success(
                f"Prediction completed successfully for {len(results):,} record(s)."
            )

            combined = st.session_state["combined"]
            high = int((results["Risk Level"] == "High").sum())
            medium = int((results["Risk Level"] == "Medium").sum())
            low = int((results["Risk Level"] == "Low").sum())
            avg_days = float(results["Predicted Days to Failure"].mean())

            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                metric_card("Records", f"{len(results):,}", "Analyzed")
            with m2:
                metric_card("🔴 High", f"{high:,}", "Immediate attention")
            with m3:
                metric_card("🟠 Medium", f"{medium:,}", "Plan maintenance")
            with m4:
                metric_card("🟢 Low", f"{low:,}", "Monitor normally")
            with m5:
                metric_card("Avg. Days", f"{avg_days:.1f}", "Estimated to failure")

            risk_counts = (
                results["Risk Level"]
                .value_counts()
                .reindex(RISK_ORDER, fill_value=0)
                .rename_axis("Risk Level")
                .reset_index(name="Records")
            )

            chart_col, priority_col = st.columns([1.15, 0.85])

            with chart_col:
                st.subheader("📊 Risk distribution")
                fig = px.bar(
                    risk_counts,
                    x="Risk Level",
                    y="Records",
                    color="Risk Level",
                    color_discrete_map=RISK_COLORS,
                    category_orders={"Risk Level": RISK_ORDER},
                    text="Records",
                )
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    margin=dict(l=10, r=10, t=20, b=10),
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

            with priority_col:
                st.subheader("🚨 Maintenance priority")
                if high > 0:
                    banner_color = "rgba(239,68,68,.16)"
                    border_color = "rgba(239,68,68,.42)"
                    message = (
                        f"<b>{high} high-risk record(s)</b> need the most urgent review."
                    )
                elif medium > 0:
                    banner_color = "rgba(245,158,11,.16)"
                    border_color = "rgba(245,158,11,.42)"
                    message = (
                        f"<b>No high-risk records.</b> {medium} record(s) should be scheduled for maintenance planning."
                    )
                else:
                    banner_color = "rgba(16,185,129,.16)"
                    border_color = "rgba(16,185,129,.42)"
                    message = (
                        "<b>No high or medium-risk records detected.</b> Continue routine monitoring."
                    )

                st.markdown(
                    f"""
                    <div class="risk-banner" style="background:{banner_color};border-color:{border_color};">
                        {message}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                priority_df = combined[
                    [
                        "Record",
                        "Risk Level",
                        "Predicted Days to Failure",
                        "Failure Probability (%)",
                    ]
                ].sort_values(
                    ["Predicted Days to Failure", "Failure Probability (%)"],
                    ascending=[True, False],
                )
                st.dataframe(
                    priority_df.head(8),
                    use_container_width=True,
                    hide_index=True,
                    height=255,
                )

            st.divider()
            st.subheader("🔍 Detailed prediction results")

            filter_col1, filter_col2 = st.columns([0.45, 0.55])
            with filter_col1:
                selected_risks = st.multiselect(
                    "Filter by risk",
                    RISK_ORDER,
                    default=RISK_ORDER,
                )
            with filter_col2:
                max_rows = st.slider(
                    "Rows to display",
                    min_value=10,
                    max_value=min(max(len(combined), 10), 200),
                    value=min(len(combined), 50),
                )

            filtered = combined[
                combined["Risk Level"].isin(selected_risks)
            ].head(max_rows)

            st.dataframe(
                filtered,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Failure Probability (%)": st.column_config.NumberColumn(
                        format="%.1f%%"
                    ),
                    "Predicted Days to Failure": st.column_config.NumberColumn(
                        format="%.2f"
                    ),
                },
            )

            download_data = combined.to_csv(
                index=False, encoding="utf-8-sig"
            ).encode("utf-8-sig")

            st.download_button(
                "📥 Download complete prediction report",
                data=download_data,
                file_name="boiler_prediction_report.csv",
                mime="text/csv",
                use_container_width=True,
            )

            with st.expander("👀 View uploaded source data"):
                st.dataframe(
                    raw_data,
                    use_container_width=True,
                    hide_index=True,
                )

        except FileNotFoundError as error:
            st.error("Model files could not be found.")
            st.code(str(error))
        except ValueError as error:
            st.error("Could not process the uploaded data.")
            st.warning(str(error))
        except Exception as error:
            st.error("An unexpected error occurred during prediction.")
            st.code(f"{type(error).__name__}: {error}")
            st.caption(
                "Check that the three model files are present in the "
                "`models` folder and the package versions match "
                "`requirements.txt`."
            )


# ============================================================
# SINGLE PREDICTION
# ============================================================
elif page == "⚡ Single Prediction":
    render_hero(
        "Single boiler assessment",
        "Enter a live sensor snapshot manually to get an immediate "
        "failure-risk assessment and estimated remaining time.",
        kicker="LIVE SENSOR ASSESSMENT",
    )

    st.markdown("### 👩‍💻 Built by")
    about_col, github_col = st.columns([0.72, 0.28])
    with about_col:
        st.markdown(
            f"""
            <div class="small-card">
                <div class="hero-kicker">PROJECT DEVELOPER</div>
                <div style="font-size:1.2rem;font-weight:800;color:#f8fafc;">{DEVELOPER_NAME}</div>
                <div style="color:#94a3b8;margin-top:.45rem;">{DEVELOPER_ROLE}</div>
                <div style="color:#cbd5e1;margin-top:.7rem;line-height:1.55;">
                    Built this dashboard around a classification + regression machine-learning pipeline for industrial boiler failure prediction and maintenance prioritization.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with github_col:
        st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
        st.link_button("🐙 View Developer GitHub", GITHUB_URL, use_container_width=True)
        st.caption("Project source & ML portfolio")

    try:
        classifier, regressor, scaler = load_models()

        defaults = {
            "temperature_C": 180.0,
            "pressure_psi": 290.0,
            "vibration_level": 0.40,
            "run_hours_since_service": 400.0,
            "fuel_flow_rate": 85.0,
            "water_level_percent": 70.0,
            "ambient_temp_C": 28.0,
        }

        with st.form("single_prediction_form"):
            left, right = st.columns(2)

            values = {}
            for i, feature in enumerate(FEATURE_COLUMNS):
                target_col = left if i < 4 else right
                with target_col:
                    values[feature] = st.number_input(
                        FEATURE_LABELS[feature],
                        value=float(defaults[feature]),
                        step=0.1,
                        key=f"input_{feature}",
                    )

            submitted = st.form_submit_button(
                "⚡ Run live prediction",
                use_container_width=True,
            )

        if submitted:
            features = pd.DataFrame([values], columns=FEATURE_COLUMNS)
            results = predict_records(
                classifier, regressor, scaler, features
            )
            store_prediction(features, results)

            result = results.iloc[0]
            risk = result["Risk Level"]
            days = float(result["Predicted Days to Failure"])
            probability = float(result["Failure Probability (%)"])

            st.markdown("### Assessment result")

            if risk == "High":
                bg = "rgba(239,68,68,.16)"
                border = "rgba(239,68,68,.48)"
            elif risk == "Medium":
                bg = "rgba(245,158,11,.16)"
                border = "rgba(245,158,11,.48)"
            else:
                bg = "rgba(16,185,129,.16)"
                border = "rgba(16,185,129,.48)"

            st.markdown(
                f"""
                <div class="risk-banner" style="background:{bg};border-color:{border};">
                    <div style="font-size:1.3rem;font-weight:800;">
                        {risk_icon(risk)} {risk.upper()} RISK
                    </div>
                    <div style="color:#cbd5e1;margin-top:.4rem;">
                        The model estimates approximately <b>{days:.2f} days</b> until failure.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            a, b, c = st.columns(3)
            with a:
                metric_card("Risk Level", risk, "Based on predicted days")
            with b:
                metric_card("Days Remaining", f"{days:.2f}", "Regression estimate")
            with c:
                metric_card(
                    "Failure Probability",
                    f"{probability:.1f}%",
                    "Classifier output",
                )

            st.caption(
                "Predictions are model estimates intended to support maintenance decisions."
            )

    except Exception as error:
        st.error("Could not load the prediction engine.")
        st.code(f"{type(error).__name__}: {error}")


# ============================================================
# ANALYTICS
# ============================================================
elif page == "📊 Analytics":
    render_hero(
        "Prediction analytics",
        "Explore the latest inference run through risk distribution, "
        "failure estimates, and sensor-level relationships.",
        kicker="OPERATIONS ANALYTICS",
    )

    if "combined" not in st.session_state:
        st.info(
            "Run a batch or single prediction first. Analytics for the "
            "latest prediction run will appear here."
        )
    else:
        combined = st.session_state["combined"].copy()

        left, right = st.columns(2)

        with left:
            st.subheader("Risk composition")
            risk_counts = (
                combined["Risk Level"]
                .value_counts()
                .reindex(RISK_ORDER, fill_value=0)
                .reset_index()
            )
            risk_counts.columns = ["Risk Level", "Records"]

            fig = px.pie(
                risk_counts,
                names="Risk Level",
                values="Records",
                color="Risk Level",
                color_discrete_map=RISK_COLORS,
                hole=0.58,
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=20, b=10),
                height=380,
                legend_title_text="",
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("Days-to-failure distribution")
            fig = px.histogram(
                combined,
                x="Predicted Days to Failure",
                nbins=min(30, max(10, len(combined) // 5)),
                color_discrete_sequence=["#60a5fa"],
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=20, b=10),
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Sensor relationship explorer")

        x_feature, y_feature = st.columns(2)
        with x_feature:
            x_axis = st.selectbox(
                "X-axis sensor",
                FEATURE_COLUMNS,
                format_func=lambda x: FEATURE_LABELS[x],
                index=0,
            )
        with y_feature:
            y_axis = st.selectbox(
                "Y-axis sensor",
                FEATURE_COLUMNS,
                format_func=lambda x: FEATURE_LABELS[x],
                index=1,
            )

        fig = px.scatter(
            combined,
            x=x_axis,
            y=y_axis,
            color="Risk Level",
            color_discrete_map=RISK_COLORS,
            hover_data=[
                "Record",
                "Predicted Days to Failure",
                "Failure Probability (%)",
            ],
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=20, b=10),
            height=460,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Risk-prioritized maintenance queue")
        queue = combined.sort_values(
            ["Predicted Days to Failure", "Failure Probability (%)"],
            ascending=[True, False],
        )[
            [
                "Record",
                "Risk Level",
                "Will Fail Within 7 Days",
                "Failure Probability (%)",
                "Predicted Days to Failure",
            ]
        ]
        st.dataframe(queue, use_container_width=True, hide_index=True)


# ============================================================
# MODEL INSIGHTS
# ============================================================
elif page == "🤖 Model Insights":
    render_hero(
        "Model intelligence",
        "A transparent view of the exact prediction pipeline used by "
        "this application.",
        kicker="MODEL INSIGHTS",
    )

    st.markdown("### Prediction pipeline")

    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown(
            """
            <div class="small-card">
                <div class="hero-kicker">MODEL 01</div>
                <div style="font-size:1.15rem;font-weight:700;color:#f8fafc;">Random Forest Classifier</div>
                <div style="color:#94a3b8;margin-top:.55rem;">
                    Predicts whether a failure is expected within 7 days.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p2:
        st.markdown(
            """
            <div class="small-card">
                <div class="hero-kicker">PREPROCESSING</div>
                <div style="font-size:1.15rem;font-weight:700;color:#f8fafc;">StandardScaler</div>
                <div style="color:#94a3b8;margin-top:.55rem;">
                    Applies the same feature scaling expected by the regression pipeline.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p3:
        st.markdown(
            """
            <div class="small-card">
                <div class="hero-kicker">MODEL 02</div>
                <div style="font-size:1.15rem;font-weight:700;color:#f8fafc;">Random Forest Regressor</div>
                <div style="color:#94a3b8;margin-top:.55rem;">
                    Estimates the remaining number of days before failure.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Risk policy used by the app")
    risk_policy = pd.DataFrame(
        {
            "Risk Level": ["🔴 High", "🟠 Medium", "🟢 Low"],
            "Predicted Days to Failure": ["≤ 3 days", "> 3 and ≤ 7 days", "> 7 days"],
            "Operational Interpretation": [
                "Prioritize immediate review",
                "Plan maintenance soon",
                "Continue routine monitoring",
            ],
        }
    )
    st.dataframe(risk_policy, use_container_width=True, hide_index=True)

    st.markdown("### Model input contract")
    contract = pd.DataFrame(
        {
            "Order": range(1, len(FEATURE_COLUMNS) + 1),
            "Feature": FEATURE_COLUMNS,
            "Display Name": [FEATURE_LABELS[c] for c in FEATURE_COLUMNS],
        }
    )
    st.dataframe(contract, use_container_width=True, hide_index=True)

    try:
        classifier, regressor, scaler = load_models()
        with st.expander("Technical model metadata"):
            st.write(
                {
                    "Classifier type": type(classifier).__name__,
                    "Regressor type": type(regressor).__name__,
                    "Scaler type": type(scaler).__name__,
                    "Expected features": getattr(
                        classifier, "n_features_in_", len(FEATURE_COLUMNS)
                    ),
                }
            )
    except Exception as error:
        st.warning(f"Could not read model metadata: {error}")


st.markdown(
    f"""
    <div class="footer">
        <div><strong>{APP_NAME}</strong> <span class="footer-dot">•</span> {APP_TAGLINE}</div>
        <div style="margin-top:.45rem;">
            Designed & developed by <strong>{DEVELOPER_NAME}</strong>
            <span class="footer-dot">•</span> {DEVELOPER_ROLE}
            <span class="footer-dot">•</span> <a href="{GITHUB_URL}" target="_blank">GitHub</a>
            <span class="footer-dot">•</span> {APP_VERSION}
        </div>
        <div style="margin-top:.45rem;color:#475569;">Built with Streamlit, scikit-learn and Plotly for intelligent industrial maintenance.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
