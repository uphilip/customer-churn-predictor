from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# VISUAL THEME
# ============================================================

st.markdown(
    """
    <style>

    /* Main application background */
    .stApp {
        background: linear-gradient(
            135deg,
            #f8fafc 0%,
            #eef2ff 50%,
            #f8fafc 100%
        );
    }

    /* Main title */
    h1 {
        font-weight: 800 !important;

        background: linear-gradient(
            90deg,
            #2563eb,
            #7c3aed
        );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Subtitle */
    h2 {
        color: #475569 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #111827 0%,
            #312e81 55%,
            #4c1d95 100%
        );
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.20) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(
            90deg,
            #2563eb,
            #7c3aed
        );

        color: white !important;

        border: none;

        border-radius: 10px;

        font-weight: 700;

        min-height: 48px;
    }

    .stButton > button:hover {
        background: linear-gradient(
            90deg,
            #1d4ed8,
            #6d28d9
        );

        color: white !important;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: white;

        border: 1px solid #e2e8f0;

        padding: 18px;

        border-radius: 14px;

        box-shadow:
            0 6px 18px rgba(15, 23, 42, 0.06);
    }

    /* Expander */
    div[data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "customer_churn_model.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


try:

    model = load_model()

except Exception as error:

    st.error("Unable to load the trained model.")

    st.exception(error)

    st.stop()


# ============================================================
# PROJECT HEADER
# ============================================================

st.caption(
    "3MTT AI/ML CAPSTONE PROJECT"
)

st.title(
    "Customer Churn Predictor"
)

st.subheader(
    "Machine Learning-Based Customer Retention Risk Analysis"
)

st.write(
    "**Developed by Chukwuka Uchenna**"
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("Customer Churn")

    st.write(
        "A machine-learning application designed to estimate "
        "customer churn risk."
    )

    st.divider()

    st.subheader("Model Information")

    st.write("**Algorithm**")
    st.write("Logistic Regression")

    st.write("**Problem Type**")
    st.write("Binary Classification")

    st.write("**ROC-AUC**")
    st.write("84.16%")

    st.write("**Recall**")
    st.write("78.34%")

    st.divider()

    st.subheader("Project")

    st.write("3MTT AI/ML Capstone Project")

    st.write("Developed by **Chukwuka Uchenna**")


# ============================================================
# CUSTOMER PROFILE
# ============================================================

st.header("👤 Customer Profile")

st.write(
    "Provide the customer's demographic and account information."
)

col1, col2, col3 = st.columns(3)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

with col2:

    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

with col3:

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )


col1, col2, col3 = st.columns(3)

with col1:

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

with col2:

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12,
        step=1
    )

with col3:

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )


# ============================================================
# INTERNET SERVICES
# ============================================================

st.header("🌐 Internet & Online Services")

st.write(
    "Select the customer's internet and additional online services."
)

col1, col2, col3 = st.columns(3)

with col1:

    multiple_lines = st.selectbox(
        "Multiple Lines",
        [
            "No",
            "Yes",
            "No phone service"
        ]
    )

with col2:

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

with col3:

    online_security = st.selectbox(
        "Online Security",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )


col1, col2, col3 = st.columns(3)

with col1:

    online_backup = st.selectbox(
        "Online Backup",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

with col2:

    device_protection = st.selectbox(
        "Device Protection",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

with col3:

    tech_support = st.selectbox(
        "Technical Support",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )


col1, col2 = st.columns(2)

with col1:

    streaming_tv = st.selectbox(
        "Streaming TV",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

with col2:

    streaming_movies = st.selectbox(
        "Streaming Movies",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )


# ============================================================
# CONTRACT AND BILLING
# ============================================================

st.header("💳 Contract & Billing")

st.write(
    "Provide the customer's contract and billing information."
)

col1, col2, col3 = st.columns(3)

with col1:

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

with col2:

    paperless_billing = st.selectbox(
        "Paperless Billing",
        [
            "Yes",
            "No"
        ]
    )

with col3:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


col1, col2 = st.columns(2)

with col1:

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=1000.0,
        value=70.0,
        step=0.01
    )

with col2:

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=100000.0,
        value=840.0,
        step=0.01
    )


# ============================================================
# CUSTOMER DATA
# ============================================================

customer_data = pd.DataFrame(
    {
        "gender": [gender],

        "SeniorCitizen": [
            1 if senior_citizen == "Yes" else 0
        ],

        "Partner": [partner],

        "Dependents": [dependents],

        "tenure": [tenure],

        "PhoneService": [phone_service],

        "MultipleLines": [multiple_lines],

        "InternetService": [internet_service],

        "OnlineSecurity": [online_security],

        "OnlineBackup": [online_backup],

        "DeviceProtection": [device_protection],

        "TechSupport": [tech_support],

        "StreamingTV": [streaming_tv],

        "StreamingMovies": [streaming_movies],

        "Contract": [contract],

        "PaperlessBilling": [paperless_billing],

        "PaymentMethod": [payment_method],

        "MonthlyCharges": [monthly_charges],

        "TotalCharges": [total_charges]
    }
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict = st.button(
    "🔍 Predict Customer Churn",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict:

    try:

        prediction = model.predict(
            customer_data
        )[0]

        probabilities = model.predict_proba(
            customer_data
        )[0]

        classes = list(model.classes_)

        if "Yes" in classes:

            churn_index = classes.index("Yes")

        elif 1 in classes:

            churn_index = classes.index(1)

        else:

            churn_index = 1

        churn_probability = float(
            probabilities[churn_index]
        )

        retention_probability = (
            1 - churn_probability
        )

        churn_percentage = (
            churn_probability * 100
        )

        retention_percentage = (
            retention_probability * 100
        )


        # ====================================================
        # RISK CLASSIFICATION
        # ====================================================

        if churn_probability >= 0.70:

            risk_level = "HIGH RISK"

            risk_message = (
                "The customer shows a high predicted "
                "likelihood of leaving the service. "
                "Retention action may deserve priority."
            )

        elif churn_probability >= 0.40:

            risk_level = "MEDIUM RISK"

            risk_message = (
                "The customer shows a moderate predicted "
                "likelihood of leaving the service."
            )

        else:

            risk_level = "LOW RISK"

            risk_message = (
                "The customer shows a relatively low "
                "predicted likelihood of leaving the service."
            )


        # ====================================================
        # RESULT
        # ====================================================

        st.divider()

        st.header("📈 Churn Risk Assessment")

        st.subheader(
            "Logistic Regression Output"
        )


        if risk_level == "HIGH RISK":

            st.error(
                "🔴 HIGH RISK"
            )

        elif risk_level == "MEDIUM RISK":

            st.warning(
                "🟠 MEDIUM RISK"
            )

        else:

            st.success(
                "🟢 LOW RISK"
            )


        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Churn Probability",
                f"{churn_percentage:.1f}%"
            )

        with col2:

            st.metric(
                "Retention Probability",
                f"{retention_percentage:.1f}%"
            )


        st.progress(
            churn_probability,
            text=(
                f"Estimated churn probability: "
                f"{churn_percentage:.1f}%"
            )
        )


        st.info(
            risk_message
        )


        # ====================================================
        # MODEL INSIGHTS
        # ====================================================

        st.header("🧠 Model Insights")


        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "↑ Factors Associated With Higher Churn"
            )

            st.write(
                "• Fiber optic internet service"
            )

            st.write(
                "• Month-to-month contract"
            )

            st.write(
                "• Electronic check payment"
            )

            st.write(
                "• Absence of online security"
            )

            st.write(
                "• Absence of technical support"
            )

            st.write(
                "• Streaming service usage"
            )


        with col2:

            st.subheader(
                "↓ Factors Associated With Lower Churn"
            )

            st.write(
                "• Longer customer tenure"
            )

            st.write(
                "• Two-year contract"
            )

            st.write(
                "• DSL internet service"
            )

            st.write(
                "• Established customer relationship"
            )

            st.write(
                "• Longer-term contractual commitment"
            )


        # ====================================================
        # TECHNICAL INTERPRETATION
        # ====================================================

        with st.expander(
            "🔬 Technical Interpretation"
        ):

            st.write(
                f"""
                The Logistic Regression model estimated a
                **{churn_percentage:.1f}%** probability that the
                supplied customer profile will churn.

                The estimated retention probability is
                **{retention_percentage:.1f}%**.

                During evaluation, the model achieved an
                ROC-AUC of **84.16%** and recall of
                **78.34%** for the churn class.

                The prediction is a statistical estimate based
                on the customer characteristics supplied to
                the trained machine-learning model.
                """
            )


        # ====================================================
        # SUMMARY
        # ====================================================

        st.subheader(
            "📋 Prediction Summary"
        )

        summary = pd.DataFrame(
            {
                "Metric": [
                    "Risk Classification",
                    "Churn Probability",
                    "Retention Probability",
                    "Model"
                ],

                "Result": [
                    risk_level,
                    f"{churn_percentage:.1f}%",
                    f"{retention_percentage:.1f}%",
                    "Logistic Regression"
                ]
            }
        )

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )


    except Exception as error:

        st.error(
            "An error occurred while generating the prediction."
        )

        st.exception(error)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Customer Churn Predictor | "
    "3MTT AI/ML Capstone Project | "
    "Developed by Chukwuka Uchenna"
)