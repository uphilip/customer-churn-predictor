import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# CUSTOMER CHURN PREDICTOR
# AI/ML CAPSTONE PROJECT
#
# Developed by: Chukwuka Uchenna
#
# This application uses a trained Logistic Regression model
# to estimate the probability that a customer will churn.
#
# The saved machine-learning pipeline contains the required
# preprocessing steps and the trained classification model.
# ============================================================


# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ------------------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------------------

st.markdown(
    """
    <style>

    /* -------------------------------------------------------
       GLOBAL APPLICATION
       ------------------------------------------------------- */

    .stApp {
        background-color: #f5f7fb;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }


    /* -------------------------------------------------------
       HEADER
       ------------------------------------------------------- */

    .project-header {
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #172554 55%,
            #1e3a8a 100%
        );

        padding: 35px 40px;
        border-radius: 18px;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.12);
    }

    .project-title {
        color: #ffffff;
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 5px;
        letter-spacing: -0.5px;
    }

    .project-subtitle {
        color: #cbd5e1;
        font-size: 17px;
        margin-bottom: 18px;
    }

    .developer-name {
        color: #ffffff;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }


    /* -------------------------------------------------------
       SECTION HEADINGS
       ------------------------------------------------------- */

    .section-heading {
        font-size: 23px;
        font-weight: 750;
        color: #0f172a;
        margin-top: 15px;
        margin-bottom: 15px;
    }


    /* -------------------------------------------------------
       INFORMATION CARDS
       ------------------------------------------------------- */

    .info-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
        margin-bottom: 15px;
    }


    /* -------------------------------------------------------
       RESULT PANEL
       ------------------------------------------------------- */

    .risk-panel {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 32px;
        margin-top: 15px;
        margin-bottom: 25px;
        box-shadow: 0 5px 20px rgba(15, 23, 42, 0.06);
    }

    .risk-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 24px;
    }

    .risk-header-title {
        font-size: 14px;
        font-weight: 800;
        letter-spacing: 1.2px;
        color: #64748b;
    }

    .risk-header-label {
        font-size: 13px;
        font-weight: 600;
        color: #94a3b8;
    }

    .risk-level {
        font-size: 15px;
        font-weight: 800;
        letter-spacing: 1.2px;
        margin-bottom: 5px;
    }

    .risk-percentage {
        font-size: 60px;
        line-height: 1.05;
        font-weight: 850;
        color: #0f172a;
        margin-bottom: 5px;
    }

    .risk-caption {
        font-size: 16px;
        color: #64748b;
        margin-bottom: 22px;
    }

    .risk-track {
        width: 100%;
        height: 11px;
        background: #e2e8f0;
        border-radius: 20px;
        overflow: hidden;
        margin-bottom: 22px;
    }

    .risk-fill {
        height: 100%;
        border-radius: 20px;
    }

    .risk-explanation {
        font-size: 15px;
        line-height: 1.7;
        color: #475569;
        max-width: 900px;
    }


    /* -------------------------------------------------------
       SUMMARY CARDS
       ------------------------------------------------------- */

    .summary-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 13px;
        padding: 20px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
    }

    .summary-label {
        font-size: 12px;
        font-weight: 800;
        color: #64748b;
        letter-spacing: 0.8px;
        margin-bottom: 7px;
    }

    .summary-value {
        font-size: 30px;
        font-weight: 800;
    }


    /* -------------------------------------------------------
       MODEL INSIGHT CARDS
       ------------------------------------------------------- */

    .insight-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 22px;
        height: 100%;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
    }

    .insight-title {
        font-size: 15px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 15px;
    }

    .insight-item {
        font-size: 14px;
        color: #475569;
        padding: 7px 0;
        border-bottom: 1px solid #f1f5f9;
    }


    /* -------------------------------------------------------
       BUTTON
       ------------------------------------------------------- */

    div.stButton > button {
        width: 100%;
        height: 55px;
        border-radius: 12px;
        font-size: 17px;
        font-weight: 800;
        letter-spacing: 0.3px;
    }


    /* -------------------------------------------------------
       SIDEBAR
       ------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background-color: #0f172a;
    }

    section[data-testid="stSidebar"] * {
        color: #f8fafc;
    }


    /* -------------------------------------------------------
       FOOTER
       ------------------------------------------------------- */

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 13px;
        padding-top: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# MODEL PATH
# ------------------------------------------------------------

MODEL_PATH = Path("models/customer_churn_model.pkl")


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

@st.cache_resource
def load_model():

    """
    Loads the saved machine-learning pipeline.

    The pipeline contains the preprocessing and trained
    Logistic Regression classifier used by the application.
    """

    return joblib.load(MODEL_PATH)


try:

    model = load_model()

except FileNotFoundError:

    st.error(
        "Model file not found. Please confirm that "
        "'customer_churn_model.pkl' is located inside "
        "the 'models' folder."
    )

    st.stop()


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

with st.sidebar:

    st.markdown(
        """
        <h2>📊 Churn Predictor</h2>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        """
        ### Project Information

        **Project:** Customer Churn Predictor

        **Field:** Artificial Intelligence / Machine Learning

        **Model:** Logistic Regression

        **Developer:** Chukwuka Uchenna
        """
    )

    st.markdown("---")

    st.markdown(
        """
        ### Model Performance

        **Accuracy:** 73.81%

        **Precision:** 50.43%

        **Recall:** 78.34%

        **F1-Score:** 61.36%

        **ROC-AUC:** 84.16%
        """
    )

    st.markdown("---")

    st.markdown(
        """
        ### Risk Classification

        🔴 **High:** ≥ 70%

        🟡 **Medium:** 40–69%

        🟢 **Low:** < 40%
        """
    )

    st.markdown("---")

    st.caption(
        "AI/ML Capstone Project"
    )


# ------------------------------------------------------------
# MAIN HEADER
# ------------------------------------------------------------

st.title("📊 Customer Churn Predictor")

st.markdown(
    "**Machine Learning-Based Customer Retention Risk Analysis**"
)

st.caption(
    "3MTT AI/ML Capstone Project • Developed by Chukwuka Uchenna"
)

st.divider()


st.markdown(
    """
    This application estimates the probability that a customer
    will discontinue a service based on demographic, service,
    contract and billing characteristics.
    """
)


st.divider()


# ============================================================
# CUSTOMER INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-heading">Customer Information</div>',
    unsafe_allow_html=True
)


tab_customer, tab_services, tab_account = st.tabs(
    [
        "👤 Customer Profile",
        "📡 Services",
        "💳 Account & Billing"
    ]
)


# ------------------------------------------------------------
# CUSTOMER PROFILE
# ------------------------------------------------------------

with tab_customer:

    col1, col2, col3 = st.columns(3)


    with col1:

        gender = st.selectbox(
            "Gender",
            [
                "Female",
                "Male"
            ]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [
                0,
                1
            ],
            format_func=lambda x:
                "Yes" if x == 1 else "No"
        )


    with col2:

        partner = st.selectbox(
            "Partner",
            [
                "Yes",
                "No"
            ]
        )

        dependents = st.selectbox(
            "Dependents",
            [
                "Yes",
                "No"
            ]
        )


    with col3:

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12,
            step=1
        )


# ------------------------------------------------------------
# SERVICES
# ------------------------------------------------------------

with tab_services:

    col1, col2 = st.columns(2)


    with col1:

        phone_service = st.selectbox(
            "Phone Service",
            [
                "Yes",
                "No"
            ]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "Yes",
                "No",
                "No phone service"
            ]
        )

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

        online_security = st.selectbox(
            "Online Security",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )


    with col2:

        online_backup = st.selectbox(
            "Online Backup",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        device_protection = st.selectbox(
            "Device Protection",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        tech_support = st.selectbox(
            "Technical Support",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )


# ------------------------------------------------------------
# ACCOUNT & BILLING
# ------------------------------------------------------------

with tab_account:

    col1, col2 = st.columns(2)


    with col1:

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            [
                "Yes",
                "No"
            ]
        )


    with col2:

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
            max_value=200.0,
            value=70.0,
            step=0.01
        )


    with col2:

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=10000.0,
            value=1000.0,
            step=0.01
        )


# ------------------------------------------------------------
# PREDICTION BUTTON
# ------------------------------------------------------------

st.write("")

st.markdown(
    '<div class="section-heading">Generate Prediction</div>',
    unsafe_allow_html=True
)


predict_button = st.button(
    "🚀  PREDICT CUSTOMER CHURN",
    type="primary",
    use_container_width=True
)


# ============================================================
# MACHINE LEARNING PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    customer_data = pd.DataFrame({

        "gender": [gender],

        "SeniorCitizen": [senior_citizen],

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
    })


    # --------------------------------------------------------
    # GENERATE MODEL PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(
        customer_data
    )[0]


    # --------------------------------------------------------
    # GENERATE CHURN PROBABILITY
    # --------------------------------------------------------

    churn_probability = model.predict_proba(
        customer_data
    )[0][1]


    churn_percentage = churn_probability * 100

    stay_percentage = 100 - churn_percentage


    # --------------------------------------------------------
    # DETERMINE RISK LEVEL
    # --------------------------------------------------------

    if churn_percentage >= 70:

        risk_level = "HIGH"

        risk_colour = "#dc2626"

        risk_message = (
            "The customer is classified as high risk based "
            "on the characteristics supplied to the trained "
            "machine-learning model. The predicted probability "
            "of churn is substantially higher than the "
            "probability of retention."
        )


    elif churn_percentage >= 40:

        risk_level = "MEDIUM"

        risk_colour = "#d97706"

        risk_message = (
            "The customer falls within the medium-risk range. "
            "The model identifies a meaningful possibility of "
            "churn, although the prediction is less decisive "
            "than a high-risk classification."
        )


    else:

        risk_level = "LOW"

        risk_colour = "#16a34a"

        risk_message = (
            "The customer is classified as low risk. The model "
            "estimates a higher probability of the customer "
            "remaining with the service than discontinuing it."
        )


    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-heading">Prediction Result</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # PROFESSIONAL RISK ASSESSMENT PANEL
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="risk-panel"
             style="border-left: 6px solid {risk_colour};">

            <div class="risk-header">

                <div class="risk-header-title">
                    CHURN RISK ASSESSMENT
                </div>

                <div class="risk-header-label">
                    LOGISTIC REGRESSION OUTPUT
                </div>

            </div>


            <div class="risk-level"
                 style="color: {risk_colour};">

                {risk_level} RISK

            </div>


            <div class="risk-percentage">

                {churn_percentage:.1f}%

            </div>


            <div class="risk-caption">

                Estimated probability of customer churn

            </div>


            <div class="risk-track">

                <div class="risk-fill"
                     style="
                     width: {churn_percentage}%;
                     background: {risk_colour};
                     ">
                </div>

            </div>


            <div class="risk-explanation">

                {risk_message}

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # PROBABILITY SUMMARY
    # ========================================================

    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            f"""
            <div class="summary-card">

                <div class="summary-label">
                    CHURN PROBABILITY
                </div>

                <div class="summary-value"
                     style="color: {risk_colour};">

                    {churn_percentage:.1f}%

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            f"""
            <div class="summary-card">

                <div class="summary-label">
                    RETENTION PROBABILITY
                </div>

                <div class="summary-value"
                     style="color: #16a34a;">

                    {stay_percentage:.1f}%

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # MODEL INSIGHTS
    # ========================================================

    st.write("")

    st.markdown(
        '<div class="section-heading">Model Insights</div>',
        unsafe_allow_html=True
    )


    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            """
            <div class="insight-card">

                <div class="insight-title">
                    ↑ Factors Associated With Higher Churn
                </div>

                <div class="insight-item">
                    Fiber optic internet service
                </div>

                <div class="insight-item">
                    Month-to-month contract
                </div>

                <div class="insight-item">
                    Electronic check payment method
                </div>

                <div class="insight-item">
                    Absence of online security
                </div>

                <div class="insight-item">
                    Absence of technical support
                </div>

                <div class="insight-item">
                    Streaming service usage
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            """
            <div class="insight-card">

                <div class="insight-title">
                    ↓ Factors Associated With Lower Churn
                </div>

                <div class="insight-item">
                    Longer customer tenure
                </div>

                <div class="insight-item">
                    Two-year contract
                </div>

                <div class="insight-item">
                    DSL internet service
                </div>

                <div class="insight-item">
                    Established customer relationship
                </div>

                <div class="insight-item">
                    Longer-term contractual commitment
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # PROBABILITY VISUALISATION
    # ========================================================

    st.write("")

    st.markdown(
        '<div class="section-heading">Probability Distribution</div>',
        unsafe_allow_html=True
    )


    probability_data = pd.DataFrame(
        {
            "Outcome": [
                "Retain Customer",
                "Customer Churn"
            ],

            "Probability": [
                stay_percentage,
                churn_percentage
            ]
        }
    )


    st.bar_chart(
        probability_data.set_index("Outcome")
    )


    # ========================================================
    # CUSTOMER INPUT SUMMARY
    # ========================================================

    st.write("")

    with st.expander("🔎 View Customer Input Data"):

        st.dataframe(
            customer_data.T.rename(
                columns={
                    0: "Value"
                }
            ),
            use_container_width=True
        )


    # ========================================================
    # MODEL INTERPRETATION
    # ========================================================

    st.write("")

    with st.expander("🧠 How Should This Prediction Be Interpreted?"):

        st.markdown(
            f"""
            ### Prediction

            The trained Logistic Regression model estimates a
            **{churn_percentage:.1f}% probability of churn** for
            the customer profile entered above.

            ### Important distinction

            This percentage represents a **model probability
            estimate**, not a guarantee that the customer will
            actually churn.

            The prediction is based on patterns learned from
            historical customer records.

            ### Model performance

            During evaluation, the selected model achieved:

            - **Accuracy:** 73.81%
            - **Precision:** 50.43%
            - **Recall:** 78.34%
            - **F1-score:** 61.36%
            - **ROC-AUC:** 84.16%

            The relatively strong recall means the model is able
            to identify a substantial proportion of customers
            who actually churned in the test dataset.

            Therefore, the system can serve as a decision-support
            tool for identifying customers who may require further
            retention analysis.
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    **Customer Churn Predictor**

    3MTT AI/ML Capstone Project  
    Developed by **Chukwuka Uchenna**
    """
)