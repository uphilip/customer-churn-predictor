# ============================================================
# CUSTOMER CHURN PREDICTOR
# 3MTT AI/ML CAPSTONE PROJECT
#
# Developed by: Chukwuka Uchenna
#
# Project Description:
# This application uses a trained Logistic Regression machine
# learning model to estimate the probability that a customer
# will churn based on demographic, service, contract and
# billing characteristics.
#
# The application is deployed using Streamlit.
# ============================================================


# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================
# This must be called before most other Streamlit commands.
# It controls the browser tab title, icon and page layout.

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 3. APPLICATION CONSTANTS
# ============================================================

# Path to the saved machine-learning model.
# Path(__file__).parent ensures the application works both
# locally and when deployed through Streamlit Community Cloud.

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "customer_churn_model.pkl"


# ============================================================
# 4. LOAD THE TRAINED MACHINE-LEARNING MODEL
# ============================================================
# The model was trained previously in the Jupyter notebook.
# It is loaded here so that the Streamlit application can make
# predictions without retraining the model every time the app
# starts.

@st.cache_resource
def load_model():
    """
    Load the trained machine-learning model from disk.

    Streamlit's cache_resource prevents the model from being
    loaded repeatedly every time a user interacts with the app.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file was not found at: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# Try to load the model.
try:
    model = load_model()

except Exception as error:
    st.error("The trained machine-learning model could not be loaded.")

    st.info(
        "Please confirm that the file "
        "'customer_churn_model.pkl' exists inside the models folder."
    )

    st.code(str(error))

    st.stop()


# ============================================================
# 5. APPLICATION HEADER
# ============================================================

st.title("📊 Customer Churn Predictor")

st.subheader(
    "Machine Learning-Based Customer Retention Risk Analysis"
)

st.caption(
    "3MTT AI/ML Capstone Project • Developed by Chukwuka Uchenna"
)

st.divider()


# ============================================================
# 6. PROJECT INTRODUCTION
# ============================================================
# This section explains the purpose of the application to a
# person using the system for the first time.

with st.expander("ℹ️ About this project", expanded=False):

    st.write(
        """
        **Customer Churn Predictor** is a supervised machine-learning
        application designed to estimate the likelihood that a customer
        will leave a service.

        The system uses customer demographic, service, contract and
        billing characteristics as input variables. These characteristics
        are passed to a previously trained Logistic Regression model,
        which produces a churn probability.

        The prediction is intended to support customer retention analysis.
        It should be interpreted as a model-based risk estimate rather
        than a guarantee of future customer behaviour.
        """
    )


# ============================================================
# 7. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🎯 Prediction Settings")

    st.write(
        """
        Enter the customer's information below.

        The trained machine-learning model will use these characteristics
        to estimate the customer's probability of churn.
        """
    )

    st.divider()

    st.info(
        """
        **Model:** Logistic Regression

        **Task:** Binary Classification

        **Target:** Customer Churn

        **ROC-AUC:** 0.8416
        """
    )

    st.divider()

    st.caption("3MTT AI/ML Capstone Project")

    st.caption("Developed by Chukwuka Uchenna")


# ============================================================
# 8. CUSTOMER INFORMATION
# ============================================================

st.header("👤 Customer Information")

st.write(
    "Provide the customer's demographic and relationship information."
)

col1, col2, col3 = st.columns(3)


with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )


with col2:

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )


with col3:

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12,
        step=1
    )


st.divider()


# ============================================================
# 9. TELEPHONE AND INTERNET SERVICES
# ============================================================

st.header("📡 Service Information")

col1, col2, col3 = st.columns(3)


with col1:

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )


with col2:

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )


with col3:

    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )


st.divider()


# ============================================================
# 10. ADDITIONAL SERVICES
# ============================================================

st.header("🛠️ Additional Services")

col1, col2, col3 = st.columns(3)


with col1:

    tech_support = st.selectbox(
        "Technical Support",
        ["No", "Yes", "No internet service"]
    )


with col2:

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )


with col3:

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )


st.divider()


# ============================================================
# 11. CONTRACT AND BILLING INFORMATION
# ============================================================

st.header("💳 Contract & Billing Information")

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
        ["Yes", "No"]
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
        max_value=500.0,
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


st.divider()


# ============================================================
# 12. CREATE CUSTOMER DATAFRAME
# ============================================================
# The model expects the same feature structure used during
# training.
#
# customerID is deliberately excluded because it is an
# identifier rather than a meaningful predictive feature.
#
# TotalCharges is converted to a numerical value because it was
# originally stored as text in the raw dataset.

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
# 13. DISPLAY ENTERED CUSTOMER PROFILE
# ============================================================
# This gives the user a quick opportunity to review the
# information before running the prediction.

with st.expander("🔎 Review Customer Information", expanded=False):

    st.dataframe(
        customer_data,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 14. PREDICTION BUTTON
# ============================================================

st.header("🤖 Machine Learning Prediction")

st.write(
    "Click the button below to analyse the supplied customer profile."
)

predict_button = st.button(
    "🔍 Predict Customer Churn",
    type="primary",
    use_container_width=True
)


# ============================================================
# 15. RUN MODEL PREDICTION
# ============================================================

if predict_button:

    try:

        # --------------------------------------------------------
        # Generate the predicted class.
        # --------------------------------------------------------

        prediction = model.predict(customer_data)[0]


        # --------------------------------------------------------
        # Generate prediction probabilities.
        #
        # For binary classification, predict_proba returns:
        #
        # [probability of class 0, probability of class 1]
        #
        # We identify the "Yes" class from model.classes_ instead
        # of assuming that it is always in a specific position.
        # --------------------------------------------------------

        probabilities = model.predict_proba(customer_data)[0]

        classes = list(model.classes_)

        if "Yes" in classes:

            churn_index = classes.index("Yes")

        elif 1 in classes:

            churn_index = classes.index(1)

        else:

            # Fallback for a binary classifier where the positive
            # class is represented by the second class.

            churn_index = 1


        churn_probability = float(
            probabilities[churn_index]
        )

        retention_probability = 1.0 - churn_probability


        # --------------------------------------------------------
        # Convert probabilities to percentages.
        # --------------------------------------------------------

        churn_percentage = churn_probability * 100

        retention_percentage = retention_probability * 100


        # ========================================================
        # 16. DETERMINE RISK LEVEL
        # ========================================================
        #
        # These thresholds are presentation thresholds used to
        # translate the probability into an understandable risk
        # category.
        #
        # They do not retrain or modify the machine-learning model.
        # ========================================================

        if churn_probability >= 0.70:

            risk_level = "HIGH RISK"

            risk_message = (
                "The customer shows a high predicted likelihood of "
                "churning. Retention action may deserve priority."
            )

        elif churn_probability >= 0.40:

            risk_level = "MEDIUM RISK"

            risk_message = (
                "The customer shows a moderate predicted likelihood "
                "of churning. The customer profile may benefit from "
                "closer monitoring."
            )

        else:

            risk_level = "LOW RISK"

            risk_message = (
                "The customer shows a relatively low predicted "
                "likelihood of churning based on the supplied profile."
            )


        # ========================================================
        # 17. DISPLAY MAIN RESULT
        # ========================================================

        st.divider()

        st.header("📊 Churn Risk Assessment")

        # Use Streamlit's native components rather than manually
        # injecting HTML. This prevents raw HTML tags from appearing
        # in the browser.

        if risk_level == "HIGH RISK":

            st.error(
                f"🔴 {risk_level}"
            )

        elif risk_level == "MEDIUM RISK":

            st.warning(
                f"🟠 {risk_level}"
            )

        else:

            st.success(
                f"🟢 {risk_level}"
            )


        # --------------------------------------------------------
        # Main probability metric
        # --------------------------------------------------------

        st.metric(
            label="Estimated Probability of Customer Churn",
            value=f"{churn_percentage:.1f}%"
        )


        # --------------------------------------------------------
        # Probability progress bar
        # --------------------------------------------------------

        st.progress(
            min(max(churn_probability, 0.0), 1.0)
        )


        st.caption(
            "The probability represents the output of the trained "
            "Logistic Regression model."
        )


        # --------------------------------------------------------
        # Explanation
        # --------------------------------------------------------

        st.info(
            risk_message
        )


        # ========================================================
        # 18. PROBABILITY COMPARISON
        # ========================================================

        st.subheader("Probability Summary")

        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                label="🔴 Churn Probability",
                value=f"{churn_percentage:.1f}%"
            )


        with col2:

            st.metric(
                label="🟢 Retention Probability",
                value=f"{retention_percentage:.1f}%"
            )


        # ========================================================
        # 19. MODEL INSIGHTS
        # ========================================================
        # These insights explain common patterns identified from
        # the model's feature coefficients during model analysis.
        #
        # They are presented as general model-associated factors,
        # not as causal claims.
        # ========================================================

        st.subheader("🧠 Model Insights")

        col1, col2 = st.columns(2)


        with col1:

            st.markdown(
                """
                **↑ Factors Associated With Higher Churn**

                - Fiber optic internet service
                - Month-to-month contract
                - Electronic check payment method
                - Absence of online security
                - Absence of technical support
                - Streaming service usage
                """
            )


        with col2:

            st.markdown(
                """
                **↓ Factors Associated With Lower Churn**

                - Longer customer tenure
                - Two-year contract
                - DSL internet service
                - Established customer relationship
                - Longer-term contractual commitment
                """
            )


        # ========================================================
        # 20. INTERPRETATION NOTE
        # ========================================================

        st.subheader("📌 Interpretation")

        if churn_probability >= 0.70:

            st.write(
                f"""
                The model estimates a **{churn_percentage:.1f}%**
                probability of churn and a **{retention_percentage:.1f}%**
                probability of retention.

                Based on the application's predefined risk thresholds,
                this customer is classified as **high churn risk**.
                """
            )

        elif churn_probability >= 0.40:

            st.write(
                f"""
                The model estimates a **{churn_percentage:.1f}%**
                probability of churn and a **{retention_percentage:.1f}%**
                probability of retention.

                Based on the application's predefined risk thresholds,
                this customer is classified as **medium churn risk**.
                """
            )

        else:

            st.write(
                f"""
                The model estimates a **{churn_percentage:.1f}%**
                probability of churn and a **{retention_percentage:.1f}%**
                probability of retention.

                Based on the application's predefined risk thresholds,
                this customer is classified as **low churn risk**.
                """)


        # ========================================================
        # 21. TECHNICAL INFORMATION
        # ========================================================

        with st.expander("🔧 Technical Model Information"):

            st.write(
                """
                **Machine Learning Approach**

                The application uses supervised machine learning for
                binary customer churn classification.

                **Selected Model**

                Logistic Regression was selected because it achieved
                strong churn recall and ROC-AUC performance during
                model evaluation.

                **Evaluation Results**

                - Accuracy: 73.81%
                - Precision: 50.43%
                - Recall: 78.34%
                - F1-score: 61.36%
                - ROC-AUC: 84.16%

                **Important Interpretation**

                Accuracy was not used as the only model-selection
                criterion. Recall was particularly important because
                identifying customers who actually churn is an important
                objective of a customer retention system.
                """
            )


    # ============================================================
    # 22. HANDLE PREDICTION ERRORS
    # ============================================================

    except Exception as error:

        st.error(
            "The prediction could not be completed."
        )

        st.warning(
            "Please check that the input variables match the "
            "features used when the machine-learning model was trained."
        )

        # Technical error details are placed inside an expander
        # so ordinary users do not see unnecessary technical output.

        with st.expander("Technical error details"):

            st.code(
                str(error)
            )


# ============================================================
# 23. FOOTER
# ============================================================

st.divider()

st.caption(
    "Customer Churn Predictor • 3MTT AI/ML Capstone Project"
)

st.caption(
    "Developed by Chukwuka Uchenna"
)