"""
Streamlit Web Application for Loan Approval Prediction.
Section: 6B - Machine Learning Risk Assessment System

Run with:
    streamlit run streamlit_app.py
"""

import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    MODEL_FILE_PATH,
    METRICS_PATH,
    OUTPUTS_DIR,
    FEATURE_COLS,
    DISCLAIMER_TEXT,
)
from src.predict import load_model, make_prediction, validate_input

# Configure Streamlit page
st.set_page_config(
    page_title="Loan Approval Prediction | Sec 6B",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_cached_model():
    """Loads and caches model pipeline artifact and candidate models across Streamlit sessions."""
    if not MODEL_FILE_PATH.exists():
        return None, {}, {}
    try:
        import joblib
        raw = joblib.load(MODEL_FILE_PATH)
        if isinstance(raw, dict):
            best_pipe = raw.get("pipeline")
            candidates = raw.get("candidate_pipelines", {})
            metadata = raw.get("metadata", {})
            return best_pipe, candidates, metadata
        return raw, {}, {}
    except Exception as e:
        st.error(f"Error loading model artifact: {e}")
        return None, {}, {}


pipeline, candidate_pipelines, metadata = get_cached_model()
is_synthetic = metadata.get("is_synthetic", False)

# Main Title & Header
st.title("🏦 Loan Approval Prediction System")
st.caption("B.Tech CSE &bull; Section 6B &bull; Machine Learning Risk Assessment System")

if is_synthetic:
    st.warning("⚠️ **SYNTHETIC DATA — for demonstration only**. The model was trained on demonstration data.")

# Model availability check
if pipeline is None:
    st.error(
        "❌ **Model Artifact Not Found!**\n\n"
        "The trained pipeline was not found at `models/loan_model.joblib`. "
        "Please run `python train_model.py` in your terminal to train and persist the model first."
    )
    st.stop()

# Preset test cases for fast verification
PRESET_CASES = {
    "Custom / Manual Entry": None,
    "1. High Income + Good Credit (Strong Applicant)": {
        "ApplicantIncome": 25000, "CoapplicantIncome": 5000, "LoanAmount": 120, "Loan_Amount_Term": 360,
        "Credit_History": 1, "Gender": "Male", "Married": "Yes", "Dependents": "0", "Education": "Graduate",
        "Self_Employed": "No", "Property_Area": "Urban"
    },
    "2. Low Income + Good Credit (Marginal Repayment)": {
        "ApplicantIncome": 1500, "CoapplicantIncome": 0, "LoanAmount": 100, "Loan_Amount_Term": 360,
        "Credit_History": 1, "Gender": "Male", "Married": "No", "Dependents": "0", "Education": "Graduate",
        "Self_Employed": "No", "Property_Area": "Urban"
    },
    "3. High Income + Poor Credit (High Income Delinquent)": {
        "ApplicantIncome": 25000, "CoapplicantIncome": 5000, "LoanAmount": 120, "Loan_Amount_Term": 360,
        "Credit_History": 0, "Gender": "Male", "Married": "Yes", "Dependents": "0", "Education": "Graduate",
        "Self_Employed": "No", "Property_Area": "Urban"
    },
    "4. Low Income + Poor Credit (High Risk Delinquent)": {
        "ApplicantIncome": 1500, "CoapplicantIncome": 0, "LoanAmount": 250, "Loan_Amount_Term": 360,
        "Credit_History": 0, "Gender": "Male", "Married": "No", "Dependents": "2", "Education": "Not Graduate",
        "Self_Employed": "No", "Property_Area": "Rural"
    },
    "5. High Loan + Low Income (Overleveraged)": {
        "ApplicantIncome": 2000, "CoapplicantIncome": 0, "LoanAmount": 500, "Loan_Amount_Term": 360,
        "Credit_History": 1, "Gender": "Female", "Married": "No", "Dependents": "1", "Education": "Graduate",
        "Self_Employed": "No", "Property_Area": "Rural"
    },
    "6. Low Loan + High Income (Conservative Safe)": {
        "ApplicantIncome": 12000, "CoapplicantIncome": 4000, "LoanAmount": 60, "Loan_Amount_Term": 180,
        "Credit_History": 1, "Gender": "Female", "Married": "Yes", "Dependents": "0", "Education": "Graduate",
        "Self_Employed": "No", "Property_Area": "Semiurban"
    },
    "7. Short Term (120 Months) vs Long Term": {
        "ApplicantIncome": 4500, "CoapplicantIncome": 1500, "LoanAmount": 150, "Loan_Amount_Term": 120,
        "Credit_History": 1, "Gender": "Male", "Married": "Yes", "Dependents": "1", "Education": "Graduate",
        "Self_Employed": "No", "Property_Area": "Semiurban"
    },
    "8. Rural Property Area": {
        "ApplicantIncome": 4500, "CoapplicantIncome": 1500, "LoanAmount": 150, "Loan_Amount_Term": 360,
        "Credit_History": 1, "Gender": "Male", "Married": "Yes", "Dependents": "1", "Education": "Graduate",
        "Self_Employed": "No", "Property_Area": "Rural"
    },
}

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Evaluation Settings")

    # Algorithm Selector
    available_models = list(candidate_pipelines.keys()) if candidate_pipelines else ["Random Forest"]
    best_algo = metadata.get("best_algorithm", "Random Forest")
    default_idx = available_models.index(best_algo) if best_algo in available_models else 0

    selected_model_name = st.selectbox(
        "Select Machine Learning Model:",
        options=available_models,
        index=default_idx,
        help="Compare predictions across Random Forest, Decision Tree, and Logistic Regression.",
    )
    active_pipeline = candidate_pipelines.get(selected_model_name, pipeline)

    st.markdown("---")
    st.subheader("🧪 Quick Test Presets")
    st.write("Select a scenario to automatically test different borrower risk profiles:")
    selected_preset_key = st.selectbox(
        "Choose Applicant Scenario:",
        options=list(PRESET_CASES.keys()),
        index=0,
    )
    preset_data = PRESET_CASES[selected_preset_key]

    st.markdown("---")
    st.info(f"🛡️ **Educational Notice:**\n{DISCLAIMER_TEXT}")

# Initialize session defaults from preset or standard values
def get_val(key, default):
    if preset_data and key in preset_data:
        return preset_data[key]
    return default

# Tab Navigation
tab_predict, tab_explain, tab_metrics, tab_eda = st.tabs(
    ["🔮 Loan Evaluation", "🔍 Feature Importance & Explainability", "📊 Model Comparison & Diagnostics", "📈 Exploratory Data Analysis"]
)

with tab_predict:
    st.subheader("Applicant Details (All 11 Input Features)")
    st.write(
        "Provide applicant financial and demographic parameters. The model computes underwriting "
        "ratios (Total Household Income, Estimated Monthly Payment EMI, and Debt-to-Income DTI ratio) "
        "to evaluate the application."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 💰 Financial & Loan Details")
        applicant_income = st.number_input(
            "Applicant Monthly Income ($)",
            min_value=0,
            max_value=1_000_000,
            value=int(get_val("ApplicantIncome", 5500)),
            step=250,
            help="Primary monthly income in dollars",
            key="inp_app_inc",
        )
        coapplicant_income = st.number_input(
            "Co-applicant Monthly Income ($)",
            min_value=0,
            max_value=1_000_000,
            value=int(get_val("CoapplicantIncome", 1500)),
            step=250,
            help="Secondary household income in dollars",
            key="inp_coapp_inc",
        )
        loan_amount = st.number_input(
            "Loan Amount ($ in Thousands)",
            min_value=1,
            max_value=10_000,
            value=int(get_val("LoanAmount", 130)),
            step=10,
            help="Requested loan amount in thousands (e.g. 130 = $130,000)",
            key="inp_loan_amt",
        )
        loan_term = st.selectbox(
            "Loan Term (Months)",
            options=[120, 180, 240, 360, 480],
            index=[120, 180, 240, 360, 480].index(int(get_val("Loan_Amount_Term", 360))),
            format_func=lambda x: f"{x} Months ({x//12} Years)",
            help="Amortization duration in months",
            key="inp_term",
        )
        credit_history = st.selectbox(
            "Credit History Guidelines Met?",
            options=[1, 0],
            index=0 if int(get_val("Credit_History", 1)) == 1 else 1,
            format_func=lambda x: "Yes (Positive Credit Record = 1.0)" if x == 1 else "No (Delinquent / None = 0.0)",
            help="Credit bureau rating record",
            key="inp_credit",
        )

    with col2:
        st.markdown("#### 👤 Demographics & Profile")
        gender = st.selectbox(
            "Gender",
            options=["Male", "Female"],
            index=0 if get_val("Gender", "Male") == "Male" else 1,
            key="inp_gender",
        )
        married = st.selectbox(
            "Marital Status",
            options=["Yes", "No"],
            index=0 if get_val("Married", "Yes") == "Yes" else 1,
            key="inp_married",
        )
        dependents = st.selectbox(
            "Number of Dependents",
            options=["0", "1", "2", "3+"],
            index=["0", "1", "2", "3+"].index(str(get_val("Dependents", "0"))),
            key="inp_dep",
        )
        education = st.selectbox(
            "Education Level",
            options=["Graduate", "Not Graduate"],
            index=0 if get_val("Education", "Graduate") == "Graduate" else 1,
            key="inp_edu",
        )
        self_employed = st.selectbox(
            "Self Employed?",
            options=["No", "Yes"],
            index=0 if get_val("Self_Employed", "No") == "No" else 1,
            key="inp_self",
        )
        property_area = st.selectbox(
            "Property Area",
            options=["Semiurban", "Urban", "Rural"],
            index=["Semiurban", "Urban", "Rural"].index(get_val("Property_Area", "Semiurban")),
            key="inp_prop",
        )

    # Real-time applicant summary indicators
    st.markdown("---")
    tot_inc = applicant_income + coapplicant_income
    monthly_emi = (loan_amount * 1000.0) / max(loan_term, 1)
    dti = (monthly_emi / max(tot_inc, 1)) * 100.0
    lti = (loan_amount * 1000.0) / max(tot_inc * 12.0, 1.0)

    u_col1, u_col2, u_col3, u_col4 = st.columns(4)
    u_col1.metric("Household Monthly Income", f"${tot_inc:,.0f}")
    u_col2.metric("Requested Loan", f"${loan_amount * 1000:,.0f}")
    u_col3.metric("Estimated Monthly EMI", f"${monthly_emi:,.2f}")
    u_col4.metric(
        "Debt-to-Income (DTI)",
        f"{dti:.1f}%",
        delta="Safe (< 35%)" if dti <= 35 else ("Manageable (35-50%)" if dti <= 50 else "High Risk (> 50%)"),
        delta_color="normal" if dti <= 50 else "inverse",
    )

    predict_btn = st.button("🔮 Evaluate Loan Application", type="primary", use_container_width=True)

    if predict_btn or preset_data is not None:
        input_data = {
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": str(credit_history),
            "Gender": gender,
            "Married": married,
            "Education": education,
            "Dependents": dependents,
            "Self_Employed": self_employed,
            "Property_Area": property_area,
        }

        is_valid, errors, cleaned = validate_input(input_data)
        if not is_valid:
            for field, err in errors.items():
                st.error(f"**{field}:** {err}")
        else:
            result = make_prediction(active_pipeline, cleaned)
            st.divider()

            res_col1, res_col2 = st.columns([1, 1.4])
            with res_col1:
                st.markdown(f"### Evaluation Result ({selected_model_name})")
                if result["prediction"] == 1:
                    st.success(f"## ✅ {result['result_label']}")
                    m_p1, m_p2 = st.columns(2)
                    m_p1.metric("Approval Probability", f"{result['approval_probability']}%")
                    m_p2.metric("Rejection Risk", f"{result['rejection_probability']}%")
                else:
                    st.error(f"## ❌ {result['result_label']}")
                    m_p1, m_p2 = st.columns(2)
                    m_p1.metric("Approval Probability", f"{result['approval_probability']}%")
                    m_p2.metric("Rejection Risk", f"{result['rejection_probability']}%")

                st.progress(result["approval_probability"] / 100.0)
                st.caption(f"Estimated by fitted {selected_model_name} model probabilities.")

            with res_col2:
                st.markdown("### 💡 Decision Rationale & Contributing Factors")
                st.info(result["explanation"])

                f_col1, f_col2 = st.columns(2)
                with f_col1:
                    st.markdown("**Positive Contributing Factors:**")
                    if result["positive_factors"]:
                        for factor in result["positive_factors"]:
                            st.markdown(f"- ✅ {factor}")
                    else:
                        st.markdown("- None identified.")
                with f_col2:
                    st.markdown("**Risk Factors Identified:**")
                    if result["risk_factors"]:
                        for factor in result["risk_factors"]:
                            st.markdown(f"- ⚠️ {factor}")
                    else:
                        st.markdown("- No critical risk factors identified.")

            if result["term_warning"]:
                st.warning(f"⚠️ **Notice on Loan Term:** {result['term_warning']}")

with tab_explain:
    st.subheader("Feature Importance & Explainability")
    st.write(
        "Machine learning models evaluate all input attributes. In lending credit scoring, "
        "Credit History is a leading predictor, but debt-to-income, loan amount, household income, "
        "and property area actively shape the final decision boundary."
    )

    if (OUTPUTS_DIR / "feature_importance.png").exists():
        st.image(str(OUTPUTS_DIR / "feature_importance.png"), caption="Top Feature Importances from Fitted Model")

    st.markdown("---")
    st.subheader("Why Did the Original Decision Tree Only Use Credit History?")
    st.markdown(
        """
        - **Greedy Split Collapse:** In the original setup, GridSearchCV restricted `max_depth` to 2.
          A depth-2 tree has at most 3 split nodes. Because Credit History had the single largest initial Gini reduction,
          the root split on Credit History, leaving no splits for Applicant Income, Coapplicant Income, Education, Property Area, or Dependents.
        - **Omission of Gender:** Gender was completely dropped from the training set and the web UI.
        - **Absence of Underwriting Financial Ratios:** Raw ApplicantIncome and LoanAmount in isolation didn't capture the borrower's debt-to-income ratio (EMI / Total Income).
        - **The Fix:** We implemented `LoanFeatureEngineer` (computing Total Income, Monthly EMI, DTI, and Loan-to-Income),
          integrated all 11 features, and evaluated Random Forest (which uses feature subsampling across trees to evaluate all inputs),
          producing accurate, well-calibrated probabilities.
        """
    )

with tab_metrics:
    st.subheader("Model Evaluation & Algorithm Comparison")
    metrics_data = {}
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

    # Metric Highlights
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Best Model Test Accuracy", f"{metrics_data.get('test_accuracy', 0)*100:.2f}%")
    mcol2.metric("Best Model ROC-AUC", f"{metrics_data.get('test_roc_auc', 0):.4f}")
    mcol3.metric("5-Fold CV Accuracy", f"{metrics_data.get('best_cv_accuracy', 0)*100:.2f}%")
    mcol4.metric("Majority Baseline Acc", f"{metrics_data.get('baseline_accuracy', 0)*100:.2f}%")

    st.markdown("---")
    m_col1, m_col2 = st.columns(2)

    with m_col1:
        if (OUTPUTS_DIR / "model_comparison.png").exists():
            st.image(str(OUTPUTS_DIR / "model_comparison.png"), caption="Model Comparison (DT vs RF vs Logistic Regression)")
        if (OUTPUTS_DIR / "roc_curves.png").exists():
            st.image(str(OUTPUTS_DIR / "roc_curves.png"), caption="ROC Curves Comparison on Test Set")

    with m_col2:
        if (OUTPUTS_DIR / "confusion_matrix.png").exists():
            st.image(str(OUTPUTS_DIR / "confusion_matrix.png"), caption="Confusion Matrix on Unseen Test Set")
        if (OUTPUTS_DIR / "decision_tree.png").exists():
            st.image(str(OUTPUTS_DIR / "decision_tree.png"), caption="Decision Tree Architecture (First 3 Split Levels)")

    # Model comparison table
    if "model_comparison" in metrics_data:
        st.markdown("### Detailed Algorithm Performance Comparison")
        comp_records = []
        for model_k, scores in metrics_data["model_comparison"].items():
            comp_records.append({
                "Algorithm": model_k,
                "Test Accuracy": f"{scores.get('accuracy', 0)*100:.2f}%",
                "Test F1-Score": f"{scores.get('f1', 0)*100:.2f}%",
                "Test ROC-AUC": f"{scores.get('roc_auc', 0):.4f}",
                "5-Fold CV Accuracy": f"{scores.get('cv_accuracy', 0)*100:.2f}%",
            })
        st.table(pd.DataFrame(comp_records))

with tab_eda:
    st.subheader("Exploratory Data Analysis (EDA) Insights")
    st.write("Visualizations generated directly from the training dataset:")

    eda_col1, eda_col2 = st.columns(2)
    with eda_col1:
        if (OUTPUTS_DIR / "loan_status_distribution.png").exists():
            st.image(str(OUTPUTS_DIR / "loan_status_distribution.png"), caption="1. Loan Approval Status Distribution")
        if (OUTPUTS_DIR / "credit_history_vs_loan_status.png").exists():
            st.image(str(OUTPUTS_DIR / "credit_history_vs_loan_status.png"), caption="2. Credit History vs Approval Status")
        if (OUTPUTS_DIR / "correlation_heatmap.png").exists():
            st.image(str(OUTPUTS_DIR / "correlation_heatmap.png"), caption="3. Correlation Heatmap")

    with eda_col2:
        if (OUTPUTS_DIR / "income_distribution.png").exists():
            st.image(str(OUTPUTS_DIR / "income_distribution.png"), caption="4. Applicant Income Distribution")
        if (OUTPUTS_DIR / "loan_amount_distribution.png").exists():
            st.image(str(OUTPUTS_DIR / "loan_amount_distribution.png"), caption="5. Loan Amount Distribution")
        if (OUTPUTS_DIR / "demographic_approval_rates.png").exists():
            st.image(str(OUTPUTS_DIR / "demographic_approval_rates.png"), caption="6. Demographic Approval Rates")
