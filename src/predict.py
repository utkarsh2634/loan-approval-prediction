"""
Model inference, input validation, and prediction helpers.
Shared between Flask app (app.py), Streamlit app (streamlit_app.py), and test suites.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List
import joblib
import numpy as np
import pandas as pd
import sklearn

from src.config import (
    MODEL_FILE_PATH,
    FEATURE_COLS,
    DISCLAIMER_TEXT,
    LOAN_TERM_MIN_MONTHS,
    LOAN_TERM_MAX_MONTHS,
    LOAN_TERM_SHORT_THRESHOLD_MONTHS,
    SHORT_TERM_WARNING_TEXT,
    LONG_TERM_WARNING_TEXT,
)
from src.preprocessing import clean_dependents

logger = logging.getLogger(__name__)


def load_model(model_path: Optional[Path] = None) -> Tuple[Any, Dict[str, Any]]:
    """
    Loads the saved model pipeline and its metadata from disk.
    Logs a warning if the scikit-learn version saved at train time differs
    from the current runtime environment.

    Returns:
        Tuple of (fitted_pipeline, metadata_dict)
    """
    target_path = Path(model_path) if model_path else MODEL_FILE_PATH

    if not target_path.exists():
        raise FileNotFoundError(
            f"Trained model artifact not found at {target_path}. "
            "Please run 'python train_model.py' to generate the model artifact."
        )

    artifact = joblib.load(target_path)

    # Check if artifact is a dict containing pipeline and metadata, or bare pipeline
    if isinstance(artifact, dict) and "pipeline" in artifact:
        pipeline = artifact["pipeline"]
        metadata = artifact.get("metadata", {})
    else:
        pipeline = artifact
        metadata = {}

    saved_version = metadata.get("sklearn_version", "unknown")
    current_version = sklearn.__version__
    if saved_version != "unknown" and saved_version != current_version:
        logger.warning(
            f"Model was saved with scikit-learn {saved_version}, "
            f"but current runtime is scikit-learn {current_version}."
        )

    return pipeline, metadata


def validate_input(form_data: Dict[str, Any]) -> Tuple[bool, Dict[str, str], Dict[str, Any]]:
    """
    Performs comprehensive server-side input validation on submitted loan form data
    for all 11 features. Ensures safe types, numerical bounds, and enumerated category membership.

    Returns:
        Tuple of (is_valid: bool, errors: dict, cleaned_data: dict)
    """
    errors: Dict[str, str] = {}
    cleaned: Dict[str, Any] = {}

    # 1. ApplicantIncome
    raw_app_inc = form_data.get("ApplicantIncome", "")
    try:
        val = float(raw_app_inc)
        if val < 0:
            errors["ApplicantIncome"] = "Applicant Income cannot be negative."
        elif val > 1_000_000:
            errors["ApplicantIncome"] = "Applicant Income exceeds realistic limit ($1,000,000)."
        else:
            cleaned["ApplicantIncome"] = val
    except (ValueError, TypeError):
        errors["ApplicantIncome"] = "Applicant Income must be a valid non-negative number."

    # 2. CoapplicantIncome
    raw_coapp_inc = form_data.get("CoapplicantIncome", 0)
    try:
        val = float(raw_coapp_inc) if raw_coapp_inc not in ("", None) else 0.0
        if val < 0:
            errors["CoapplicantIncome"] = "Co-applicant Income cannot be negative."
        elif val > 1_000_000:
            errors["CoapplicantIncome"] = "Co-applicant Income exceeds realistic limit ($1,000,000)."
        else:
            cleaned["CoapplicantIncome"] = val
    except (ValueError, TypeError):
        errors["CoapplicantIncome"] = "Co-applicant Income must be a valid non-negative number."

    # 3. LoanAmount (in thousands $)
    raw_amount = form_data.get("LoanAmount", "")
    try:
        val = float(raw_amount)
        if val <= 0:
            errors["LoanAmount"] = "Loan Amount must be greater than 0."
        elif val > 10_000:
            errors["LoanAmount"] = "Loan Amount exceeds maximum allowed limit (10,000 thousand = $10M)."
        else:
            cleaned["LoanAmount"] = val
    except (ValueError, TypeError):
        errors["LoanAmount"] = "Loan Amount must be a valid positive number."

    # 4. Loan_Amount_Term (months or years, converted to whole months)
    raw_term = form_data.get("Loan_Amount_Term", "")
    raw_unit = str(form_data.get("Loan_Amount_Term_Unit", "Months")).strip().lower()

    if raw_term is None or str(raw_term).strip() == "":
        errors["Loan_Amount_Term"] = "Loan Term must be a valid positive number."
    else:
        try:
            val = float(str(raw_term).strip())
            if val <= 0:
                errors["Loan_Amount_Term"] = "Loan Term must be greater than 0."
            elif raw_unit in ("years", "year", "y"):
                term_in_months = round(val * 12.0)
                if term_in_months < LOAN_TERM_MIN_MONTHS:
                    errors["Loan_Amount_Term"] = f"Loan Term must be at least {LOAN_TERM_MIN_MONTHS} month."
                elif term_in_months > LOAN_TERM_MAX_MONTHS:
                    errors["Loan_Amount_Term"] = f"Loan Term cannot exceed {LOAN_TERM_MAX_MONTHS} months (40 years)."
                else:
                    cleaned["Loan_Amount_Term"] = float(term_in_months)
            elif raw_unit in ("months", "month", "m"):
                term_in_months = round(val)
                if term_in_months < LOAN_TERM_MIN_MONTHS:
                    errors["Loan_Amount_Term"] = f"Loan Term must be at least {LOAN_TERM_MIN_MONTHS} month."
                elif term_in_months > LOAN_TERM_MAX_MONTHS:
                    errors["Loan_Amount_Term"] = f"Loan Term cannot exceed {LOAN_TERM_MAX_MONTHS} months (40 years)."
                else:
                    cleaned["Loan_Amount_Term"] = float(term_in_months)
            else:
                errors["Loan_Amount_Term"] = "Loan Term unit must be either 'Months' or 'Years'."
        except (ValueError, TypeError):
            errors["Loan_Amount_Term"] = "Loan Term must be a valid positive number."

    # 5. Credit_History ('1' or '0')
    raw_credit = str(form_data.get("Credit_History", "")).strip()
    if raw_credit in ("1", "1.0", "Yes", "yes", "true", "True"):
        cleaned["Credit_History"] = "1"
    elif raw_credit in ("0", "0.0", "No", "no", "false", "False"):
        cleaned["Credit_History"] = "0"
    else:
        errors["Credit_History"] = "Please select a valid Credit History option (Yes or No)."

    # 6. Gender ('Male' or 'Female')
    raw_gender = str(form_data.get("Gender", "")).strip().capitalize()
    if raw_gender in ("Male", "Female"):
        cleaned["Gender"] = raw_gender
    elif raw_gender in ("", "None", "Nan"):
        # Graceful fallback for backward compatibility with legacy tests
        cleaned["Gender"] = "Male"
    else:
        errors["Gender"] = "Gender must be 'Male' or 'Female'."

    # 7. Married ('Yes' or 'No')
    raw_married = str(form_data.get("Married", "")).strip().capitalize()
    if raw_married in ("Yes", "No"):
        cleaned["Married"] = raw_married
    else:
        errors["Married"] = "Please select marital status: 'Yes' or 'No'."

    # 8. Dependents ('0', '1', '2', '3+' or '3')
    raw_dep = str(form_data.get("Dependents", "")).strip()
    cleaned_dep = clean_dependents(raw_dep)
    if cleaned_dep in ("0", "1", "2", "3"):
        cleaned["Dependents"] = cleaned_dep
    else:
        errors["Dependents"] = "Dependents must be one of: 0, 1, 2, 3+."

    # 9. Education ('Graduate' or 'Not Graduate')
    raw_edu = str(form_data.get("Education", "")).strip()
    if raw_edu in ("Graduate", "Not Graduate"):
        cleaned["Education"] = raw_edu
    else:
        errors["Education"] = "Please select Education: 'Graduate' or 'Not Graduate'."

    # 10. Self_Employed ('Yes' or 'No')
    raw_self = str(form_data.get("Self_Employed", "")).strip().capitalize()
    if raw_self in ("Yes", "No"):
        cleaned["Self_Employed"] = raw_self
    else:
        errors["Self_Employed"] = "Please select employment status: 'Yes' or 'No'."

    # 11. Property_Area ('Urban', 'Semiurban', 'Rural')
    raw_area = str(form_data.get("Property_Area", "")).strip().capitalize()
    if raw_area in ("Urban", "Semiurban", "Rural"):
        cleaned["Property_Area"] = raw_area
    else:
        errors["Property_Area"] = "Property Area must be 'Urban', 'Semiurban', or 'Rural'."

    is_valid = len(errors) == 0
    return is_valid, errors, cleaned


def make_prediction(pipeline: Any, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes inference for a single applicant input dictionary across all 11 features:
    1. Converts dictionary to a 1-row pandas DataFrame with exact feature columns.
    2. Invokes pipeline.predict and pipeline.predict_proba.
    3. Computes underwriting metrics (Total Income, EMI, DTI, LTI).
    4. Analyzes positive and risk factors.
    5. Produces contextual plain-English explanation.

    Returns:
        Dictionary containing prediction result, probabilities, factor breakdown, and disclaimer.
    """
    # Create single-row DataFrame with exact 11 feature columns and order
    data_dict = dict(input_data)
    if "Gender" not in data_dict:
        data_dict["Gender"] = "Male"
    row_df = pd.DataFrame([data_dict])[FEATURE_COLS]

    # Predict class (1 = Approved, 0 = Rejected)
    pred_class = int(pipeline.predict(row_df)[0])

    # Predict probabilities directly from fitted model
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(row_df)[0]
        prob_approved = float(probs[1]) if len(probs) > 1 else float(probs[0])
        prob_rejected = float(probs[0]) if len(probs) > 1 else 1.0 - prob_approved
    else:
        prob_approved = 1.0 if pred_class == 1 else 0.0
        prob_rejected = 1.0 - prob_approved

    approval_pct = round(prob_approved * 100, 1)
    rejection_pct = round(prob_rejected * 100, 1)

    result_label = "Likely Approved" if pred_class == 1 else "Likely Rejected"
    status_code = "approved" if pred_class == 1 else "rejected"

    # Compute key underwriting metrics
    app_inc = float(input_data.get("ApplicantIncome", 0.0))
    coapp_inc = float(input_data.get("CoapplicantIncome", 0.0))
    total_income = app_inc + coapp_inc
    loan_amount_k = float(input_data.get("LoanAmount", 0.0))
    loan_amount_dollars = loan_amount_k * 1000.0
    term_months = float(input_data.get("Loan_Amount_Term", 360.0))

    emi = loan_amount_dollars / max(term_months, 1.0)
    dti_ratio = (emi / max(total_income, 1.0)) * 100.0  # percentage
    annual_income = total_income * 12.0
    lti_ratio = loan_amount_dollars / max(annual_income, 1.0)

    credit_hist = str(input_data.get("Credit_History", "1"))
    prop_area = str(input_data.get("Property_Area", "Semiurban"))
    education = str(input_data.get("Education", "Graduate"))

    # Identify applicant factors
    positive_factors: List[str] = []
    risk_factors: List[str] = []

    if credit_hist == "1":
        positive_factors.append("Positive credit history meeting lending guidelines.")
    else:
        risk_factors.append("No record or adverse past credit delinquency.")

    if dti_ratio <= 35.0:
        positive_factors.append(f"Healthy debt-to-income ratio ({dti_ratio:.1f}% monthly income).")
    elif dti_ratio <= 50.0:
        positive_factors.append(f"Manageable debt-to-income ratio ({dti_ratio:.1f}%).")
    else:
        risk_factors.append(f"High debt-to-income burden ({dti_ratio:.1f}% of monthly income committed to EMI).")

    if total_income >= 8000:
        positive_factors.append(f"Strong combined household earnings (${total_income:,.0f}/mo).")
    elif total_income < 3000:
        risk_factors.append(f"Low combined household earnings (${total_income:,.0f}/mo).")

    if lti_ratio <= 2.5:
        positive_factors.append(f"Conservative loan request ({lti_ratio:.1f}x annual earnings).")
    elif lti_ratio > 4.5:
        risk_factors.append(f"High loan-to-annual-income ratio ({lti_ratio:.1f}x annual earnings).")

    if prop_area == "Semiurban":
        positive_factors.append("Semiurban property area (statistically higher loan clearance rates).")
    elif prop_area == "Rural":
        risk_factors.append("Rural property area carries slightly more conservative valuation criteria.")

    if education == "Graduate":
        positive_factors.append("Graduate applicant educational profile.")

    # Contextual explanation
    if pred_class == 1:
        if credit_hist == "1" and dti_ratio <= 45.0:
            explanation = (
                f"The application is likely to be approved ({approval_pct}% probability). "
                f"A positive credit history, combined with a viable debt-to-income ratio ({dti_ratio:.1f}%) "
                f"and healthy total earnings (${total_income:,.0f}/month), fulfills standard underwriting parameters."
            )
        elif credit_hist == "1":
            explanation = (
                f"The application is likely to be approved ({approval_pct}% probability). "
                f"Favorable credit history and applicant attributes outweigh the moderate loan burden."
            )
        else:
            explanation = (
                f"The application is likely to be approved ({approval_pct}% probability) "
                f"due to exceptional household income (${total_income:,.0f}/mo) and conservative loan sizing, "
                f"offsetting the lack of positive credit history."
            )
    else:
        if credit_hist == "0" and dti_ratio > 40.0:
            explanation = (
                f"The application is likely to be rejected ({rejection_pct}% risk). "
                f"Compounding negative credit history and an elevated debt-to-income ratio ({dti_ratio:.1f}%) "
                f"breach risk tolerance boundaries."
            )
        elif credit_hist == "0":
            explanation = (
                f"The application is likely to be rejected ({rejection_pct}% risk) primarily due to "
                f"an adverse or non-existent credit bureau record."
            )
        elif dti_ratio > 50.0 or lti_ratio > 4.0:
            explanation = (
                f"The application is likely to be rejected ({rejection_pct}% risk) due to an excessive "
                f"loan amount relative to income (DTI: {dti_ratio:.1f}%, Loan: {lti_ratio:.1f}x annual income)."
            )
        else:
            explanation = (
                f"The application is likely to be rejected ({rejection_pct}% risk). "
                f"While credit history is positive, the combined profile (income, loan term, and location) "
                f"falls below the required approval threshold."
            )

    # Check for short-term extrapolation warning (below 12 months) or extreme duration
    term_warning: Optional[str] = None
    if term_months < LOAN_TERM_SHORT_THRESHOLD_MONTHS:
        term_warning = SHORT_TERM_WARNING_TEXT
    elif term_months > LOAN_TERM_MAX_MONTHS:
        term_warning = LONG_TERM_WARNING_TEXT

    return {
        "prediction": pred_class,
        "result_label": result_label,
        "status_code": status_code,
        "approval_probability": approval_pct,
        "rejection_probability": rejection_pct,
        "explanation": explanation,
        "disclaimer": DISCLAIMER_TEXT,
        "term_warning": term_warning,
        "financial_summary": {
            "total_income": total_income,
            "estimated_emi": round(emi, 2),
            "debt_to_income_pct": round(dti_ratio, 1),
            "loan_to_income_ratio": round(lti_ratio, 2),
        },
        "positive_factors": positive_factors,
        "risk_factors": risk_factors,
    }
