"""
Automated unit tests for input validation and prediction helpers.
"""

import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.predict import make_prediction, validate_input
from src.preprocessing import build_pipeline


@pytest.fixture
def mock_pipeline():
    """Builds and fits a minimal mock pipeline for inference tests."""
    df = pd.DataFrame(
        [
            {
                "ApplicantIncome": 5000,
                "CoapplicantIncome": 1000,
                "LoanAmount": 120,
                "Loan_Amount_Term": 360,
                "Credit_History": "1",
                "Gender": "Male",
                "Self_Employed": "No",
                "Married": "Yes",
                "Education": "Graduate",
                "Dependents": "0",
                "Property_Area": "Semiurban",
            },
            {
                "ApplicantIncome": 2000,
                "CoapplicantIncome": 0,
                "LoanAmount": 250,
                "Loan_Amount_Term": 360,
                "Credit_History": "0",
                "Gender": "Female",
                "Self_Employed": "Yes",
                "Married": "No",
                "Education": "Not Graduate",
                "Dependents": "3",
                "Property_Area": "Rural",
            },
        ]
    )
    y = [1, 0]
    pipe = build_pipeline(classifier=DecisionTreeClassifier(max_depth=2, random_state=42))
    pipe.fit(df, y)
    return pipe


def test_validate_input_valid():
    """Verifies that valid form inputs are cleanly parsed without errors."""
    valid_data = {
        "ApplicantIncome": "6000",
        "CoapplicantIncome": "2000",
        "LoanAmount": "150",
        "Loan_Amount_Term": "360",
        "Credit_History": "1",
        "Self_Employed": "No",
        "Married": "Yes",
        "Education": "Graduate",
        "Dependents": "1",
        "Property_Area": "Urban",
    }
    is_valid, errors, cleaned = validate_input(valid_data)
    assert is_valid is True
    assert len(errors) == 0
    assert cleaned["ApplicantIncome"] == 6000.0
    assert cleaned["CoapplicantIncome"] == 2000.0
    assert cleaned["LoanAmount"] == 150.0
    assert cleaned["Credit_History"] == "1"


def test_validate_input_catches_invalid_values():
    """Verifies that server-side validation catches negative values, illegal categories, and strings in numbers."""
    invalid_data = {
        "ApplicantIncome": "-500",      # Negative income
        "CoapplicantIncome": "abc",     # Non-numeric
        "LoanAmount": "0",              # Zero amount
        "Loan_Amount_Term": "1000",     # Exceeds max months
        "Credit_History": "Invalid",    # Non-binary
        "Self_Employed": "Maybe",       # Illegal category
        "Married": "Unknown",           # Illegal category
        "Education": "HighSchool",      # Illegal category
        "Dependents": "10",             # Illegal dependents
        "Property_Area": "Mars",        # Illegal area
    }
    is_valid, errors, _ = validate_input(invalid_data)
    assert is_valid is False
    assert "ApplicantIncome" in errors
    assert "CoapplicantIncome" in errors
    assert "LoanAmount" in errors
    assert "Loan_Amount_Term" in errors
    assert "Credit_History" in errors
    assert "Self_Employed" in errors
    assert "Property_Area" in errors


def test_make_prediction_strong_and_weak_profiles(mock_pipeline):
    """
    Verifies that make_prediction returns the required labels, probabilities,
    and disclaimer without crashing on both strong and weak profiles.
    """
    strong_profile = {
        "ApplicantIncome": 8000,
        "CoapplicantIncome": 3000,
        "LoanAmount": 100,
        "Loan_Amount_Term": 360,
        "Credit_History": "1",
        "Self_Employed": "No",
        "Married": "Yes",
        "Education": "Graduate",
        "Dependents": "0",
        "Property_Area": "Semiurban",
    }

    weak_profile = {
        "ApplicantIncome": 1500,
        "CoapplicantIncome": 0,
        "LoanAmount": 300,
        "Loan_Amount_Term": 360,
        "Credit_History": "0",
        "Self_Employed": "Yes",
        "Married": "No",
        "Education": "Not Graduate",
        "Dependents": "3",
        "Property_Area": "Rural",
    }

    # Execute prediction on strong profile
    res_strong = make_prediction(mock_pipeline, strong_profile)
    assert res_strong["result_label"] in ("Likely Approved", "Likely Rejected")
    assert 0.0 <= res_strong["approval_probability"] <= 100.0
    assert 0.0 <= res_strong["rejection_probability"] <= 100.0
    assert "disclaimer" in res_strong
    assert "educational" in res_strong["disclaimer"].lower()

    # Execute prediction on weak profile
    res_weak = make_prediction(mock_pipeline, weak_profile)
    assert res_weak["result_label"] in ("Likely Approved", "Likely Rejected")
    assert "explanation" in res_weak


def test_loan_term_unit_conversions():
    """
    Verifies that terms in months and years (including decimals):
    6 months, 1 year, 1.5 years, 30 years, and 480 months
    convert cleanly to the correct whole-month values.
    """
    base_form = {
        "ApplicantIncome": "5000",
        "CoapplicantIncome": "0",
        "LoanAmount": "100",
        "Credit_History": "1",
        "Self_Employed": "No",
        "Married": "Yes",
        "Education": "Graduate",
        "Dependents": "0",
        "Property_Area": "Urban",
    }

    test_cases = [
        ("6", "Months", 6.0),
        ("1", "Years", 12.0),
        ("1.5", "Years", 18.0),
        ("30", "Years", 360.0),
        ("480", "Months", 480.0),
    ]

    for val, unit, expected_months in test_cases:
        form = dict(base_form, Loan_Amount_Term=val, Loan_Amount_Term_Unit=unit)
        is_valid, errors, cleaned = validate_input(form)
        assert is_valid is True, f"Failed for {val} {unit}: {errors}"
        assert cleaned["Loan_Amount_Term"] == expected_months, (
            f"Expected {expected_months} months for {val} {unit}, got {cleaned.get('Loan_Amount_Term')}"
        )


def test_loan_term_invalid_rejections():
    """
    Verifies that 0, negative, empty, non-numeric letters, and above-maximum values
    are rejected with a clear per-field error and do not raise unhandled exceptions.
    """
    base_form = {
        "ApplicantIncome": "5000",
        "CoapplicantIncome": "0",
        "LoanAmount": "100",
        "Credit_History": "1",
        "Self_Employed": "No",
        "Married": "Yes",
        "Education": "Graduate",
        "Dependents": "0",
        "Property_Area": "Urban",
    }

    invalid_term_cases = [
        ("0", "Months"),
        ("0", "Years"),
        ("-6", "Months"),
        ("-1", "Years"),
        ("", "Months"),
        ("   ", "Months"),
        ("abc", "Months"),
        ("five", "Years"),
        ("481", "Months"),
        ("41", "Years"),
        ("1000", "Months"),
    ]

    for val, unit in invalid_term_cases:
        form = dict(base_form, Loan_Amount_Term=val, Loan_Amount_Term_Unit=unit)
        is_valid, errors, cleaned = validate_input(form)
        assert is_valid is False, f"Expected validation failure for term '{val}' {unit}"
        assert "Loan_Amount_Term" in errors, f"Expected 'Loan_Amount_Term' error key for term '{val}' {unit}"
        assert len(errors["Loan_Amount_Term"]) > 0


def test_validate_all_11_features_including_gender():
    """Verifies that all 11 features are accepted, validated, and typed correctly."""
    form_11 = {
        "ApplicantIncome": "7500",
        "CoapplicantIncome": "2500",
        "LoanAmount": "180",
        "Loan_Amount_Term": "360",
        "Credit_History": "1",
        "Gender": "Female",
        "Married": "Yes",
        "Education": "Graduate",
        "Dependents": "2",
        "Self_Employed": "No",
        "Property_Area": "Semiurban",
    }
    is_valid, errors, cleaned = validate_input(form_11)
    assert is_valid is True
    assert len(errors) == 0
    assert cleaned["Gender"] == "Female"
    assert cleaned["ApplicantIncome"] == 7500.0
    assert cleaned["CoapplicantIncome"] == 2500.0
    assert cleaned["LoanAmount"] == 180.0
    assert cleaned["Loan_Amount_Term"] == 360.0
    assert cleaned["Credit_History"] == "1"
    assert cleaned["Property_Area"] == "Semiurban"


def test_model_responsiveness_to_financial_inputs():
    """
    Verifies that changing income, loan amount, and property area
    directly produces changes in model approval probability (the core issue reported by user).
    """
    from src.predict import load_model
    pipeline, _ = load_model()

    safe_applicant = {
        "ApplicantIncome": 15000,
        "CoapplicantIncome": 5000,
        "LoanAmount": 50,
        "Loan_Amount_Term": 360,
        "Credit_History": "1",
        "Gender": "Male",
        "Married": "Yes",
        "Education": "Graduate",
        "Dependents": "0",
        "Self_Employed": "No",
        "Property_Area": "Semiurban",
    }

    overleveraged_applicant = {
        "ApplicantIncome": 1800,
        "CoapplicantIncome": 0,
        "LoanAmount": 500,
        "Loan_Amount_Term": 360,
        "Credit_History": "1",
        "Gender": "Male",
        "Married": "Yes",
        "Education": "Graduate",
        "Dependents": "0",
        "Self_Employed": "No",
        "Property_Area": "Semiurban",
    }

    res_safe = make_prediction(pipeline, safe_applicant)
    res_overleveraged = make_prediction(pipeline, overleveraged_applicant)

    # Both have Credit_History = 1, but overleveraged applicant has low income and huge loan
    assert res_safe["approval_probability"] > res_overleveraged["approval_probability"], (
        f"Safe approval prob ({res_safe['approval_probability']}%) must exceed "
        f"overleveraged approval prob ({res_overleveraged['approval_probability']}%)!"
    )
    # Overleveraged should have higher rejection probability
    assert res_overleveraged["rejection_probability"] > res_safe["rejection_probability"]
    # Check financial metrics exist
    assert "financial_summary" in res_safe
    assert res_safe["financial_summary"]["total_income"] == 20000.0
    assert "positive_factors" in res_safe


