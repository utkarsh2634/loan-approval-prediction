"""
Automated unit tests for Flask application routes, input handling, and HTTP status codes.
"""

import pytest
from app import app


@pytest.fixture
def client():
    """Configures and provides a Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_get_home_route(client):
    """Verifies that GET / returns 200 OK and contains project title."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Loan Approval Prediction" in response.data


def test_get_predict_route(client):
    """Verifies that GET /predict returns 200 OK and contains input form fields."""
    response = client.get("/predict")
    assert response.status_code == 200
    assert b"ApplicantIncome" in response.data
    assert b"Credit_History" in response.data


def test_get_dashboard_route(client):
    """Verifies that GET /dashboard returns 200 OK."""
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert b"Dashboard" in response.data or b"Metrics" in response.data


def test_get_about_route(client):
    """Verifies that GET /about returns 200 OK and explains Decision Trees."""
    response = client.get("/about")
    assert response.status_code == 200
    assert b"Decision Tree" in response.data


def test_get_health_route(client):
    """Verifies that GET /health returns 200 OK with JSON status payload."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.is_json
    data = response.get_json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_post_predict_valid_submission(client):
    """Verifies that POST /predict with valid form data succeeds and renders result."""
    payload = {
        "ApplicantIncome": "6500",
        "CoapplicantIncome": "2500",
        "LoanAmount": "130",
        "Loan_Amount_Term": "360",
        "Credit_History": "1",
        "Self_Employed": "No",
        "Married": "Yes",
        "Education": "Graduate",
        "Dependents": "0",
        "Property_Area": "Semiurban",
    }
    response = client.post("/predict", data=payload)
    assert response.status_code == 200
    # Must contain prediction label and disclaimer
    assert (b"Likely Approved" in response.data) or (b"Likely Rejected" in response.data)
    assert b"educational prediction" in response.data.lower()


def test_post_predict_invalid_submission(client):
    """Verifies that POST /predict with invalid data returns 400 and renders field errors without crash."""
    invalid_payload = {
        "ApplicantIncome": "-100",  # Illegal negative income
        "CoapplicantIncome": "invalid_number",
        "LoanAmount": "0",          # Must be > 0
        "Loan_Amount_Term": "360",
        "Credit_History": "1",
        "Self_Employed": "No",
        "Married": "Yes",
        "Education": "Graduate",
        "Dependents": "0",
        "Property_Area": "Semiurban",
    }
    response = client.post("/predict", data=invalid_payload)
    assert response.status_code == 400
    assert b"Applicant Income cannot be negative" in response.data


def test_custom_404_page(client):
    """Verifies that navigating to a non-existent URL returns 404 with custom error template."""
    response = client.get("/this-route-does-not-exist-at-all")
    assert response.status_code == 404
    assert b"Page Not Found" in response.data
    assert b"Error 404" in response.data


def test_post_predict_short_term_warning_and_normal_term(client):
    """
    Verifies that:
    1. POST /predict with a 6-month term returns 200, containing 'Likely Approved' or
       'Likely Rejected', educational disclaimer, and the short-term warning.
    2. POST /predict with a normal 360-month term returns 200 with no short-term warning.
    (Does not assert specific prediction outcomes)
    """
    base_data = {
        "ApplicantIncome": "6500",
        "CoapplicantIncome": "2500",
        "LoanAmount": "130",
        "Credit_History": "1",
        "Self_Employed": "No",
        "Married": "Yes",
        "Education": "Graduate",
        "Dependents": "0",
        "Property_Area": "Semiurban",
    }

    # 1. 6-Month Term Submission
    short_term_data = dict(base_data, Loan_Amount_Term="6", Loan_Amount_Term_Unit="Months")
    resp_short = client.post("/predict", data=short_term_data)
    assert resp_short.status_code == 200
    assert (b"Likely Approved" in resp_short.data) or (b"Likely Rejected" in resp_short.data)
    assert b"educational prediction" in resp_short.data.lower()
    assert b"Short loan terms are rare in the training data, so this prediction is less reliable." in resp_short.data

    # 2. Normal 360-Month Term Submission
    normal_term_data = dict(base_data, Loan_Amount_Term="360", Loan_Amount_Term_Unit="Months")
    resp_normal = client.post("/predict", data=normal_term_data)
    assert resp_normal.status_code == 200
    assert (b"Likely Approved" in resp_normal.data) or (b"Likely Rejected" in resp_normal.data)
    assert b"educational prediction" in resp_normal.data.lower()
    assert b"Short loan terms are rare in the training data" not in resp_normal.data


def test_post_predict_invalid_loan_terms_return_400_no_500(client):
    """
    Verifies that 0, negative, empty, non-numeric letters, and above-maximum loan terms
    submitted to POST /predict return HTTP 400 with a clear field error and no 500 error.
    """
    base_data = {
        "ApplicantIncome": "6500",
        "CoapplicantIncome": "2500",
        "LoanAmount": "130",
        "Credit_History": "1",
        "Self_Employed": "No",
        "Married": "Yes",
        "Education": "Graduate",
        "Dependents": "0",
        "Property_Area": "Semiurban",
    }

    invalid_terms = [
        ("0", "Months"),
        ("-12", "Months"),
        ("", "Months"),
        ("abc", "Months"),
        ("500", "Months"),
        ("45", "Years"),
    ]

    for term_val, term_unit in invalid_terms:
        payload = dict(base_data, Loan_Amount_Term=term_val, Loan_Amount_Term_Unit=term_unit)
        response = client.post("/predict", data=payload)
        assert response.status_code == 400, f"Expected 400 for term '{term_val}' {term_unit}, got {response.status_code}"
        assert b"Loan Term" in response.data

