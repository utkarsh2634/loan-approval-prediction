"""
Flask web application for Loan Approval Prediction.
Section: 6B - Educational Machine Learning Project

Provides:
- GET /         : Home page with project overview
- GET /predict  : Loan evaluation input form
- POST /predict : Server-side input validation and prediction
- GET /dashboard: Model metrics & EDA visual diagnostic dashboard
- GET /about    : Methodology, Decision Tree theory, and ethical guidelines
- GET /health   : Health-check endpoint for Render deployment monitoring
"""

import json
import logging
import os
from pathlib import Path
import sys
from typing import Dict, Any, Optional

from flask import (
    Flask,
    abort,
    jsonify,
    render_template,
    request,
    send_from_directory,
)

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    MODEL_FILE_PATH,
    METRICS_PATH,
    OUTPUTS_DIR,
    STATIC_DIR,
    TEMPLATES_DIR,
    TRAIN_DATA_PATH,
)
from src.predict import load_model, make_prediction, validate_input

# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(
    __name__,
    static_folder=str(STATIC_DIR),
    template_folder=str(TEMPLATES_DIR),
)

# Application security configurations
app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY", "loan-approval-prediction-sec6b-secret-key-2026"
)

# Global model state cache (loaded once at startup)
MODEL_PIPELINE = None
MODEL_METADATA: Dict[str, Any] = {}
MODEL_LOADED = False


def init_model() -> None:
    """Loads the trained pipeline artifact into memory at application startup."""
    global MODEL_PIPELINE, MODEL_METADATA, MODEL_LOADED
    try:
        if MODEL_FILE_PATH.exists():
            MODEL_PIPELINE, MODEL_METADATA = load_model(MODEL_FILE_PATH)
            MODEL_LOADED = True
            logger.info("Successfully loaded trained model pipeline into memory.")
        else:
            logger.warning(
                f"Model artifact not found at {MODEL_FILE_PATH}. "
                "The app will start, but predictions require running train_model.py."
            )
            # As a convenience safety net: if train.csv exists, train automatically once
            if TRAIN_DATA_PATH.exists():
                logger.info("data/train.csv found. Initiating automatic one-time training...")
                try:
                    from train_model import train_and_evaluate
                    train_and_evaluate()
                    MODEL_PIPELINE, MODEL_METADATA = load_model(MODEL_FILE_PATH)
                    MODEL_LOADED = True
                    logger.info("Automatic one-time training completed successfully.")
                except Exception as train_err:
                    logger.error(f"Automatic one-time training failed: {train_err}")
    except Exception as e:
        logger.error(f"Error loading model artifact: {e}")
        MODEL_PIPELINE = None
        MODEL_METADATA = {}
        MODEL_LOADED = False


# Load model at import time so Gunicorn has the model ready across workers
init_model()


@app.context_processor
def inject_global_template_vars():
    """Injects helpful status flags into all rendered Jinja2 templates."""
    is_synth = MODEL_METADATA.get("is_synthetic", False)
    return {
        "model_loaded": MODEL_LOADED,
        "is_synthetic": is_synth,
    }


@app.route("/", methods=["GET"])
def index():
    """Home page route providing project background and feature overviews."""
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    Loan evaluation route:
    - GET  : Renders the applicant input form.
    - POST : Validates inputs, executes Decision Tree inference, and renders results.
    """
    global MODEL_PIPELINE, MODEL_LOADED

    # Ensure model is initialized if possible
    if not MODEL_LOADED and MODEL_FILE_PATH.exists():
        init_model()

    if request.method == "GET":
        return render_template("predict.html", form_data={}, errors={})

    # Handle POST submission
    form_data = request.form.to_dict()

    # 1. Server-side validation
    is_valid, errors, cleaned_data = validate_input(form_data)
    if not is_valid:
        logger.warning(f"Form submission contained {len(errors)} validation errors.")
        return render_template(
            "predict.html",
            form_data=form_data,
            errors=errors,
        ), 400

    # 2. Check if model artifact is available
    if not MODEL_LOADED or MODEL_PIPELINE is None:
        logger.error("Predict requested but model artifact is not loaded.")
        errors["general"] = (
            "Model artifact not loaded. Please execute 'python train_model.py' to train the model."
        )
        return render_template(
            "predict.html",
            form_data=form_data,
            errors=errors,
        ), 503

    # 3. Perform inference
    try:
        prediction_result = make_prediction(MODEL_PIPELINE, cleaned_data)
        logger.info(
            f"Prediction completed: {prediction_result['result_label']} "
            f"({prediction_result['approval_probability']}% prob)"
        )
        return render_template(
            "result.html",
            result=prediction_result,
            input_data=cleaned_data,
        )
    except Exception as e:
        logger.error(f"Inference error during make_prediction: {e}", exc_info=True)
        errors["general"] = f"An unexpected error occurred while assessing the application: {e}"
        return render_template(
            "predict.html",
            form_data=form_data,
            errors=errors,
        ), 500


@app.route("/dashboard", methods=["GET"])
def dashboard():
    """
    Model performance dashboard:
    Reads metrics.json at runtime to display test accuracy, baseline,
    confusion matrix, and EDA charts generated by the training pipeline.
    """
    metrics: Dict[str, Any] = {}
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metrics.json: {e}")

    return render_template("dashboard.html", metrics=metrics)


@app.route("/about", methods=["GET"])
def about():
    """About & methodology page explaining Decision Tree concepts and ethical choices."""
    return render_template("about.html")


@app.route("/health", methods=["GET"])
def health():
    """
    Render health check endpoint:
    Returns HTTP 200 with JSON payload describing status and model readiness.
    """
    status_payload = {
        "status": "ok",
        "model_loaded": MODEL_LOADED,
        "model_path": str(MODEL_FILE_PATH),
        "is_synthetic": MODEL_METADATA.get("is_synthetic", False),
        "data_source": MODEL_METADATA.get("data_source", "none"),
        "sklearn_version": MODEL_METADATA.get("sklearn_version", "unknown"),
        "test_accuracy": MODEL_METADATA.get("metrics", {}).get("test_accuracy"),
        "baseline_accuracy": MODEL_METADATA.get("metrics", {}).get("baseline_accuracy"),
    }
    return jsonify(status_payload), 200


@app.route("/outputs/<path:filename>")
def serve_output_file(filename: str):
    """Safely serves saved diagnostic plots and charts from the outputs/ directory."""
    # Prevent directory traversal
    safe_path = (OUTPUTS_DIR / filename).resolve()
    if not str(safe_path).startswith(str(OUTPUTS_DIR.resolve())):
        abort(403)
    if not safe_path.exists():
        abort(404)
    return send_from_directory(str(OUTPUTS_DIR), filename)


@app.errorhandler(404)
def page_not_found(e):
    """Custom 404 handler."""
    return render_template(
        "error.html",
        status_code=404,
        error_title="Page Not Found",
        error_message="The requested URL was not found on this server. Please check the address or return home.",
    ), 404


@app.errorhandler(500)
def internal_server_error(e):
    """Custom 500 handler."""
    logger.error(f"Internal server error: {e}")
    return render_template(
        "error.html",
        status_code=500,
        error_title="Internal Server Error",
        error_message="An unexpected server error occurred. Please try again shortly.",
    ), 500


if __name__ == "__main__":
    # In local development, port defaults to 5000
    debug_mode = os.environ.get("FLASK_DEBUG", "0").strip() in ("1", "true", "True")
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Loan Approval Prediction server on http://127.0.0.1:{port} (Debug: {debug_mode})")
    app.run(host="127.0.0.1", port=port, debug=debug_mode)
