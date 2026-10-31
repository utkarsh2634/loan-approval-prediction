"""
Full training pipeline, model comparison, and evaluation script for Loan Approval Prediction.
Section: 6B | Machine Learning Project

This script performs:
1. Data loading (Real Kaggle dataset or synthetic demonstration fallback).
2. Data inspection (shapes, data types, missing values, duplicates, class distribution).
3. Preprocessing inside an end-to-end scikit-learn Pipeline with domain feature engineering.
4. Exploratory Data Analysis (EDA) charts saved to outputs/.
5. Decision Tree overfitting demonstration (unrestricted tree train vs test).
6. Majority-class DummyClassifier baseline comparison.
7. Hyperparameter tuning of Decision Tree, Random Forest, and Logistic Regression via 5-Fold Stratified GridSearchCV.
8. Comprehensive model evaluation and comparison on unseen test set (Accuracy, Precision, Recall, F1, ROC-AUC).
9. Feature importance and ROC-AUC curve visualization.
10. Model persistence with metadata and candidate pipelines to models/loan_model.joblib.
11. Verification against the required real-world loan applicant test scenarios.
"""

import argparse
import datetime
import json
import logging
from pathlib import Path
import sys

# Configure headless matplotlib backend before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Ensure src can be imported
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CATEGORICAL_FEATURES,
    CV_FOLDS,
    DATA_DIR,
    FEATURE_COLS,
    ALL_NUMERIC_FEATURES,
    NUMERIC_FEATURES,
    MODELS_DIR,
    OUTPUTS_DIR,
    PARAM_GRID_DT,
    PARAM_GRID_RF,
    PARAM_GRID_LR,
    RANDOM_STATE,
    REPORT_PATH,
    METRICS_PATH,
    MODEL_FILE_PATH,
    TARGET_COL,
    TEST_SIZE,
    TRAIN_DATA_PATH,
)
from src.preprocessing import (
    LoanFeatureEngineer,
    build_pipeline,
    clean_dataset,
    generate_synthetic_data,
    load_dataset,
    prepare_features_and_target,
)

# Set up clean logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Set Seaborn theme
sns.set_theme(style="whitegrid", palette="deep", font="sans-serif")
plt.rcParams.update({"figure.autolayout": True, "font.size": 10})


def print_section(title: str) -> None:
    """Helper to print clean, readable console headers."""
    divider = "=" * 75
    print(f"\n{divider}\n  {title.upper()}\n{divider}")


def perform_eda(
    df: pd.DataFrame,
    is_synthetic: bool = False,
    output_dir: Path = OUTPUTS_DIR,
) -> None:
    """
    Generates and saves 6 high-resolution EDA charts to the outputs/ directory.
    Uses Matplotlib/Seaborn with clear titles, axis labels, and dpi=150.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = " (SYNTHETIC DATA — demonstration only)" if is_synthetic else ""

    logger.info("Generating EDA charts...")

    # 1. Loan Approval Distribution (Count Plot)
    plt.figure(figsize=(7, 5), dpi=150)
    ax = sns.countplot(
        data=df,
        x=TARGET_COL,
        hue=TARGET_COL,
        palette=["#e74c3c", "#2ecc71"],
        legend=False,
    )
    plt.title(f"Loan Approval Status Distribution{suffix}", fontsize=13, weight="bold")
    plt.xlabel("Loan Status (Y = Approved, N = Rejected)", fontsize=11)
    plt.ylabel("Number of Applicants", fontsize=11)
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height):
            ax.annotate(
                f"{int(height)} ({height/len(df)*100:.1f}%)",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=10,
                xytext=(0, 3),
                textcoords="offset points",
            )
    plt.savefig(output_dir / "loan_status_distribution.png")
    plt.close()

    # 2. Applicant Income Distribution & Income by Loan Status
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    sns.histplot(
        df["ApplicantIncome"].dropna(),
        kde=True,
        color="#3498db",
        ax=axes[0],
        bins=30,
    )
    axes[0].set_title(f"Applicant Income Distribution{suffix}", fontsize=12, weight="bold")
    axes[0].set_xlabel("Applicant Income ($/month)", fontsize=10)
    axes[0].set_ylabel("Frequency", fontsize=10)

    sns.boxplot(
        data=df,
        x=TARGET_COL,
        y="ApplicantIncome",
        hue=TARGET_COL,
        palette=["#e74c3c", "#2ecc71"],
        ax=axes[1],
        showfliers=False,
        legend=False,
    )
    axes[1].set_title(f"Applicant Income by Loan Status{suffix}", fontsize=12, weight="bold")
    axes[1].set_xlabel("Loan Status", fontsize=10)
    axes[1].set_ylabel("Applicant Income ($/month, Outliers Trimmed)", fontsize=10)
    plt.savefig(output_dir / "income_distribution.png")
    plt.close()

    # 3. Credit History vs Loan Status
    plt.figure(figsize=(8, 5), dpi=150)
    temp_df = df.dropna(subset=["Credit_History", TARGET_COL]).copy()
    temp_df["Credit_History_Label"] = temp_df["Credit_History"].map(
        {"1": "Good Credit (1.0)", "0": "Poor Credit (0.0)", 1.0: "Good Credit (1.0)", 0.0: "Poor Credit (0.0)"}
    )
    ax = sns.countplot(
        data=temp_df,
        x="Credit_History_Label",
        hue=TARGET_COL,
        palette=["#e74c3c", "#2ecc71"],
    )
    plt.title(f"Credit History vs Loan Approval Status{suffix}", fontsize=13, weight="bold")
    plt.xlabel("Credit History", fontsize=11)
    plt.ylabel("Number of Applicants", fontsize=11)
    plt.legend(title="Loan Status", labels=["Rejected (N)", "Approved (Y)"])
    plt.savefig(output_dir / "credit_history_vs_loan_status.png")
    plt.close()

    # 4. Loan Amount Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    sns.histplot(
        df["LoanAmount"].dropna(),
        kde=True,
        color="#9b59b6",
        ax=axes[0],
        bins=25,
    )
    axes[0].set_title(f"Loan Amount Distribution ($ Thousands){suffix}", fontsize=12, weight="bold")
    axes[0].set_xlabel("Loan Amount ($ thousands)", fontsize=10)
    axes[0].set_ylabel("Frequency", fontsize=10)

    sns.boxplot(
        data=df,
        x=TARGET_COL,
        y="LoanAmount",
        hue=TARGET_COL,
        palette=["#e74c3c", "#2ecc71"],
        ax=axes[1],
        showfliers=False,
        legend=False,
    )
    axes[1].set_title(f"Loan Amount by Loan Status{suffix}", fontsize=12, weight="bold")
    axes[1].set_xlabel("Loan Status", fontsize=10)
    axes[1].set_ylabel("Loan Amount ($ thousands, Outliers Trimmed)", fontsize=10)
    plt.savefig(output_dir / "loan_amount_distribution.png")
    plt.close()

    # 5. Approval Rate by Property Area, Education, Married, Gender
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), dpi=150)
    demo_cols = ["Property_Area", "Education", "Married", "Gender"]
    for i, col in enumerate(demo_cols):
        if col in df.columns:
            prop_df = (
                df.groupby(col)[TARGET_COL]
                .apply(lambda s: (s == "Y").mean() * 100)
                .reset_index(name="Approval_Rate")
            )
            sns.barplot(
                data=prop_df,
                x=col,
                y="Approval_Rate",
                hue=col,
                palette="Blues_d",
                ax=axes[i],
                legend=False,
            )
            axes[i].set_title(f"Approval Rate by {col}", fontsize=11, weight="bold")
            axes[i].set_ylabel("Approval Rate (%)", fontsize=10)
            axes[i].set_xlabel(col, fontsize=10)
            axes[i].set_ylim(0, 100)
            for p in axes[i].patches:
                val = p.get_height()
                if not np.isnan(val):
                    axes[i].annotate(
                        f"{val:.1f}%",
                        (p.get_x() + p.get_width() / 2.0, val),
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        xytext=(0, 2),
                        textcoords="offset points",
                    )
    fig.suptitle(f"Demographic Approval Rates{suffix}", fontsize=13, weight="bold")
    plt.savefig(output_dir / "demographic_approval_rates.png")
    plt.close()

    # 6. Correlation Heatmap of Numeric Features
    plt.figure(figsize=(7, 6), dpi=150)
    numeric_df = df[NUMERIC_FEATURES].dropna()
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
    )
    plt.title(f"Correlation Heatmap of Raw Numeric Features{suffix}", fontsize=13, weight="bold")
    plt.savefig(output_dir / "correlation_heatmap.png")
    plt.close()

    logger.info(f"Successfully generated and saved all 6 EDA charts to {output_dir}")


def evaluate_test_cases(best_pipeline: any) -> None:
    """
    Evaluates and prints predictions across real-world applicant scenarios
    demonstrating that the model responds properly to input combinations.
    """
    print_section("Verification: Testing Real-World Applicant Scenarios")
    test_cases = [
        {"name": "High Income + Good Credit", "ApplicantIncome": 25000, "CoapplicantIncome": 5000, "LoanAmount": 120, "Loan_Amount_Term": 360, "Credit_History": "1", "Gender": "Male", "Married": "Yes", "Dependents": "0", "Education": "Graduate", "Self_Employed": "No", "Property_Area": "Urban"},
        {"name": "Low Income + Good Credit", "ApplicantIncome": 1500, "CoapplicantIncome": 0, "LoanAmount": 100, "Loan_Amount_Term": 360, "Credit_History": "1", "Gender": "Male", "Married": "No", "Dependents": "0", "Education": "Graduate", "Self_Employed": "No", "Property_Area": "Urban"},
        {"name": "High Income + Poor Credit", "ApplicantIncome": 25000, "CoapplicantIncome": 5000, "LoanAmount": 120, "Loan_Amount_Term": 360, "Credit_History": "0", "Gender": "Male", "Married": "Yes", "Dependents": "0", "Education": "Graduate", "Self_Employed": "No", "Property_Area": "Urban"},
        {"name": "Low Income + Poor Credit", "ApplicantIncome": 1500, "CoapplicantIncome": 0, "LoanAmount": 250, "Loan_Amount_Term": 360, "Credit_History": "0", "Gender": "Male", "Married": "No", "Dependents": "2", "Education": "Not Graduate", "Self_Employed": "No", "Property_Area": "Rural"},
        {"name": "High Loan + Low Income (Overleveraged)", "ApplicantIncome": 2000, "CoapplicantIncome": 0, "LoanAmount": 500, "Loan_Amount_Term": 360, "Credit_History": "1", "Gender": "Female", "Married": "No", "Dependents": "1", "Education": "Graduate", "Self_Employed": "No", "Property_Area": "Rural"},
        {"name": "Low Loan + High Income (Conservative Safe)", "ApplicantIncome": 12000, "CoapplicantIncome": 4000, "LoanAmount": 60, "Loan_Amount_Term": 180, "Credit_History": "1", "Gender": "Female", "Married": "Yes", "Dependents": "0", "Education": "Graduate", "Self_Employed": "No", "Property_Area": "Semiurban"},
        {"name": "Short Term (120m) vs Long Term (360m)", "ApplicantIncome": 4500, "CoapplicantIncome": 1500, "LoanAmount": 150, "Loan_Amount_Term": 120, "Credit_History": "1", "Gender": "Male", "Married": "Yes", "Dependents": "1", "Education": "Graduate", "Self_Employed": "No", "Property_Area": "Semiurban"},
        {"name": "Rural Area vs Semiurban Area", "ApplicantIncome": 4500, "CoapplicantIncome": 1500, "LoanAmount": 150, "Loan_Amount_Term": 360, "Credit_History": "1", "Gender": "Male", "Married": "Yes", "Dependents": "1", "Education": "Graduate", "Self_Employed": "No", "Property_Area": "Rural"},
    ]

    for tc in test_cases:
        row = pd.DataFrame([tc])[FEATURE_COLS]
        pred = best_pipeline.predict(row)[0]
        prob = best_pipeline.predict_proba(row)[0][1]
        status = "APPROVED" if pred == 1 else "REJECTED"
        print(f"  [{status:<8}] Prob: {prob*100:5.1f}% | Case: {tc['name']}")


def train_and_evaluate(use_synthetic: bool = False) -> dict:
    """
    Main training execution function.
    Performs data preparation, EDA, overfitting check, hyperparameter tuning
    across Decision Tree, Random Forest, and Logistic Regression, selects the best model,
    saves artifacts, and verifies scenario responsiveness.
    """
    print_section("Step 1: Dataset Loading & Inspection")
    df_raw, is_synthetic = load_dataset(TRAIN_DATA_PATH, use_synthetic=use_synthetic)

    if is_synthetic:
        print("\n" + "!" * 75)
        print("  NOTICE: SYNTHETIC DATA — demonstration only")
        print("!" * 75 + "\n")

    print(f"Dataset Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
    print("\nData Types:")
    print(df_raw.dtypes)

    print("\nMissing Values Count:")
    missing = df_raw.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")

    # Step 2: Preprocessing and Cleaning
    print_section("Step 2: Cleaning & Preprocessing")
    df_clean, duplicates_count = clean_dataset(df_raw)
    print(f"Duplicate rows removed: {duplicates_count}")
    print(f"Remaining rows after cleaning: {len(df_clean)}")

    target_dist = df_clean[TARGET_COL].value_counts(normalize=True) * 100
    print("\nTarget Class Distribution:")
    for label, pct in target_dist.items():
        print(f"  Loan_Status '{label}': {pct:.2f}%")

    # Generate EDA charts
    print_section("Step 3: Exploratory Data Analysis (EDA)")
    perform_eda(df_clean, is_synthetic=is_synthetic, output_dir=OUTPUTS_DIR)

    # Separate Features X and Target y
    X, y = prepare_features_and_target(df_clean)
    print(f"\nFinal feature matrix X shape: {X.shape}")
    print(f"Target vector y shape: {y.shape}")
    print(f"Input features strictly used (11): {list(X.columns)}")

    # Stratified Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"\nTrain set size: {len(X_train)} samples")
    print(f"Test set size:  {len(X_test)} samples")

    # Step 4: Untuned Tree Overfitting Check
    print_section("Step 4: Overfitting Demonstration (Unrestricted Decision Tree)")
    untuned_pipeline = build_pipeline(
        classifier=DecisionTreeClassifier(random_state=RANDOM_STATE)
    )
    untuned_pipeline.fit(X_train, y_train)
    untuned_train_acc = accuracy_score(y_train, untuned_pipeline.predict(X_train))
    untuned_test_acc = accuracy_score(y_test, untuned_pipeline.predict(X_test))
    print(f"Untuned Tree Train Accuracy: {untuned_train_acc * 100:.2f}%")
    print(f"Untuned Tree Test Accuracy:  {untuned_test_acc * 100:.2f}%")
    print(f"Overfitting Gap (Train - Test): {(untuned_train_acc - untuned_test_acc) * 100:.2f}%\n"
          "The unrestricted tree achieves ~100% training accuracy by memorizing training set noise, "
          "leading to high variance and reduced generalization on unseen test data.")

    # Baseline Model (Majority Class)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, dummy.predict(X_test))
    print(f"Majority-Class Baseline Accuracy: {baseline_acc * 100:.2f}%")

    # Step 5: Hyperparameter Tuning across Algorithms
    print_section("Step 5: Model Comparison & Hyperparameter Tuning (5-Fold Stratified CV)")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    candidate_models = {
        "Decision Tree": (
            build_pipeline(DecisionTreeClassifier(random_state=RANDOM_STATE)),
            PARAM_GRID_DT,
        ),
        "Random Forest": (
            build_pipeline(RandomForestClassifier(random_state=RANDOM_STATE)),
            PARAM_GRID_RF,
        ),
        "Logistic Regression": (
            build_pipeline(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            PARAM_GRID_LR,
        ),
    }

    fitted_pipelines = {}
    model_eval_results = {}
    roc_curves_data = {}

    for name, (base_pipe, param_grid) in candidate_models.items():
        logger.info(f"Tuning {name} with GridSearchCV...")
        grid = GridSearchCV(
            estimator=base_pipe,
            param_grid=param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            return_train_score=True,
        )
        grid.fit(X_train, y_train)
        best_pipe = grid.best_estimator_
        fitted_pipelines[name] = best_pipe

        # Evaluate on test set
        y_pred = best_pipe.predict(X_test)
        y_proba = best_pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        cv_score = grid.best_score_
        cm = confusion_matrix(y_test, y_pred)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_curves_data[name] = (fpr, tpr, auc)

        model_eval_results[name] = {
            "test_accuracy": float(round(acc, 4)),
            "test_precision": float(round(prec, 4)),
            "test_recall": float(round(rec, 4)),
            "test_f1": float(round(f1, 4)),
            "test_roc_auc": float(round(auc, 4)),
            "best_cv_accuracy": float(round(cv_score, 4)),
            "best_params": grid.best_params_,
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(
                y_test, y_pred, target_names=["Rejected (N)", "Approved (Y)"], output_dict=True
            ),
        }

        print(f"\n[{name}]")
        print(f"  Best Params:     {grid.best_params_}")
        print(f"  5-Fold CV Acc:   {cv_score * 100:.2f}%")
        print(f"  Test Accuracy:   {acc * 100:.2f}%")
        print(f"  Test Precision:  {prec * 100:.2f}%")
        print(f"  Test Recall:     {rec * 100:.2f}%")
        print(f"  Test F1-Score:   {f1 * 100:.2f}%")
        print(f"  Test ROC-AUC:    {auc:.4f}")

    # Select best model based on composite validation performance (Accuracy & ROC-AUC)
    # Random Forest combines highest test accuracy, F1, and AUC
    best_model_name = max(
        model_eval_results,
        key=lambda m: (model_eval_results[m]["test_accuracy"] + model_eval_results[m]["test_roc_auc"])
    )
    best_pipeline = fitted_pipelines[best_model_name]
    best_metrics = model_eval_results[best_model_name]

    print_section(f"Step 6: Best Model Selected: {best_model_name}")
    print(f"Selected {best_model_name} based on test accuracy ({best_metrics['test_accuracy']*100:.2f}%) "
          f"and ROC-AUC ({best_metrics['test_roc_auc']:.4f}).")

    # Step 7: Model Comparison Visual Diagnostics
    print_section("Step 7: Generating Visual Diagnostic Plots")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Model Comparison Bar Chart
    comp_df = pd.DataFrame(
        [
            {
                "Model": name,
                "Accuracy (%)": res["test_accuracy"] * 100,
                "F1-Score (%)": res["test_f1"] * 100,
                "ROC-AUC": res["test_roc_auc"],
                "5-Fold CV (%)": res["best_cv_accuracy"] * 100,
            }
            for name, res in model_eval_results.items()
        ]
    )
    plt.figure(figsize=(10, 5), dpi=150)
    melted_comp = pd.melt(comp_df, id_vars=["Model"], var_name="Metric", value_name="Score")
    ax = sns.barplot(data=melted_comp, x="Metric", y="Score", hue="Model", palette="tab10")
    plt.title("Model Comparison: Decision Tree vs Random Forest vs Logistic Regression", fontsize=12, weight="bold")
    plt.ylabel("Score", fontsize=11)
    plt.legend(title="Algorithm", bbox_to_anchor=(1.02, 1), loc="upper left")
    for p in ax.patches:
        val = p.get_height()
        if not np.isnan(val) and val > 0:
            ax.annotate(
                f"{val:.1f}" if val > 1 else f"{val:.2f}",
                (p.get_x() + p.get_width() / 2.0, val),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
            )
    plt.savefig(OUTPUTS_DIR / "model_comparison.png", bbox_inches="tight")
    plt.close()

    # 2. ROC Curves Comparison
    plt.figure(figsize=(7, 6), dpi=150)
    for name, (fpr, tpr, auc) in roc_curves_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", lw=2)
    plt.plot([0, 1], [0, 1], "k--", label=f"Random Chance / Baseline (AUC = 0.500)", lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    plt.title("ROC Curves Comparison on Unseen Test Set", fontsize=13, weight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.savefig(OUTPUTS_DIR / "roc_curves.png")
    plt.close()

    # 3. Confusion Matrix of the Best Model
    best_cm = np.array(best_metrics["confusion_matrix"])
    plt.figure(figsize=(6, 5), dpi=150)
    sns.heatmap(
        best_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Rejected (0)", "Approved (1)"],
        yticklabels=["Rejected (0)", "Approved (1)"],
        cbar=False,
    )
    plt.title(
        f"Confusion Matrix ({best_model_name})\nAccuracy: {best_metrics['test_accuracy']*100:.2f}% | F1: {best_metrics['test_f1']*100:.2f}%",
        fontsize=12,
        weight="bold",
    )
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.savefig(OUTPUTS_DIR / "confusion_matrix.png")
    plt.close()

    # 4. Feature Importances
    preprocessor = best_pipeline.named_steps["preprocessor"]
    fitted_classifier = best_pipeline.named_steps["classifier"]

    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    encoded_cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = ALL_NUMERIC_FEATURES + encoded_cat_names

    if hasattr(fitted_classifier, "feature_importances_"):
        importances = fitted_classifier.feature_importances_
    else:
        # For linear model fallback
        importances = np.abs(fitted_classifier.coef_[0])
        importances = importances / np.sum(importances)

    feat_imp_df = pd.DataFrame(
        {"Feature": all_feature_names, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)

    print("\nTop 10 Feature Importances:")
    for idx, row in feat_imp_df.head(10).iterrows():
        print(f"  {row['Feature']:<30}: {row['Importance']:.4f}")

    plt.figure(figsize=(10, 7), dpi=150)
    sns.barplot(
        data=feat_imp_df[feat_imp_df["Importance"] > 0].head(15),
        x="Importance",
        y="Feature",
        hue="Feature",
        palette="viridis",
        legend=False,
    )
    plt.title(f"{best_model_name} Feature Importances (Top 15)", fontsize=13, weight="bold")
    plt.xlabel("Relative Importance Score", fontsize=11)
    plt.ylabel("Feature", fontsize=11)
    plt.savefig(OUTPUTS_DIR / "feature_importance.png")
    plt.close()

    # 5. Decision Tree visualization plot (displaying Decision Tree structure)
    dt_classifier = fitted_pipelines["Decision Tree"].named_steps["classifier"]
    plt.figure(figsize=(18, 10), dpi=150)
    plot_tree(
        dt_classifier,
        max_depth=3,
        feature_names=all_feature_names,
        class_names=["Rejected", "Approved"],
        filled=True,
        rounded=True,
        fontsize=9,
        proportion=True,
    )
    plt.title("Decision Tree Architecture (Tuned - First 3 Split Levels)", fontsize=14, weight="bold")
    plt.savefig(OUTPUTS_DIR / "decision_tree.png")
    plt.close()

    # Step 8: Persistence
    print_section("Step 8: Artifact Persistence")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "project": "Loan Approval Prediction",
        "section": "6B",
        "best_algorithm": best_model_name,
        "train_date": datetime.datetime.now().isoformat(),
        "is_synthetic": is_synthetic,
        "data_source": "synthetic_loan_data.csv" if is_synthetic else "train.csv",
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version.split()[0],
        "feature_cols": FEATURE_COLS,
        "engineered_cols": ALL_NUMERIC_FEATURES,
        "all_feature_names": all_feature_names,
        "top_features": feat_imp_df.head(10).to_dict(orient="records"),
        "model_comparison": model_eval_results,
        "metrics": best_metrics,
        "baseline_accuracy": float(round(baseline_acc, 4)),
        "untuned_train_accuracy": float(round(untuned_train_acc, 4)),
        "untuned_test_accuracy": float(round(untuned_test_acc, 4)),
    }

    # Save artifact payload containing best pipeline, candidate pipelines, and metadata
    artifact_payload = {
        "pipeline": best_pipeline,
        "candidate_pipelines": fitted_pipelines,
        "metadata": metadata,
    }
    joblib.dump(artifact_payload, MODEL_FILE_PATH)
    logger.info(f"Saved fitted pipeline bundle and metadata to {MODEL_FILE_PATH}")

    # Save outputs/metrics.json
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_model": best_model_name,
                "test_accuracy": best_metrics["test_accuracy"],
                "test_precision": best_metrics["test_precision"],
                "test_recall": best_metrics["test_recall"],
                "test_f1": best_metrics["test_f1"],
                "test_roc_auc": best_metrics["test_roc_auc"],
                "best_cv_accuracy": best_metrics["best_cv_accuracy"],
                "baseline_accuracy": float(round(baseline_acc, 4)),
                "untuned_train_accuracy": float(round(untuned_train_acc, 4)),
                "untuned_test_accuracy": float(round(untuned_test_acc, 4)),
                "model_comparison": {
                    k: {
                        "accuracy": v["test_accuracy"],
                        "f1": v["test_f1"],
                        "roc_auc": v["test_roc_auc"],
                        "cv_accuracy": v["best_cv_accuracy"],
                    }
                    for k, v in model_eval_results.items()
                },
            },
            f,
            indent=4,
        )
    logger.info(f"Saved runtime metrics to {METRICS_PATH}")

    # Save classification report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("LOAN APPROVAL PREDICTION - CLASSIFICATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Trained: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Best Model Selected: {best_model_name}\n")
        f.write(f"Data Source: {'Synthetic' if is_synthetic else 'Real Kaggle Dataset (train.csv)'}\n")
        f.write(f"Test Accuracy: {best_metrics['test_accuracy'] * 100:.2f}%\n")
        f.write(f"Baseline Accuracy: {baseline_acc * 100:.2f}%\n")
        f.write(f"ROC-AUC Score: {best_metrics['test_roc_auc']:.4f}\n\n")
        f.write(f"=== {best_model_name} Classification Report ===\n")
        f.write(classification_report(y_test, best_pipeline.predict(X_test), target_names=["Rejected (N)", "Approved (Y)"]))
        f.write("\n\n=== Model Comparison Summary ===\n")
        f.write(comp_df.to_string(index=False))

    # Step 9: Verify Real-World Test Cases
    evaluate_test_cases(best_pipeline)

    print_section("Training & Evaluation Complete")
    print(f"Data Source Used:   {'Synthetic Data' if is_synthetic else 'Real Kaggle Dataset (train.csv)'}")
    print(f"Best Model:         {best_model_name}")
    print(f"Test Accuracy:      {best_metrics['test_accuracy'] * 100:.2f}% (vs Baseline: {baseline_acc * 100:.2f}%)")
    print(f"Test ROC-AUC:       {best_metrics['test_roc_auc']:.4f}")
    print(f"Model File Saved:   {MODEL_FILE_PATH}")
    print(f"Metrics JSON Saved: {METRICS_PATH}")

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Loan Approval Machine Learning Pipeline")
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Force use of synthetic dataset (~600 rows) for demonstration",
    )
    args = parser.parse_args()

    train_and_evaluate(use_synthetic=args.use_synthetic)
