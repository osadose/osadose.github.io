"""
Configuration constants for the ML pipeline.
Defines paths, columns, and model hyperparameters.
"""

from pathlib import Path

# --- Paths ---
DATA_DIR = Path("data")
TRAIN_FILE = DATA_DIR / "NLFS_2024Q1_INDIVIDUAL.xlsx"
TEST_FILE = DATA_DIR / "NLFS_2024_Q2.csv"

OUTPUT_DIR = Path("output")

# --- Columns ---
# Update these if your dataset uses different names
TARGET_COL = "target"
TEXT_COLS = ["q1_text", "q2_text", "q3_text"]  # Example text columns
CODE_COLS = ["code1", "code2"]

# --- Model parameters ---
CV_FOLDS = 5
SEED = 42




"""
Utility functions for logging and reproducibility.
"""

import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import random
import os


def setup_logging(output_dir: Path):
    """Configure both file and console logging."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler()
        ],
    )
    logging.info(f"Logging initialized. Log file: {log_path}")
    return logging.getLogger(), log_path


def timestamp() -> str:
    """Return a timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def set_seed(seed: int = 42):
    """Ensure full reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)




"""
Data utilities for reproducible ML pipeline.
Handles loading, cleaning, and preparing train/test datasets safely.

Follows RAP (Reproducible Analytical Pipeline) principles:
- Deterministic outputs
- Transparent data transformations
- Logged processing steps
"""

import logging
import pandas as pd
from typing import Tuple
from coder.config import TARGET_COL, TEXT_COLS, CODE_COLS


def prepare_data(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Prepare training and test data for the ML pipeline."""

    if TARGET_COL not in df_train.columns:
        possible_targets = [c for c in df_train.columns if "target" in c.lower() or "code" in c.lower()]
        msg = f"'{TARGET_COL}' not found in training data. Possible alternatives: {possible_targets}"
        logging.error(msg)
        raise KeyError(msg)

    # --- Clean training data ---
    df_train = df_train.dropna(subset=[TARGET_COL]).copy()
    for col in TEXT_COLS:
        if col not in df_train.columns:
            df_train[col] = ""
    df_train["combined_text"] = df_train[TEXT_COLS].astype(str).agg(" ".join, axis=1)

    # --- Clean test data ---
    df_test = df_test.copy()
    for col in TEXT_COLS:
        if col not in df_test.columns:
            df_test[col] = ""
    df_test["combined_text"] = df_test[TEXT_COLS].astype(str).agg(" ".join, axis=1)

    # --- Extract features and labels ---
    for col in CODE_COLS:
        if col not in df_train.columns:
            df_train[col] = 0
        if col not in df_test.columns:
            df_test[col] = 0

    X_train = df_train[["combined_text"] + CODE_COLS]
    y_train = df_train[TARGET_COL].astype(str)

    if TARGET_COL in df_test.columns:
        y_test = df_test[TARGET_COL].astype(str)
    else:
        y_test = pd.Series([None] * len(df_test), name=TARGET_COL)
        logging.warning("No target column in test data — running in prediction-only mode.")

    X_test = df_test[["combined_text"] + CODE_COLS]

    logging.info(f"✅ Prepared data: Train={len(X_train)} samples, Test={len(X_test)} samples")
    return X_train, y_train, X_test, y_test



"""
Model training and selection using Logistic Regression.
Applies reproducibility and transparent hyperparameter tuning.
"""

import logging
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from pathlib import Path
from coder.utils import timestamp
from coder.config import CV_FOLDS, SEED


def build_pipeline() -> Pipeline:
    """Constructs the feature extraction and model pipeline."""
    text_transformer = TfidfVectorizer(max_features=3000)
    preprocessor = ColumnTransformer([
        ("text", text_transformer, "combined_text"),
        ("codes", StandardScaler(with_mean=False), ["code1", "code2"]),
    ])
    model = LogisticRegression(max_iter=1000, random_state=SEED)
    return Pipeline([("preprocessor", preprocessor), ("classifier", model)])


def train_model(X_train, y_train):
    """Train Logistic Regression with GridSearchCV."""
    logging.info("🧠 Running GridSearchCV with 5-fold cross-validation...")
    pipe = build_pipeline()
    param_grid = {
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__solver": ["lbfgs"],
        "classifier__penalty": ["l2"],
    }
    gs = GridSearchCV(pipe, param_grid, cv=CV_FOLDS, scoring="f1_weighted", n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)

    best_model = gs.best_estimator_
    best_score = gs.best_score_
    best_params = gs.best_params_
    logging.info(f"🏆 Best F1_weighted={best_score:.4f} with params={best_params}")
    return best_model, best_score, best_params


def save_model(model, output_dir: Path) -> Path:
    """Save the trained model."""
    output_dir.mkdir(exist_ok=True, parents=True)
    model_path = output_dir / f"best_model_{timestamp()}.joblib"
    joblib.dump(model, model_path)
    logging.info(f"💾 Model saved to {model_path}")
    return model_path



"""
Evaluation utilities for ML pipeline.
Generates one clean classification report and one predictions file.
"""

import logging
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report
from coder.utils import timestamp


def evaluate_and_save(model, X_test, y_test, output_dir: Path, prefix="D2"):
    """Evaluate model and save one report and one predictions file."""
    output_dir.mkdir(exist_ok=True, parents=True)
    ts = timestamp()

    logging.info("🔍 Evaluating model...")
    y_pred = model.predict(X_test)

    # Classification report (if y_test available)
    if y_test.notna().any():
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "label"})
        report_path = output_dir / f"{prefix}_classification_report_{ts}.csv"
        report_df.to_csv(report_path, index=False)
        logging.info(f"📊 Classification report saved: {report_path}")
    else:
        report_path = None
        logging.warning("No true labels found; skipped classification report.")

    # Predictions
    pred_path = output_dir / f"predictions_{prefix}_from_LG_{ts}.csv"
    out_df = X_test.copy()
    out_df["predicted_target"] = y_pred
    out_df.to_csv(pred_path, index=False)
    logging.info(f"💾 Predictions saved: {pred_path}")

    return report_path, pred_path




"""
Main entry point for reproducible ML pipeline.
Follows RAP best practices: reproducibility, transparency, logging.
"""

import pandas as pd
import logging
from coder.utils import setup_logging, set_seed
from coder.config import TRAIN_FILE, TEST_FILE, OUTPUT_DIR
from coder.data_utils import prepare_data
from coder.model import train_model, save_model
from coder.evaluation import evaluate_and_save


def main():
    logger, _ = setup_logging(OUTPUT_DIR)
    set_seed(42)
    logger.info("🚀 Starting Reproducible ML Pipeline")

    # --- Load Data ---
    logger.info(f"Loading training data from: {TRAIN_FILE}")
    df_train = pd.read_excel(TRAIN_FILE) if TRAIN_FILE.suffix == ".xlsx" else pd.read_csv(TRAIN_FILE)

    logger.info(f"Loading test data from: {TEST_FILE}")
    df_test = pd.read_excel(TEST_FILE) if TEST_FILE.suffix == ".xlsx" else pd.read_csv(TEST_FILE)

    # --- Prepare ---
    X_train, y_train, X_test, y_test = prepare_data(df_train, df_test)

    # --- Train ---
    best_model, best_score, best_params = train_model(X_train, y_train)

    # --- Save Model ---
    save_model(best_model, OUTPUT_DIR)

    # --- Evaluate ---
    evaluate_and_save(best_model, X_test, y_test, OUTPUT_DIR, prefix="D2")

    logger.info("✅ Pipeline completed successfully.")


if __name__ == "__main__":
    main()


