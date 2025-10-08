
"""
Configuration file for reproducible ML pipeline (RAP style).
Contains constants, hyperparameters, and reusable settings.
"""

RANDOM_STATE = 42
CV_FOLDS = 5
TEST_SIZE = 0.2
SCORING = "f1_weighted"
N_JOBS = -1

# Data structure
TARGET_COL = "target"
TEXT_COLS = ["description"]     # Update this based on your dataset
CODE_COLS = []                  # e.g., ["industry_code"]

# Minimum samples required per class for training
MIN_SAMPLES_PER_CLASS_TUNING = 7



"""
Data utility functions for loading and preparing datasets.
"""

import logging
import pandas as pd
from pathlib import Path


def load_data(train_path: str, test_path: str):
    """
    Load train and test data with automatic file type detection.
    """
    logging.info(f"Loading TRAIN from: {train_path}")
    if str(train_path).lower().endswith((".xlsx", ".xls")):
        df_train = pd.read_excel(train_path)
    else:
        df_train = pd.read_csv(train_path)

    logging.info(f"Loading TEST from: {test_path}")
    if str(test_path).lower().endswith((".xlsx", ".xls")):
        df_test = pd.read_excel(test_path)
    else:
        df_test = pd.read_csv(test_path)

    return df_train, df_test



"""
Preprocessing module — handles text and feature cleaning for RAP pipeline.
"""

import logging
import pandas as pd
from coder.config import TARGET_COL, TEXT_COLS, CODE_COLS


def prepare_data(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """
    Clean, combine, and prepare train/test data for modeling.

    Returns:
        X_train, y_train, X_test, y_test (all pandas objects)
    """
    if TARGET_COL not in df_train.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found in training data.")

    # Drop missing targets
    df_train = df_train.dropna(subset=[TARGET_COL])

    # Fill NaNs and combine text fields
    for df in [df_train, df_test]:
        df[TEXT_COLS] = df[TEXT_COLS].fillna("")
        df["combined_text"] = df[TEXT_COLS].apply(lambda x: " ".join(x.astype(str)), axis=1)

    X_train = df_train[["combined_text"] + CODE_COLS]
    y_train = df_train[TARGET_COL].astype(str)

    if TARGET_COL in df_test.columns:
        y_test = df_test[TARGET_COL].astype(str)
    else:
        y_test = pd.Series([None] * len(df_test), name=TARGET_COL)
        logging.warning("Test data missing target column — proceeding with prediction-only mode.")

    X_test = df_test[["combined_text"] + CODE_COLS]

    logging.info(f"Prepared data successfully: {len(X_train)} train, {len(X_test)} test samples")
    return X_train, y_train, X_test, y_test



"""
Model module — builds, tunes, and returns the Logistic Regression model.
"""

import logging
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from coder.config import RANDOM_STATE, CV_FOLDS, SCORING, N_JOBS


def build_pipeline():
    """Create ML pipeline: TF-IDF + Logistic Regression."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=300, random_state=RANDOM_STATE))
    ])


def build_param_grid():
    """Hyperparameter grid."""
    return {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__solver": ["lbfgs"]
    }


def train_best_model(X_train, y_train, min_samples=7):
    """Train the model with GridSearchCV and return the best estimator."""
    # Filter rare classes
    counts = y_train.value_counts()
    keep = counts[counts >= min_samples].index
    mask = y_train.isin(keep)
    X_train, y_train = X_train[mask], y_train[mask]

    logging.info(f"Filtered classes to {len(keep)} (≥{min_samples} samples each).")

    model = build_pipeline()
    grid = build_param_grid()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    gs = GridSearchCV(
        estimator=model,
        param_grid=grid,
        cv=cv,
        scoring=SCORING,
        n_jobs=N_JOBS,
        refit=True,
        verbose=0
    )
    gs.fit(X_train["combined_text"], y_train)

    logging.info(f"Best params: {gs.best_params_}, score={gs.best_score_:.4f}")
    return gs.best_estimator_



"""
Evaluation module — saves predictions and summary report.
"""

import logging
import pandas as pd
from sklearn.metrics import classification_report


def evaluate_and_save(model, X_test, y_test, output_dir="output"):
    """
    Evaluate trained model on D2 (test set) and save predictions.
    """
    output_dir = pd.Path(output_dir) if not isinstance(output_dir, pd.Path) else output_dir
    output_dir.mkdir(exist_ok=True)

    preds = model.predict(X_test["combined_text"])
    results = pd.DataFrame({
        "text": X_test["combined_text"],
        "true_target": y_test,
        "predicted_target": preds
    })

    pred_path = output_dir / "predictions_D2_from_LG.csv"
    results.to_csv(pred_path, index=False)

    if y_test.notna().any():
        rep = classification_report(y_test, preds, output_dict=True, zero_division=0)
        rep_path = output_dir / "classification_report_D2.csv"
        pd.DataFrame(rep).transpose().to_csv(rep_path)
        logging.info(f"Saved classification report to: {rep_path}")

    logging.info(f"Saved predictions to: {pred_path}")




"""
Main entry point — orchestrates reproducible ML pipeline.
"""

import logging
from pathlib import Path
import time
from coder.data_utils import load_data
from coder.preprocessing import prepare_data
from coder.model import train_best_model
from coder.evaluation import evaluate_and_save

# Setup output and logging
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

logging.basicConfig(
    filename=output_dir / "run.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8"
)

def main():
    start = time.time()
    logging.info("🚀 Starting Reproducible ML Pipeline")

    # Load and prepare data
    df_train, df_test = load_data("data/train.xlsx", "data/test.csv")
    X_train, y_train, X_test, y_test = prepare_data(df_train, df_test)

    # Train model
    model = train_best_model(X_train, y_train)

    # Evaluate and save
    evaluate_and_save(model, X_test, y_test, output_dir="output")

    logging.info(f"✅ Pipeline completed in {round(time.time() - start, 2)}s")


if __name__ == "__main__":
    main()





















































