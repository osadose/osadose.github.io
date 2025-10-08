"""
Configuration file for evaluate_LG_on_D2 modular version.
Keeps constants consistent with original working script.
"""

import math
import os
from pathlib import Path

# -------------------------------
# File paths
# -------------------------------
TRAIN_FILE = "NLFS_2024Q1_INDIVIDUAL 2.xlsx"
TRAIN_FILE_FALLBACK = "NLFS_2024Q1_INDIVIDUAL.xlsx"
TEST_FILE = "NLFS_2024_Q2.csv"

# -------------------------------
# Randomness / CV / Scoring
# -------------------------------
RANDOM_STATE = 42
CV_FOLDS = 5
SCORING = "f1_weighted"
N_JOBS = 1  # safer for Windows
VERBOSE = 1
TEST_SIZE = 0.20

# -------------------------------
# Feature settings
# -------------------------------
MAX_WORD_FEATURES = 10000
WORD_NGRAM_RANGE = (1, 3)
WORD_MIN_DF = 3
USE_CHAR_NGRAMS = False
CHAR_NGRAM_RANGE = (3, 5)
CHAR_MIN_DF = 2
CHAR_MAX_FEATURES = 30000
USE_DOMAIN_NORMALIZER = True
USE_ELASTICNET_GRID = False

# -------------------------------
# Columns (exactly from working script)
# -------------------------------
TEXT_COLS = ["mjj2a", "mjj2b", "mjj3a", "mjj3b"]
CODE_COLS = ["mjj2ccleanmaingroup", "mjj3ccleansection", "mjj3cclean"]
TARGET_COL = "mjj2cclean"

# -------------------------------
# Class frequency thresholds
# -------------------------------
MIN_SAMPLES_PER_CLASS_TRAIN_REQUIRED = 5
MIN_SAMPLES_PER_CLASS_TUNING = math.ceil(
    MIN_SAMPLES_PER_CLASS_TRAIN_REQUIRED / (1.0 - TEST_SIZE)
)

# -------------------------------
# Output directory
# -------------------------------
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)



"""
General utilities: file resolution, seeding, and simple logging setup.
"""

import os
import numpy as np
import random
from coder.config import TRAIN_FILE, TRAIN_FILE_FALLBACK, TEST_FILE


def set_seed(seed=42):
    """Ensure reproducibility."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_train_file():
    """Find the training file."""
    if os.path.exists(TRAIN_FILE):
        return TRAIN_FILE
    if os.path.exists(TRAIN_FILE_FALLBACK):
        print(f"[Warn] '{TRAIN_FILE}' not found. Using fallback '{TRAIN_FILE_FALLBACK}'.")
        return TRAIN_FILE_FALLBACK
    raise FileNotFoundError(f"Training file not found in {os.getcwd()}")


def resolve_test_file():
    """Find the test file."""
    if os.path.exists(TEST_FILE):
        return TEST_FILE
    raise FileNotFoundError(f"Test file not found in {os.getcwd()}")



"""
Data loading and preparation utilities for D1 (train) and D2 (test).
"""

import pandas as pd
from coder.config import TEXT_COLS, CODE_COLS, TARGET_COL


def load_data(train_path: str, test_path: str):
    """Load and clean the D1 and D2 datasets."""
    # Load
    df_train = pd.read_excel(train_path, engine="openpyxl") if train_path.lower().endswith("xlsx") else pd.read_csv(train_path)
    df_test = pd.read_csv(test_path) if test_path.lower().endswith(".csv") else pd.read_excel(test_path, engine="openpyxl")

    # Validate required columns
    required = [TARGET_COL] + TEXT_COLS + CODE_COLS
    missing_train = [c for c in required if c not in df_train.columns]
    missing_test = [c for c in required if c not in df_test.columns]
    if missing_train:
        raise ValueError(f"TRAIN missing expected column(s): {missing_train}")
    if missing_test:
        raise ValueError(f"TEST missing expected column(s): {missing_test}")

    # Clean text columns
    df_train = df_train.dropna(subset=[TARGET_COL]).copy()
    df_train[TEXT_COLS] = df_train[TEXT_COLS].fillna("")
    df_test[TEXT_COLS] = df_test[TEXT_COLS].fillna("")

    df_train["combined_text"] = df_train[TEXT_COLS].apply(lambda x: " ".join(x.astype(str)), axis=1)
    df_test["combined_text"] = df_test[TEXT_COLS].apply(lambda x: " ".join(x.astype(str)), axis=1)

    return df_train, df_test



"""
Model pipeline builder for Logistic Regression (TF-IDF + code OHE).
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from coder.config import (
    MAX_WORD_FEATURES, WORD_NGRAM_RANGE, WORD_MIN_DF,
    USE_CHAR_NGRAMS, CHAR_NGRAM_RANGE, CHAR_MIN_DF, CHAR_MAX_FEATURES,
    USE_DOMAIN_NORMALIZER, USE_ELASTICNET_GRID,
    CODE_COLS, RANDOM_STATE, SCORING, CV_FOLDS, N_JOBS, VERBOSE
)
from coder.preprocessing import TextCleaner, DomainNormalizer


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor():
    text_steps = [("cleaner", TextCleaner())]
    if USE_DOMAIN_NORMALIZER:
        text_steps.append(("domain", DomainNormalizer()))
    text_steps.append(("tfidf", TfidfVectorizer(
        max_features=MAX_WORD_FEATURES,
        ngram_range=WORD_NGRAM_RANGE,
        min_df=WORD_MIN_DF,
        sublinear_tf=True,
        strip_accents="unicode",
    )))

    transformers = [("text", Pipeline(text_steps), "combined_text")]
    transformers.append(("codes", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_ohe())
    ]), CODE_COLS))

    return ColumnTransformer(transformers=transformers)


def build_pipeline():
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("classifier", LogisticRegression(
            max_iter=800,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            tol=1e-3,
        )),
    ])


def build_param_grid():
    base_grid = [{
        "classifier__solver": ["lbfgs", "liblinear"],
        "classifier__C": [0.5, 1.0, 2.0, 5.0, 10.0],
        "classifier__penalty": ["l2"],
    }]
    if not USE_ELASTICNET_GRID:
        return base_grid
    enet_grid = [{
        "classifier__solver": ["saga"],
        "classifier__penalty": ["elasticnet"],
        "classifier__l1_ratio": [0.1, 0.5],
        "classifier__C": [0.5, 1.0],
    }]
    return base_grid + enet_grid


def tune_model(X_train, y_train):
    """Run GridSearchCV and return best estimator."""
    grid = GridSearchCV(
        estimator=build_pipeline(),
        param_grid=build_param_grid(),
        cv=CV_FOLDS,
        scoring=SCORING,
        n_jobs=N_JOBS,
        verbose=VERBOSE,
        refit=True,
    )
    grid.fit(X_train, y_train)
    print(f"Best F1_weighted={grid.best_score_:.4f}")
    print(f"Best Params={grid.best_params_}")
    return grid.best_estimator_



"""
Evaluation and report generation for D2 dataset.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from coder.metrics import compute_metrics, normalize_code_str, to_level


def evaluate_on_D2(model, df2, y1_full, output_dir):
    """Predict and evaluate on D2, saving CSV reports."""
    df2["target_norm4"] = df2["mjj2cclean"].astype(str).map(normalize_code_str)
    df2_eval = df2.dropna(subset=["mjj2cclean"]).copy()
    X2_all = df2[["combined_text", "mjj2ccleanmaingroup", "mjj3ccleansection", "mjj3cclean"]]
    y2_all_pred = model.predict(X2_all)
    try:
        proba = model.predict_proba(X2_all)
        proba_max = proba.max(axis=1)
    except Exception:
        proba_max = np.full(len(X2_all), np.nan)

    out = df2.copy()
    out["pred_mjj2cclean"] = pd.Series(y2_all_pred).astype(str).map(normalize_code_str)
    out["pred_confidence"] = proba_max
    pred_path = output_dir / "predictions_D2_from_LG.csv"
    out.to_csv(pred_path, index=False)
    print(f"[Saved] {pred_path}")

    if len(df2_eval) > 0:
        X2_eval = df2_eval[["combined_text", "mjj2ccleanmaingroup", "mjj3ccleansection", "mjj3cclean"]]
        y_true = df2_eval["target_norm4"]
        y_pred = model.predict(X2_eval)
        y_pred_norm = pd.Series(y_pred).astype(str).map(normalize_code_str)
        _ = compute_metrics(y_true, y_pred_norm, "D2 full labelled subset")

        report = classification_report(y_true, y_pred_norm, digits=4, zero_division=0, output_dict=True)
        report_path = output_dir / "classification_report_D2_full.csv"
        pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "label"}).to_csv(report_path, index=False)
        print(f"[Saved] {report_path}")
    else:
        print("[Warn] No labelled rows in D2.")



"""
Main script for evaluate_LG_on_D2 modular version.
"""

from coder.utils import resolve_train_file, resolve_test_file, set_seed
from coder.data_utils import load_data
from coder.model import tune_model
from coder.evaluation import evaluate_on_D2
from coder.config import OUTPUT_DIR, RANDOM_STATE


def main():
    set_seed(RANDOM_STATE)
    print("🚀 Running Modularized evaluate_LG_on_D2...")

    train_path = resolve_train_file()
    test_path = resolve_test_file()
    df_train, df_test = load_data(train_path, test_path)

    X_train = df_train[["combined_text", "mjj2ccleanmaingroup", "mjj3ccleansection", "mjj3cclean"]]
    y_train = df_train["mjj2cclean"].astype(str)

    print("[Info] Training model...")
    best_model = tune_model(X_train, y_train)

    print("[Info] Evaluating on D2...")
    evaluate_on_D2(best_model, df_test, y_train, OUTPUT_DIR)
    print("✅ All tasks completed successfully.")


if __name__ == "__main__":
    main()



"""
Text and domain preprocessing utilities for the ML pipeline.

Follows RAP principles:
- Deterministic text transformations
- Transparent domain normalization
- Modular and reproducible components

This module defines two custom scikit-learn compatible transformers:
    1. TextCleaner - basic text cleaning
    2. DomainNormalizer - applies domain-specific replacements
"""

import re
import string
import logging
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Cleans text deterministically:
    - Lowercase
    - Remove punctuation
    - Collapse extra spaces

    Returns a cleaned text Series suitable for TF-IDF.
    """

    def __init__(self):
        self._regex_punct = re.compile(f"[{re.escape(string.punctuation)}]")
        self._regex_space = re.compile(r"\s+")

    def fit(self, X, y=None):
        # No fitting required, but return self for pipeline compatibility
        return self

    def transform(self, X):
        # Convert to pandas Series if input is list-like
        if not isinstance(X, pd.Series):
            X = pd.Series(X)

        cleaned = (
            X.astype(str)
            .str.lower()
            .str.replace(self._regex_punct, " ", regex=True)
            .str.replace(self._regex_space, " ", regex=True)
            .str.strip()
        )
        logging.debug("TextCleaner: Cleaned text applied successfully.")
        return cleaned


class DomainNormalizer(BaseEstimator, TransformerMixin):
    """
    Applies domain-specific replacements to text.
    Example:
        "mgr" → "manager"
        "engnr" → "engineer"
        "adm" → "administrator"

    Deterministic and reproducible.
    """

    def __init__(self):
        # Define domain-specific abbreviations and corrections
        self.replacements = {
            r"\bmgr\b": "manager",
            r"\bengnr\b": "engineer",
            r"\beng\b": "engineer",
            r"\badmin?\b": "administrator",
            r"\bass?t\b": "assistant",
            r"\bsupt\b": "superintendent",
            r"\btech\b": "technician",
            r"\binsp\b": "inspector",
            r"\bclrk\b": "clerk",
            r"\boff\b": "officer",
        }

    def fit(self, X, y=None):
        return self  # no fitting required

    def transform(self, X):
        if not isinstance(X, pd.Series):
            X = pd.Series(X)
        X = X.astype(str).str.lower()

        # Apply replacements in order
        for pattern, repl in self.replacements.items():
            X = X.str.replace(pattern, repl, regex=True)

        logging.debug("DomainNormalizer: Applied domain-specific replacements.")
        return X



























