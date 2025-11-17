"""
Configuration constants for the ML pipeline (uses column names from evaluate_LG_on_D2).
"""
import math

# Data files (put your files in the project root or update paths)
TRAIN_FILE = "NLFS_2024Q1_INDIVIDUAL 2.xlsx"
TRAIN_FILE_FALLBACK = "NLFS_2024Q1_INDIVIDUAL.xlsx"
TEST_FILE = "NLFS_2024_Q2.csv"

# Randomness & CV
RANDOM_STATE = 42
CV_FOLDS = 5
TEST_SIZE = 0.20
SCORING = "f1_weighted"
N_JOBS = 1
VERBOSE = 1

# Columns (from your working script)
TEXT_COLS = ["mjj2a", "mjj2b", "mjj3a", "mjj3b"]
# code columns used as categorical features
CODE_FEAT_COLS = ["mjj2ccleanmaingroup", "mjj3ccleansection"]
# targets (multi-output)
TARGET_ISCO = "mjj2cclean"
TARGET_ISIC = "mjj3cclean"
TARGET_COLS = [TARGET_ISCO, TARGET_ISIC]

# TFIDF / Preprocessing
MAX_WORD_FEATURES = 10000
WORD_NGRAM_RANGE = (1, 3)
WORD_MIN_DF = 3
USE_DOMAIN_NORMALIZER = True

# Minimum samples thresholds (same logic as your script)
MIN_SAMPLES_PER_CLASS_TRAIN_REQUIRED = 5
MIN_SAMPLES_PER_CLASS_TUNING = math.ceil(
    MIN_SAMPLES_PER_CLASS_TRAIN_REQUIRED / (1.0 - TEST_SIZE)
)



"""Small utilities: logging setup and timestamp helper."""
import logging
from pathlib import Path
from datetime import datetime
import sys

def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    # Configure a file handler and a console handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt))
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(fmt))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Remove existing handlers to avoid duplicates
    if logger.handlers:
        for h in logger.handlers[:]:
            logger.removeHandler(h)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logging.info(f"Logging initialized. Log file: {log_path}")
    return log_path

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")




"""Load data and prepare combined_text fields (matching original script behavior)."""

import os
import pandas as pd
from coder.config import TRAIN_FILE, TRAIN_FILE_FALLBACK, TEST_FILE, TEXT_COLS, CODE_FEAT_COLS, TARGET_COLS

def resolve_train_file():
    if os.path.exists(TRAIN_FILE):
        return TRAIN_FILE
    if os.path.exists(TRAIN_FILE_FALLBACK):
        print(f"[Warn] {TRAIN_FILE} not found. Falling back to {TRAIN_FILE_FALLBACK}")
        return TRAIN_FILE_FALLBACK
    raise FileNotFoundError(f"Could not find {TRAIN_FILE} or fallback in current working directory.")

def resolve_test_file():
    if os.path.exists(TEST_FILE):
        return TEST_FILE
    raise FileNotFoundError(f"Could not find {TEST_FILE} in current working directory.")

def load_data():
    train_path = resolve_train_file()
    test_path = resolve_test_file()

    # Training
    if train_path.lower().endswith((".xlsx", ".xls")):
        df_train = pd.read_excel(train_path, engine="openpyxl")
    else:
        df_train = pd.read_csv(train_path)

    # Test
    if test_path.lower().endswith((".xlsx", ".xls")):
        df_test = pd.read_excel(test_path, engine="openpyxl")
    else:
        df_test = pd.read_csv(test_path)

    # Validate required columns (targets + text + code features)
    required = list(TARGET_COLS) + TEXT_COLS + CODE_FEAT_COLS
    missing_train = [c for c in required if c not in df_train.columns]
    missing_test = [c for c in required if c not in df_test.columns]
    if missing_train:
        raise KeyError(f"TRAIN missing expected column(s): {missing_train}")
    if missing_test:
        raise KeyError(f"TEST missing expected column(s): {missing_test}")

    # Keep training rows with non-missing both targets (as your original script kept primary target)
    # We'll drop rows missing ISCO (target 1) for model training — follow the original behavior.
    df_train = df_train.dropna(subset=[TARGET_COLS[0]]).copy()

    # Fill text columns and create combined_text
    df_train[TEXT_COLS] = df_train[TEXT_COLS].fillna("")
    df_train["combined_text"] = df_train[TEXT_COLS].apply(lambda x: " ".join(x.astype(str)), axis=1)

    df_test = df_test.copy()
    df_test[TEXT_COLS] = df_test[TEXT_COLS].fillna("")
    df_test["combined_text"] = df_test[TEXT_COLS].apply(lambda x: " ".join(x.astype(str)), axis=1)

    return df_train, df_test



"""Preprocessing components used inside the pipeline (TF-IDF + code OHE)."""

import re, string
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from coder.config import (
    MAX_WORD_FEATURES, WORD_NGRAM_RANGE, WORD_MIN_DF,
    USE_DOMAIN_NORMALIZER, CODE_FEAT_COLS
)

class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        return pd.Series(X).astype(str).str.lower().str.translate(str.maketrans("", "", string.punctuation))

class DomainNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.replacements = [
            (re.compile(r"\bsheeps\b"), "sheep"),
            (re.compile(r"\bcaws\b"), "cows"),
            (re.compile(r"\bkuli[\-\s]?kuli\b"), "kuli_kuli"),
            (re.compile(r"\bnon[o0]\b"), "nono"),
            (re.compile(r"\btuwo?n?\b"), "tuwo"),
        ]
    def fit(self, X, y=None): return self
    def transform(self, X):
        s = pd.Series(X).astype(str)
        for pat, repl in self.replacements:
            s = s.str.replace(pat, repl, regex=True)
        return s

def _make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

def build_preprocessor():
    text_steps = [("clean", TextCleaner())]
    if USE_DOMAIN_NORMALIZER:
        text_steps.append(("domain", DomainNormalizer()))
    text_steps.append(("tfidf", TfidfVectorizer(
        max_features=MAX_WORD_FEATURES,
        ngram_range=WORD_NGRAM_RANGE,
        min_df=WORD_MIN_DF,
        sublinear_tf=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w[\w\-]+\b"
    )))

    text_pipe = Pipeline(text_steps)

    code_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", _make_ohe())
    ])

    pre = ColumnTransformer([
        ("text", text_pipe, "combined_text"),
        ("codes", code_pipe, CODE_FEAT_COLS)
    ], remainder="drop", sparse_threshold=1.0)

    return pre





"""Model building and grid search helper (multi-output)"""

from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from coder.preprocessing import build_preprocessor
from coder.config import RANDOM_STATE, CV_FOLDS, N_JOBS, VERBOSE, SCORING

def build_base_pipeline():
    """Build a pipeline producing features then multi-output classifier."""
    # base estimator used for each target
    base_clf = LogisticRegression(max_iter=800, class_weight="balanced", random_state=RANDOM_STATE, tol=1e-3)
    multi = MultiOutputClassifier(base_clf, n_jobs=1)  # n_jobs=1 to avoid Windows pickle issues
    pipe = Pipeline([("pre", build_preprocessor()), ("clf", multi)])
    return pipe

def build_param_grid():
    # We tune the underlying estimator via parameter names "clf__estimator__..." (MultiOutputClassifier wraps estimator)
    grid = [{
        "clf__estimator__solver": ["lbfgs", "liblinear"],
        "clf__estimator__C": [0.5, 1.0, 2.0, 5.0],
        "clf__estimator__penalty": ["l2"],
    }]
    return grid

def run_grid_search(X, y_multi, output_dir):
    """
    Run GridSearchCV for the multi-output pipeline.
    y_multi must be a DataFrame with two columns (ISCO, ISIC)
    """
    pipe = build_base_pipeline()
    grid = build_param_grid()
    # We do cross-validation on the ISCO label (primary) for splitting; for multioutput we still feed both outputs to fit()
    # Use StratifiedKFold on the first target for splits (best-effort)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    # GridSearchCV expects y to be 1D for stratified CV; we pass the first target for splitting but call fit with multi-target
    gs = GridSearchCV(estimator=pipe, param_grid=grid, cv=skf, scoring=SCORING, n_jobs=N_JOBS, refit=True, verbose=VERBOSE)
    # GridSearchCV will internally use X and y for fitting; to ensure it sees the multi-output targets we pass y as the first column
    # Workaround: fit with X and the first target for CV splits, then refit manually on best params with full multi-output labels.
    primary_y = y_multi.iloc[:, 0]
    gs.fit(X, primary_y)
    best_params = gs.best_params_
    # Rebuild pipeline with best params and refit using full multi-output targets
    best_pipe = build_base_pipeline()
    # set params
    best_pipe.set_params(**best_params)
    best_pipe.fit(X, y_multi)  # fit with multi-output DataFrame
    return gs, best_pipe




"""Evaluation and reporting for multi-output predictions (simple combined report)."""

import re
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

def normalize_code_str(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)
    digits = re.sub(r"\D", "", s)
    return digits.zfill(4) if digits else s

def make_combined_classification_report(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute classification_report for each target column and combine
    into a single dataframe with a column indicating the target name.
    """
    parts = []
    for col in y_true_df.columns:
        rpt = classification_report(y_true_df[col], y_pred_df[col], digits=4, output_dict=True, zero_division=0)
        df_rpt = pd.DataFrame(rpt).transpose().reset_index().rename(columns={"index":"label"})
        df_rpt.insert(0, "target", col)
        parts.append(df_rpt)
    combined = pd.concat(parts, axis=0, ignore_index=True)
    return combined

def save_predictions_with_confidence(df_input, pred_df, proba_list, output_path, isco_name, isic_name):
    """
    Save the original test DataFrame augmented with predictions and confidences.
    proba_list: list of predict_proba arrays (one per estimator) or None
    """
    out = df_input.copy()
    out[f"pred_{isco_name}"] = pred_df[isco_name].astype(str)
    out[f"pred_{isic_name}"] = pred_df[isic_name].astype(str)

    # confidences: proba_list is a list/tuple of arrays [n_samples, n_classes]
    try:
        # If proba_list is list-like with arrays for each output:
        if proba_list is not None and isinstance(proba_list, (list, tuple)):
            # For each output, take max prob across columns per sample
            conf_isco = np.nan
            conf_isic = np.nan
            if len(proba_list) >= 1 and proba_list[0] is not None:
                conf_isco = np.max(proba_list[0], axis=1)
            if len(proba_list) >= 2 and proba_list[1] is not None:
                conf_isic = np.max(proba_list[1], axis=1)
            out[f"conf_{isco_name}"] = conf_isco
            out[f"conf_{isic_name}"] = conf_isic
    except Exception:
        out["conf_failed"] = True

    out.to_csv(output_path, index=False)




"""
Main script that runs the multi-output (ISCO + ISIC) pipeline and writes:
- one combined classification report (both targets)
- predictions CSV with both predicted columns & confidences
- saved model
- run.log in output/
"""

from pathlib import Path
import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.base import clone
from coder.config import (
    TARGET_ISCO, TARGET_ISIC, TARGET_COLS,
    CODE_FEAT_COLS, TEXT_COLS,
    MIN_SAMPLES_PER_CLASS_TUNING, CV_FOLDS
)
from coder.utils import setup_logging, timestamp
from coder.data_utils import load_data
from coder.model import run_grid_search
from coder.evaluate import (
    normalize_code_str,
    make_combined_classification_report,
    save_predictions_with_confidence
)

def main():
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = timestamp()
    setup_logging(out_dir)
    logging.info("Starting multi-output ISCO+ISIC pipeline...")

    # Load data
    df_train, df_test = load_data()
    logging.info(f"Loaded train: {len(df_train)} rows, test: {len(df_test)} rows")

    # Prepare modeling inputs
    X_full = df_train[["combined_text"] + CODE_FEAT_COLS]
    # Build multi-output y DataFrame (ISCO, ISIC). If ISIC is missing for some rows, we'll keep as string 'nan'
    y_full = pd.DataFrame({
        TARGET_ISCO: df_train[TARGET_ISCO].astype(str),
        TARGET_ISIC: df_train[TARGET_ISIC].astype(str)
    })

    # Filter rare classes for tuning using ISCO (primary) — same as your existing logic
    vc = y_full[TARGET_ISCO].value_counts()
    rare_for_tuning = vc[vc < MIN_SAMPLES_PER_CLASS_TUNING].index.tolist()
    df_filtered = df_train[~df_train[TARGET_ISCO].astype(str).isin(rare_for_tuning)].copy()
    Xf = df_filtered[["combined_text"] + CODE_FEAT_COLS]
    yf = pd.DataFrame({
        TARGET_ISCO: df_filtered[TARGET_ISCO].astype(str),
        TARGET_ISIC: df_filtered[TARGET_ISIC].astype(str)
    })
    logging.info(f"Filtered for tuning: {len(df_filtered)} rows, classes kept: {yf[TARGET_ISCO].nunique()}")

    # Stratified hold-out (by ISCO)
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tr_idx, ho_idx = next(sss.split(Xf, yf[TARGET_ISCO]))
    X_tr, X_ho = Xf.iloc[tr_idx], Xf.iloc[ho_idx]
    y_tr = yf.iloc[tr_idx].reset_index(drop=True)
    y_ho = yf.iloc[ho_idx].reset_index(drop=True)

    # Run grid search (tune using ISCO stratified CV, then refit best on full multi-output)
    gs, best_pipe = run_grid_search(X_tr, y_tr, out_dir)

    logging.info(f"GridSearch done. Best params (from GridSearch): {gs.best_params_ if hasattr(gs,'best_params_') else 'N/A'}")

    # Evaluate on holdout (predict multi-output)
    y_ho_pred = best_pipe.predict(X_ho)
    # y_ho_pred is an array-like shape (n_samples, n_outputs)
    y_ho_pred_df = pd.DataFrame(y_ho_pred, columns=[TARGET_ISCO, TARGET_ISIC])
    # Normalize before reporting
    y_ho_true_norm = y_ho.applymap(normalize_code_str)
    y_ho_pred_norm = y_ho_pred_df.applymap(normalize_code_str)

    # Combined classification report (both targets) for D1 holdout
    combined_report_df = make_combined_classification_report(y_ho_true_norm, y_ho_pred_norm)
    out_d1 = out_dir / f"classification_report_D1_combined_{ts}.csv"
    combined_report_df.to_csv(out_d1, index=False)
    logging.info(f"Saved combined D1 classification report: {out_d1}")

    # Refit best model on full D1 (multi-output)
    final_model = clone(best_pipe)
    final_model.fit(X_full, y_full)
    model_path = out_dir / f"best_model_{ts}.joblib"
    joblib.dump(final_model, model_path)
    logging.info(f"Saved final model: {model_path}")

    # Predict on D2 (all rows)
    X2 = df_test[["combined_text"] + CODE_FEAT_COLS]
    preds = final_model.predict(X2)  # returns array (n_samples, 2)
    preds_df = pd.DataFrame(preds, columns=[TARGET_ISCO, TARGET_ISIC]).astype(str).applymap(normalize_code_str)

    # Try predict_proba per estimator: MultiOutputClassifier exposes estimators_ list each with predict_proba
    proba_list = None
    try:
        # gather predict_proba arrays (may not be available if base estimator doesn't implement it)
        proba_list = []
        for est in final_model.named_steps["clf"].estimators_:
            if hasattr(est, "predict_proba"):
                proba_list.append(est.predict_proba(final_model.named_steps["pre"].transform(X2)))
            else:
                proba_list.append(None)
    except Exception:
        proba_list = None

    # Save predictions with confidences
    pred_save = out_dir / f"predictions_D2_ISCO_ISIC_{ts}.csv"
    save_predictions_with_confidence(df_test, preds_df, proba_list, pred_save, TARGET_ISCO, TARGET_ISIC)
    logging.info(f"Saved D2 predictions: {pred_save}")

    # Evaluate on labelled subset of D2 (rows where primary target exists)
    if TARGET_ISCO in df_test.columns and df_test[TARGET_ISCO].notna().sum() > 0:
        df2_eval = df_test.dropna(subset=[TARGET_ISCO]).copy()
        X2_eval = df2_eval[["combined_text"] + CODE_FEAT_COLS]
        preds_eval = final_model.predict(X2_eval)
        preds_eval_df = pd.DataFrame(preds_eval, columns=[TARGET_ISCO, TARGET_ISIC]).astype(str).applymap(normalize_code_str)
        y2_true_df = pd.DataFrame({
            TARGET_ISCO: df2_eval[TARGET_ISCO].astype(str).map(normalize_code_str),
            TARGET_ISIC: df2_eval[TARGET_ISIC].astype(str).map(normalize_code_str)
        })
        combined_report_d2 = make_combined_classification_report(y2_true_df, preds_eval_df)
        out_d2 = out_dir / f"classification_report_D2_combined_{ts}.csv"
        combined_report_d2.to_csv(out_d2, index=False)
        logging.info(f"Saved combined D2 classification report: {out_d2}")
    else:
        logging.warning("No labelled D2 rows for evaluation; saved predictions only.")

    logging.info("Pipeline finished successfully.")

if __name__ == "__main__":
    main()



