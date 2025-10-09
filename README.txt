"""
Config — stores constants and parameters used throughout the ML pipeline.
"""

import math

# ---------------- File Paths ----------------
TRAIN_FILE = "NLFS_2024Q1_INDIVIDUAL 2.xlsx"
TRAIN_FILE_FALLBACK = "NLFS_2024Q1_INDIVIDUAL.xlsx"
TEST_FILE = "NLFS_2024_Q2.csv"

# ---------------- CV / Randomness ----------------
RANDOM_STATE = 42
CV_FOLDS = 5
SCORING = "f1_weighted"
N_JOBS = 1  # Set to 1 to avoid Windows pickling issues
VERBOSE = 1
TEST_SIZE = 0.20

# ---------------- Text & Code Features ----------------
MAX_WORD_FEATURES = 10000
WORD_NGRAM_RANGE = (1, 3)
WORD_MIN_DF = 3

USE_CHAR_NGRAMS = False
CHAR_NGRAM_RANGE = (3, 5)
CHAR_MIN_DF = 2
CHAR_MAX_FEATURES = 30000

USE_DOMAIN_NORMALIZER = True
USE_ELASTICNET_GRID = False

# ---------------- Data Columns ----------------
TEXT_COLS = ["mjj2a", "mjj2b", "mjj3a", "mjj3b"]
CODE_COLS = ["mjj2ccleanmaingroup", "mjj3ccleansection", "mjj3cclean"]
TARGET_COL = "mjj2cclean"

# ---------------- Class Filtering ----------------
MIN_SAMPLES_PER_CLASS_TRAIN_REQUIRED = 5
MIN_SAMPLES_PER_CLASS_TUNING = math.ceil(
    MIN_SAMPLES_PER_CLASS_TRAIN_REQUIRED / (1.0 - TEST_SIZE)
)



"""
Handles safe data loading, validation, and basic preparation.
"""

import os
import pandas as pd
from coder.config import TRAIN_FILE, TRAIN_FILE_FALLBACK, TEST_FILE, TARGET_COL, TEXT_COLS, CODE_COLS


def resolve_train_file() -> str:
    """Find the training file with fallback."""
    if os.path.exists(TRAIN_FILE):
        return TRAIN_FILE
    if os.path.exists(TRAIN_FILE_FALLBACK):
        print(f"[Warn] {TRAIN_FILE} not found. Falling back to {TRAIN_FILE_FALLBACK}")
        return TRAIN_FILE_FALLBACK
    raise FileNotFoundError(f"Could not find training data in current directory.")


def resolve_test_file() -> str:
    """Find the external test file."""
    if os.path.exists(TEST_FILE):
        return TEST_FILE
    raise FileNotFoundError(f"Could not find {TEST_FILE}")


def load_data():
    """Load both train (Excel/CSV) and test (Excel/CSV) files."""
    train_path = resolve_train_file()
    test_path = resolve_test_file()

    print(f"[Info] Loading TRAIN from: {train_path}")
    df_train = (
        pd.read_excel(train_path, engine="openpyxl")
        if train_path.endswith((".xlsx", ".xls"))
        else pd.read_csv(train_path)
    )

    print(f"[Info] Loading TEST from: {test_path}")
    df_test = (
        pd.read_excel(test_path, engine="openpyxl")
        if test_path.endswith((".xlsx", ".xls"))
        else pd.read_csv(test_path)
    )

    # Validation
    required = [TARGET_COL] + TEXT_COLS + CODE_COLS
    missing_train = [c for c in required if c not in df_train.columns]
    missing_test = [c for c in required if c not in df_test.columns]
    if missing_train:
        raise KeyError(f"Training data missing columns: {missing_train}")
    if missing_test:
        raise KeyError(f"Test data missing columns: {missing_test}")

    return df_train, df_test



"""
Handles text cleaning, domain normalization, and combined feature creation.
"""

import re
import string
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from coder.config import TARGET_COL, TEXT_COLS


# --- Simple Text Cleaner ---
class TextCleaner(BaseEstimator, TransformerMixin):
    """Lowercase and remove punctuation."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        return pd.Series(X).astype(str).str.lower().str.translate(str.maketrans("", "", string.punctuation))


# --- Domain Normalizer ---
class DomainNormalizer(BaseEstimator, TransformerMixin):
    """Normalize domain-specific variants (sheeps→sheep, caws→cows, etc.)"""
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


# --- Combine Text Columns ---
def prepare_combined_text(df: pd.DataFrame) -> pd.DataFrame:
    """Create a combined text field for modeling."""
    df = df.copy()
    df[TEXT_COLS] = df[TEXT_COLS].fillna("")
    df["combined_text"] = df[TEXT_COLS].apply(lambda x: " ".join(x.astype(str)), axis=1)
    return df



"""
Build, tune, and train Logistic Regression pipeline.
"""

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

from coder.preprocessing import TextCleaner, DomainNormalizer
from coder.config import (
    CODE_COLS, RANDOM_STATE, SCORING, CV_FOLDS, N_JOBS, VERBOSE,
    MAX_WORD_FEATURES, WORD_NGRAM_RANGE, WORD_MIN_DF,
    USE_CHAR_NGRAMS, CHAR_NGRAM_RANGE, CHAR_MIN_DF, CHAR_MAX_FEATURES,
    USE_DOMAIN_NORMALIZER, USE_ELASTICNET_GRID
)


def make_ohe(sparse=True):
    """Safe OneHotEncoder wrapper for sklearn compatibility."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=sparse)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=sparse)


def build_preprocessor() -> ColumnTransformer:
    """Create the full ColumnTransformer pipeline."""
    text_steps = [("clean", TextCleaner())]
    if USE_DOMAIN_NORMALIZER:
        text_steps.append(("domain", DomainNormalizer()))
    text_steps.append((
        "tfidf_word",
        TfidfVectorizer(
            max_features=MAX_WORD_FEATURES,
            ngram_range=WORD_NGRAM_RANGE,
            min_df=WORD_MIN_DF,
            sublinear_tf=True,
            strip_accents="unicode",
            token_pattern=r"(?u)\b\w[\w\-]+\b"
        )
    ))

    transformers = [
        ("word_text", Pipeline(text_steps), "combined_text"),
        ("codes", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", make_ohe(sparse=True))
        ]), CODE_COLS)
    ]

    if USE_CHAR_NGRAMS:
        transformers.append(("char_text", Pipeline([
            ("clean", TextCleaner()),
            ("tfidf_char", TfidfVectorizer(
                analyzer="char",
                ngram_range=CHAR_NGRAM_RANGE,
                min_df=CHAR_MIN_DF,
                max_features=CHAR_MAX_FEATURES,
                sublinear_tf=True
            ))
        ]), "combined_text"))

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=1.0)


def build_pipeline():
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("classifier", LogisticRegression(
            max_iter=800, class_weight="balanced",
            random_state=RANDOM_STATE, tol=1e-3
        ))
    ])


def build_param_grid():
    base = [{
        "classifier__solver": ["lbfgs", "liblinear"],
        "classifier__C": [0.5, 1.0, 2.0, 5.0, 10.0],
        "classifier__penalty": ["l2"]
    }]
    if USE_ELASTICNET_GRID:
        base.append({
            "classifier__solver": ["saga"],
            "classifier__penalty": ["elasticnet"],
            "classifier__l1_ratio": [0.1, 0.5],
            "classifier__C": [0.5, 1.0]
        })
    return base



"""
Compute metrics, evaluate on D2, and save prediction outputs.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, balanced_accuracy_score
)


def normalize_code_str(x: str) -> str:
    """Normalize numeric codes to 4-digit strings."""
    if x is None: return ""
    s = str(x).strip().replace(".0", "")
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits.zfill(4) if digits else s


def to_level(code: str, k: int) -> str:
    """Return hierarchical level substring."""
    code = normalize_code_str(code)
    return code[:k] if code else ""


def compute_metrics(y_true, y_pred, label=""):
    """Compute and print standard metrics."""
    acc = accuracy_score(y_true, y_pred)
    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"\n=== {label} ===\nAccuracy={acc:.4f} | F1_w={f1w:.4f}")
    return {"accuracy": acc, "f1_weighted": f1w}


def evaluate_on_D2(model, df2, y_train_norm4, code_cols, target_col, output_dir="output"):
    """Generate predictions for D2 and evaluate on labeled subset."""
    df2 = df2.copy()
    df2["target_norm4"] = df2[target_col].astype(str).map(normalize_code_str)

    X2_all = df2[["combined_text"] + code_cols]
    preds = model.predict(X2_all)
    preds_norm4 = pd.Series(preds, index=df2.index).astype(str).map(normalize_code_str)

    # Confidence
    try:
        proba = model.predict_proba(X2_all)
        conf = proba.max(axis=1)
    except Exception:
        conf = np.nan

    df2["predicted_target"] = preds_norm4
    df2["confidence"] = conf
    df2.to_csv(f"{output_dir}/predictions_D2_from_LG.csv", index=False)

    df2_eval = df2.dropna(subset=[target_col])
    if df2_eval.empty:
        print("[Warn] No labels in D2 to evaluate.")
        return

    y_true = df2_eval["target_norm4"]
    y_pred = df2_eval["predicted_target"]
    compute_metrics(y_true, y_pred, label="D2 (labeled subset)")

    # Hierarchical
    for k in [1, 2, 3, 4]:
        compute_metrics(
            y_true.map(lambda c: to_level(c, k)),
            y_pred.map(lambda c: to_level(c, k)),
            label=f"D2 Hierarchical {k}-digit"
        )

    report = classification_report(y_true, y_pred, digits=4, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(f"{output_dir}/classification_report_D2_full.csv")




"""
Main entry — orchestrates the evaluation of Logistic Regression on D2.
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, StratifiedKFold
from sklearn.base import clone

from coder.config import *
from coder.data_utils import load_data
from coder.preprocessing import prepare_combined_text
from coder.model import build_pipeline, build_param_grid
from coder.evaluate import compute_metrics, normalize_code_str, evaluate_on_D2


def main():
    t0 = time.time()
    df1, df2 = load_data()

    # --- Preprocess text ---
    df1 = prepare_combined_text(df1).dropna(subset=[TARGET_COL])
    df2 = prepare_combined_text(df2)

    X1_full = df1[["combined_text"] + CODE_COLS]
    y1_full = df1[TARGET_COL].astype(str)

    # --- Filter rare classes ---
    vc = y1_full.value_counts()
    rare = vc[vc < MIN_SAMPLES_PER_CLASS_TUNING].index
    df1f = df1[~df1[TARGET_COL].isin(rare)].copy()
    X1f, y1f = df1f[["combined_text"] + CODE_COLS], df1f[TARGET_COL].astype(str)

    # --- Split train/hold-out ---
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    tr_idx, ho_idx = next(sss.split(X1f, y1f))
    X1_tr, X1_ho = X1f.iloc[tr_idx], X1f.iloc[ho_idx]
    y1_tr, y1_ho = y1f.iloc[tr_idx], y1f.iloc[ho_idx]

    pipe = build_pipeline()
    params = build_param_grid()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print("[Info] Running GridSearchCV...")
    gs = GridSearchCV(pipe, params, cv=cv, scoring=SCORING, n_jobs=N_JOBS, refit=True, verbose=VERBOSE)
    gs.fit(X1_tr, y1_tr)
    print(f"[Best] {SCORING}: {gs.best_score_:.4f} | Params: {gs.best_params_}")

    # --- Evaluate on hold-out ---
    y_pred_ho = gs.best_estimator_.predict(X1_ho)
    compute_metrics(
        y1_ho.astype(str).map(normalize_code_str),
        pd.Series(y_pred_ho, index=y1_ho.index).astype(str).map(normalize_code_str),
        label="D1 (Filtered) Hold-out"
    )

    # --- Refit on full D1 ---
    best_model = clone(gs.best_estimator_)
    best_model.fit(X1_full, y1_full)

    # --- External evaluation ---
    evaluate_on_D2(best_model, df2, y1_full.astype(str).map(normalize_code_str),
                   CODE_COLS, TARGET_COL, output_dir="output")

    print(f"\n[Done] Total runtime: {time.strftime('%H:%M:%S', time.gmtime(time.time() - t0))}")


if __name__ == "__main__":
    main()











































































