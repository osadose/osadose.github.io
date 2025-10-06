#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_LG_on_D2.py  —  Option A

Workflow
--------
1) D1 (Q1 Excel)
   a) Filter rare classes for CV feasibility (require >=7 total samples)
   b) StratifiedShuffleSplit (80/20) on the filtered D1
   c) GridSearchCV (5-fold stratified) on the TRAIN only
   d) Evaluate best model on the D1 HOLD-OUT (filtered set)

2) Refit for production
   e) Refit the best model (fixed hyperparameters) on the FULL (unfiltered) D1 — includes rare classes

3) Compare D1 results
   f) Report metrics on:
      - D1 Filtered Hold-out  (Step 1d)
      - D1 Full (unfiltered) 5-fold CV with fixed params (dropping only classes with <5 samples)
      - D1 Full (unfiltered) In-sample (fit-on-full, evaluated on full)  [clearly labeled]

4) External evaluation on D2 (Q2 CSV)
   g) Predict ALL rows, save 'predictions_D2_from_LG.csv' with pred + confidence
   h) Evaluate only rows with labels:
      - Full labelled subset
      - Overlap-only (labels present in both D1 and D2)
      - Hierarchical metrics at 1/2/3/4-digit levels
   i) Save 'classification_report_D2_full.csv'

Rationale
---------
- Filtering for tuning keeps CV stable/valid. After tuning, we refit on ALL D1 to cover rare classes.
- We produce comparable metrics so you can see the effect of including rare classes:
  * Filtered hold-out (in-domain)
  * Full D1 fixed-params CV (generalization w.r.t. all classes with >=5 samples)
  * Full D1 in-sample (upper bound; informative but not a generalization measure)
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import time
import math
import re
import string
from typing import List, Dict

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, balanced_accuracy_score
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# =============================================================================
# ============================ CONFIG (EDITABLE) ===============================
# =============================================================================

# ---- File paths
TRAIN_FILE = "NLFS_2024Q1_INDIVIDUAL 2.xlsx"
TRAIN_FILE_FALLBACK = "NLFS_2024Q1_INDIVIDUAL.xlsx"
TEST_FILE = "NLFS_2024_Q2.csv"

# ---- Randomness / CV / scoring
RANDOM_STATE = 42
CV_FOLDS = 5
SCORING = "f1_weighted"

# IMPORTANT on Windows: keep n_jobs=1 in GridSearchCV to avoid pickling errors
N_JOBS = 1
VERBOSE = 1

# For the D1 hold-out split (stratified)
TEST_SIZE = 0.20  # 80/20 train/test split on D1 (stratified)

# ---- Text features (words)
MAX_WORD_FEATURES = 10000
WORD_NGRAM_RANGE = (1, 3)
WORD_MIN_DF = 3

# ---- Optional: character n-grams (OFF by default; slower but robust)
USE_CHAR_NGRAMS = False
CHAR_NGRAM_RANGE = (3, 5)
CHAR_MIN_DF = 2
CHAR_MAX_FEATURES = 30000

# ---- Optional: cheap domain normalization
USE_DOMAIN_NORMALIZER = True

# ---- Optional: small elastic-net grid (saga) — OFF by default
USE_ELASTICNET_GRID = False

# ---- Column names expected in both D1 & D2
TEXT_COLS: List[str] = ["mjj2a", "mjj2b", "mjj3a", "mjj3b"]
CODE_COLS: List[str] = ["mjj2ccleanmaingroup", "mjj3ccleansection", "mjj3cclean"]
TARGET_COL: str = "mjj2cclean"

# ---- CV feasibility threshold for D1 training folds
MIN_SAMPLES_PER_CLASS_TRAIN_REQUIRED = 5
MIN_SAMPLES_PER_CLASS_TUNING = math.ceil(
    MIN_SAMPLES_PER_CLASS_TRAIN_REQUIRED / (1.0 - TEST_SIZE)
)
# With TEST_SIZE=0.20 => ceil(5/0.8)=7 samples required for tuning set.


# =============================================================================
# ============================== UTILITIES ====================================
# =============================================================================

def resolve_train_file() -> str:
    """Resolve D1 training file path with fallback."""
    if os.path.exists(TRAIN_FILE):
        return TRAIN_FILE
    if os.path.exists(TRAIN_FILE_FALLBACK):
        print(f"[Warn] '{TRAIN_FILE}' not found. Falling back to '{TRAIN_FILE_FALLBACK}'.")
        return TRAIN_FILE_FALLBACK
    raise FileNotFoundError(
        f"Could not find '{TRAIN_FILE}' or fallback '{TRAIN_FILE_FALLBACK}' in {os.getcwd()}"
    )

def resolve_test_file() -> str:
    """Resolve D2 file path."""
    if os.path.exists(TEST_FILE):
        return TEST_FILE
    raise FileNotFoundError(f"Could not find test file '{TEST_FILE}' in {os.getcwd()}")


class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Lowercase and strip punctuation; keeps digits/underscores/hyphens inside tokens.
    This reduces feature fragmentation due to casing/punctuation.
    """
    def fit(self, X, y=None): return self
    def transform(self, X):
        return pd.Series(X).astype(str).str.lower().str.translate(
            str.maketrans("", "", string.punctuation)
        )

class DomainNormalizer(BaseEstimator, TransformerMixin):
    """
    Lightweight normalization of a few domain-specific variants/spellings
    to reduce TF-IDF sparsity. Expand cautiously; keep it cheap.
    """
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

def make_ohe(sparse=True) -> OneHotEncoder:
    """Build a OneHotEncoder compatible with sklearn>=1.4 and older versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=sparse)  # >=1.4
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=sparse)         # older


# =============================================================================
# =========================== LABEL NORMALIZATION =============================
# =============================================================================

def normalize_code_str(x: str) -> str:
    """
    Normalize ISCO-like code strings for consistent comparison and hierarchy mapping.
    - Remove trailing '.0'
    - Keep digits only; if numeric, zero-pad to 4 digits for hierarchy slicing
    - If not numeric, return cleaned string.
    """
    if x is None:
        return ""
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)
    digits = re.sub(r"\D", "", s)
    if digits == "":
        return s
    return digits.zfill(4)

def to_level(code_str: str, k: int) -> str:
    """Map a normalized 4-digit code to a coarser hierarchy level k=1..4."""
    code = normalize_code_str(code_str)
    return code[:k] if code else ""


# =============================================================================
# ============================ PREPROCESSOR ===================================
# =============================================================================

def build_preprocessor() -> ColumnTransformer:
    """
    ColumnTransformer:
      - Text branch: TextCleaner -> (optional DomainNormalizer) -> TF-IDF (word n-grams)
      - Optional char n-gram branch
      - Code columns: impute + one-hot
    """
    text_steps = [("cleaner", TextCleaner())]
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
            token_pattern=r"(?u)\b\w[\w\-]+\b",  # keep hyphenated tokens
        ),
    ))
    word_text = ("word_text", Pipeline(text_steps), "combined_text")

    transformers = [word_text]

    if USE_CHAR_NGRAMS:
        char_text = (
            "char_text",
            Pipeline([
                ("cleaner", TextCleaner()),
                ("tfidf_char", TfidfVectorizer(
                    analyzer="char",
                    ngram_range=CHAR_NGRAM_RANGE,
                    min_df=CHAR_MIN_DF,
                    sublinear_tf=True,
                    max_features=CHAR_MAX_FEATURES,
                    strip_accents=None
                )),
            ]),
            "combined_text"
        )
        transformers.append(char_text)

    code_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_ohe(sparse=True)),
    ])
    transformers.append(("codes", code_transformer, CODE_COLS))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0
    )

def build_pipeline() -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("classifier", LogisticRegression(
            max_iter=800,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            tol=1e-3
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


# =============================================================================
# =============================== METRICS =====================================
# =============================================================================

def compute_metrics(y_true, y_pred, label="") -> Dict[str, float]:
    acc  = accuracy_score(y_true, y_pred)
    p_w  = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    r_w  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    p_mi = precision_score(y_true, y_pred, average="micro", zero_division=0)
    r_mi = recall_score(y_true, y_pred, average="micro", zero_division=0)
    f1_mi= f1_score(y_true, y_pred, average="micro", zero_division=0)
    p_ma = precision_score(y_true, y_pred, average="macro", zero_division=0)
    r_ma = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_ma= f1_score(y_true, y_pred, average="macro", zero_division=0)
    ba   = balanced_accuracy_score(y_true, y_pred)

    print(f"\n=== {label} ===")
    print(f"Accuracy:             {acc:.4f}")
    print(f"Precision (micro):    {p_mi:.4f}")
    print(f"Precision (macro):    {p_ma:.4f}")
    print(f"Precision (weighted): {p_w:.4f}")
    print(f"Recall (micro):       {r_mi:.4f}")
    print(f"Recall (macro):       {r_ma:.4f}")
    print(f"Recall (weighted):    {r_w:.4f}")
    print(f"F1-score (micro):     {f1_mi:.4f}")
    print(f"F1-score (macro):     {f1_ma:.4f}")
    print(f"F1-score (weighted):  {f1_w:.4f}")
    print(f"Balanced Accuracy:    {ba:.4f}")
    return {
        "accuracy":acc,"precision_micro":p_mi,"precision_macro":p_ma,
        "precision_weighted":p_w,"recall_micro":r_mi,"recall_macro":r_ma,
        "recall_weighted":r_w,"f1_micro":f1_mi,"f1_macro":f1_ma,"f1_weighted":f1_w,"balanced_accuracy":ba
    }


# =============================================================================
# ================================== MAIN =====================================
# =============================================================================

def main():
    t0 = time.time()

    # --------------------------
    # 1) Load D1 (train) & D2 (external test)
    # --------------------------
    train_path = resolve_train_file()
    test_path = resolve_test_file()
    print(f"[Info] Loading TRAIN from: {train_path}")
    print(f"[Info] Loading TEST  from: {test_path}")

    # D1
    if train_path.lower().endswith((".xlsx", ".xls")):
        df1 = pd.read_excel(train_path, engine="openpyxl")
    else:
        df1 = pd.read_csv(train_path)

    # D2
    if test_path.lower().endswith((".csv", ".txt")):
        df2 = pd.read_csv(test_path)
    else:
        df2 = pd.read_excel(test_path, engine="openpyxl")

    # --------------------------
    # 2) Validate & prepare columns
    # --------------------------
    required = [TARGET_COL] + TEXT_COLS + CODE_COLS
    missing_train = [c for c in required if c not in df1.columns]
    missing_test = [c for c in required if c not in df2.columns]
    if missing_train:
        raise ValueError(f"TRAIN missing expected column(s): {missing_train}")
    if missing_test:
        raise ValueError(f"TEST missing expected column(s): {missing_test}")

    # D1: drop missing target for modeling, build combined_text
    df1 = df1.dropna(subset=[TARGET_COL]).copy()
    df1[TEXT_COLS] = df1[TEXT_COLS].fillna("")
    df1["combined_text"] = df1[TEXT_COLS].apply(lambda x: " ".join(x.astype(str)), axis=1)

    # D2: keep missing targets (for prediction export), build combined_text
    df2 = df2.copy()
    df2[TEXT_COLS] = df2[TEXT_COLS].fillna("")
    df2["combined_text"] = df2[TEXT_COLS].apply(lambda x: " ".join(x.astype(str)), axis=1)

    # Full D1 (unfiltered) features/labels
    X1_full = df1[["combined_text"] + CODE_COLS]
    y1_full = df1[TARGET_COL].astype(str)

    # --------------------------
    # 3) Filter rare classes in D1 for tuning CV feasibility
    # --------------------------
    vc = y1_full.value_counts()
    rare_for_tuning = vc[vc < MIN_SAMPLES_PER_CLASS_TUNING].index.tolist()
    df1_filtered = df1[~df1[TARGET_COL].astype(str).isin(rare_for_tuning)].copy()
    X1f = df1_filtered[["combined_text"] + CODE_COLS]
    y1f = df1_filtered[TARGET_COL].astype(str)

    print(f"[Info] MIN_SAMPLES_PER_CLASS_TUNING = {MIN_SAMPLES_PER_CLASS_TUNING} "
          f"(derived from TRAIN requirement {MIN_SAMPLES_PER_CLASS_TRAIN_REQUIRED} with TEST_SIZE={TEST_SIZE})")
    print(f"[Info] Filtering {len(rare_for_tuning)} classes with <{MIN_SAMPLES_PER_CLASS_TUNING} samples for tuning.")
    print(f"[Info] Remaining samples for tuning: {len(df1_filtered)}")
    print(f"[Info] Classes kept for tuning: {y1f.nunique()} / {y1_full.nunique()}")

    if y1f.nunique() < CV_FOLDS:
        raise ValueError(
            f"[Abort] Only {y1f.nunique()} classes remain after filtering, "
            f"but {CV_FOLDS}-fold stratified CV requires at least {CV_FOLDS} classes. "
            f"Consider lowering MIN_SAMPLES_PER_CLASS_TUNING or TEST_SIZE."
        )

    # --------------------------
    # 4) STRATIFIED HOLD-OUT on filtered D1
    # --------------------------
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    tr_idx, ho_idx = next(sss.split(X1f, y1f))
    X1_tr, X1_ho = X1f.iloc[tr_idx], X1f.iloc[ho_idx]
    y1_tr, y1_ho = y1f.iloc[tr_idx], y1f.iloc[ho_idx]

    pipe = build_pipeline()
    param_grid = build_param_grid()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print("[Info] Running GridSearchCV for Logistic Regression on D1-TRAIN (stratified CV)...")
    gs = GridSearchCV(
        estimator=pipe, param_grid=param_grid, cv=cv, scoring=SCORING,
        n_jobs=N_JOBS, refit=True, verbose=VERBOSE
    )
    gs.fit(X1_tr, y1_tr)

    print("\n=== Best Model (Logistic Regression on D1-TRAIN) ===")
    print(f"Best {SCORING}: {gs.best_score_:.4f}")
    print(f"Best Params: {gs.best_params_}")

    # Evaluate on filtered D1 HOLD-OUT
    y1_ho_pred = gs.best_estimator_.predict(X1_ho)
    y1_ho_true_norm4 = y1_ho.astype(str).map(normalize_code_str)
    y1_ho_pred_norm4 = pd.Series(y1_ho_pred, index=y1_ho.index).astype(str).map(normalize_code_str)
    _ = compute_metrics(y1_ho_true_norm4, y1_ho_pred_norm4, label="D1 (Filtered) Stratified Hold-out (in-domain)")

    # Save detailed report (filtered hold-out)
    report_holdout = classification_report(
        y1_ho_true_norm4, y1_ho_pred_norm4, digits=4, zero_division=0, output_dict=True
    )
    pd.DataFrame(report_holdout).transpose().reset_index().rename(columns={"index":"label"}).to_csv(
        "classification_report_D1_filtered_holdout.csv", index=False
    )
    print("[Info] classification_report saved to: classification_report_D1_filtered_holdout.csv")

    # --------------------------
    # 5) Refit BEST model on FULL (unfiltered) D1  — includes rare classes
    # --------------------------
    best_on_full = clone(gs.best_estimator_)
    best_on_full.fit(X1_full, y1_full)

    # --------------------------
    # 6) D1 FULL (unfiltered) — additional comparisons
    #    6a) Fixed-params 5-fold CV on FULL D1:
    #        We must drop only classes with <CV_FOLDS samples to avoid CV errors.
    # --------------------------
    counts_full = y1_full.value_counts()
    ok_labels = counts_full[counts_full >= CV_FOLDS].index.tolist()
    mask_cv_full = y1_full.isin(ok_labels)
    X1_cvfull = X1_full[mask_cv_full]
    y1_cvfull = y1_full[mask_cv_full]

    print(f"[Info] D1 FULL CV (fixed params): keeping {len(y1_cvfull)} rows, "
          f"dropping {len(y1_full) - len(y1_cvfull)} with labels <{CV_FOLDS} samples.")

    # 5-fold CV loop (no parallel to avoid pickling issues)
    cv2 = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    metrics_cv = []
    for fold, (tr, va) in enumerate(cv2.split(X1_cvfull, y1_cvfull), 1):
        est = clone(best_on_full)
        est.fit(X1_cvfull.iloc[tr], y1_cvfull.iloc[tr])
        pred = est.predict(X1_cvfull.iloc[va])
        y_true = pd.Series(y1_cvfull.iloc[va]).astype(str).map(normalize_code_str)
        y_pred = pd.Series(pred, index=y1_cvfull.iloc[va].index).astype(str).map(normalize_code_str)
        m = compute_metrics(y_true, y_pred, label=f"D1 Full (unfiltered) CV Fold {fold}")
        metrics_cv.append(m)
    # Aggregate CV metrics
    if metrics_cv:
        agg = {k: np.mean([m[k] for m in metrics_cv]) for k in metrics_cv[0].keys()}
        print("\n=== D1 Full (unfiltered): 5-fold CV (fixed params) — mean across folds ===")
        for k, v in agg.items():
            print(f"{k:>20}: {v:.4f}")

    # --------------------------
    #    6b) In-sample metrics on FULL D1 (fit-on-full, evaluated on full)
    #        (Informative upper bound; not a generalization metric)
    # --------------------------
    y1_full_pred = best_on_full.predict(X1_full)
    y1_full_true_norm4 = y1_full.astype(str).map(normalize_code_str)
    y1_full_pred_norm4 = pd.Series(y1_full_pred, index=y1_full.index).astype(str).map(normalize_code_str)
    _ = compute_metrics(y1_full_true_norm4, y1_full_pred_norm4, label="D1 Full (unfiltered): In-sample (fit=eval)")

    # --------------------------
    # 7) External evaluation on D2
    # --------------------------
    # Add normalized target to D2 (kept for labelled subset)
    df2["target_norm4"] = df2[TARGET_COL].astype(str).map(normalize_code_str)
    df2_eval = df2.dropna(subset=[TARGET_COL]).copy()  # labelled rows only

    # Predict for ALL rows in D2 (for export)
    X2_all = df2[["combined_text"] + CODE_COLS]
    y2_all_pred = best_on_full.predict(X2_all)
    y2_all_pred_norm4 = pd.Series(y2_all_pred, index=df2.index).astype(str).map(normalize_code_str)

    # Try to include a simple confidence score
    try:
        proba = best_on_full.predict_proba(X2_all)
        proba_max = proba.max(axis=1)
    except Exception:
        proba_max = np.full(shape=(len(X2_all),), fill_value=np.nan)

    # Save predictions for ALL rows
    out = df2.copy()
    out["pred_mjj2cclean"] = y2_all_pred_norm4
    out["pred_confidence"] = proba_max
    out.to_csv("predictions_D2_from_LG.csv", index=False)
    print("\n[Info] Saved predictions for ALL D2 rows to: predictions_D2_from_LG.csv")

    # ---- Metrics on D2 (only rows that have labels)
    print("\n=== External Evaluation on D2 (rows with non-missing target) ===")
    if len(df2_eval) == 0:
        print("[Warn] No non-missing targets in D2 to evaluate metrics.")
    else:
        X2_eval = df2_eval[["combined_text"] + CODE_COLS]
        y2_eval_raw = df2_eval["target_norm4"]
        y2_pred_eval_raw = best_on_full.predict(X2_eval)
        y2_pred_eval = pd.Series(y2_pred_eval_raw, index=df2_eval.index).astype(str).map(normalize_code_str)

        # Label coverage diagnostics
        train_labels_norm4 = set(y1_full.astype(str).map(normalize_code_str).unique())
        d2_labels_norm4 = set(y2_eval_raw.unique())
        unseen_in_train = sorted(d2_labels_norm4 - train_labels_norm4)
        if unseen_in_train:
            sample = ", ".join(unseen_in_train[:20]) + (" ..." if len(unseen_in_train) > 20 else "")
            print(f"[Info] {len(unseen_in_train)} D2 classes were unseen during D1 training (e.g., {sample}). "
                  f"These have zero recall by definition.")

        print(f"Rows in D2: {len(df2)} | with labels: {len(df2_eval)} | missing labels: {df2[TARGET_COL].isna().sum()}")

        # ---- 7a) Full labelled D2 metrics
        _ = compute_metrics(y2_eval_raw, y2_pred_eval, label="D2 (full labelled subset)")

        # ---- 7b) Overlap-only metrics (restrict to labels present in both D1 and D2)
        overlap_labels = sorted(d2_labels_norm4 & train_labels_norm4)
        mask_overlap = y2_eval_raw.isin(overlap_labels)
        if mask_overlap.sum() > 0:
            y2_eval_overlap = y2_eval_raw[mask_overlap]
            y2_pred_overlap = y2_pred_eval[mask_overlap]
            _ = compute_metrics(y2_eval_overlap, y2_pred_overlap,
                                label=f"D2 Overlap-only (|classes|={len(overlap_labels)})")
        else:
            print("[Info] No overlap between D2 and D1 labels (unexpected).")

        # ---- 7c) Hierarchical metrics at 1-/2-/3-/4-digit
        for k in [1, 2, 3, 4]:
            y_true_k = y2_eval_raw.map(lambda c: to_level(c, k))
            y_pred_k = y2_pred_eval.map(lambda c: to_level(c, k))
            _ = compute_metrics(y_true_k, y_pred_k, label=f"D2 Hierarchical level: {k}-digit")

        # ---- Save a detailed classification report for D2
        report_d2 = classification_report(
            y2_eval_raw, y2_pred_eval, digits=4, zero_division=0, output_dict=True
        )
        pd.DataFrame(report_d2).transpose().reset_index().rename(columns={"index":"label"}).to_csv(
            "classification_report_D2_full.csv", index=False
        )
        print("[Info] classification_report saved to: classification_report_D2_full.csv")

    print(f"\n[Done] Elapsed: {time.strftime('%H:%M:%S', time.gmtime(time.time() - t0))}")


if __name__ == "__main__":
    main()
