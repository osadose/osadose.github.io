
# coder/config.py
from pathlib import Path
from typing import List

# Data / output
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Files (relative to DATA_DIR)
TRAIN_FILE = DATA_DIR / "NLFS_2024Q1_INDIVIDUAL 2.xlsx"
TRAIN_FILE_FALLBACK = DATA_DIR / "NLFS_2024Q1_INDIVIDUAL.xlsx"
TEST_FILE = DATA_DIR / "NLFS_2024_Q2.csv"

# Randomness / CV / scoring
RANDOM_STATE = 42
CV_FOLDS = 5
SCORING = "f1_weighted"
N_JOBS = 1         # keep 1 on Windows to avoid pickling issues
VERBOSE = 1

TEST_SIZE = 0.20   # 80/20 stratified holdout

# Text feature params
MAX_WORD_FEATURES = 10000
WORD_NGRAM_RANGE = (1, 3)
WORD_MIN_DF = 3

# Char n-grams (optional)
USE_CHAR_NGRAMS = False
CHAR_NGRAM_RANGE = (3, 5)
CHAR_MIN_DF = 2
CHAR_MAX_FEATURES = 30000

# Domain normalizer
USE_DOMAIN_NORMALIZER = True

# Elastic net grid (optional)
USE_ELASTICNET_GRID = False

# Columns
TEXT_COLS: List[str] = ["mjj2a", "mjj2b", "mjj3a", "mjj3b"]
CODE_COLS: List[str] = ["mjj2ccleanmaingroup", "mjj3ccleansection", "mjj3cclean"]
TARGET_COL = "mjj2cclean"

# CV feasibility threshold
MIN_SAMPLES_PER_CLASS_TRAIN_REQUIRED = 5
import math
MIN_SAMPLES_PER_CLASS_TUNING = math.ceil(
    MIN_SAMPLES_PER_CLASS_TRAIN_REQUIRED / (1.0 - TEST_SIZE)
)  # typically 7 with TEST_SIZE=0.2




# coder/utils.py
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Any

# Silence benign warnings (sklearn UserWarning / FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    logging.getLogger(__name__).info(f"Logging initialized. Log file: {log_file}")

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")



# coder/preprocessing.py
from sklearn.base import BaseEstimator, TransformerMixin
import re
import string
import pandas as pd
from typing import List
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from .config import (
    USE_DOMAIN_NORMALIZER, MAX_WORD_FEATURES, WORD_NGRAM_RANGE, WORD_MIN_DF,
    USE_CHAR_NGRAMS, CHAR_NGRAM_RANGE, CHAR_MIN_DF, CHAR_MAX_FEATURES,
    TEXT_COLS, CODE_COLS
)

class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        return pd.Series(X).astype(str).str.lower().str.translate(
            str.maketrans("", "", string.punctuation)
        )

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

def make_ohe(sparse=True) -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=sparse)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=sparse)

def build_preprocessor() -> ColumnTransformer:
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
            token_pattern=r"(?u)\b\w[\w\-]+\b",
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



# coder/model.py
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.base import clone
import joblib
import os
import logging
from typing import Any, Tuple
from .preprocessing import build_preprocessor
from .config import CV_FOLDS, SCORING, N_JOBS, VERBOSE, RANDOM_STATE, USE_ELASTICNET_GRID
from .utils import timestamp
from pathlib import Path

logger = logging.getLogger(__name__)

def build_pipeline() -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("classifier", LogisticRegression(
            max_iter=800,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            tol=1e-3,
            multi_class="multinomial",
            solver="lbfgs"
        )),
    ])

def build_param_grid():
    # Use lbfgs only to avoid liblinear future-warning; keep L2 penalties
    base_grid = [{
        "classifier__solver": ["lbfgs"],
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

def run_grid_search(X_tr, y_tr, output_dir: Path) -> Tuple[Any, dict]:
    pipe = build_pipeline()
    param_grid = build_param_grid()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    logger.info("[Info] Running GridSearchCV for Logistic Regression on D1-TRAIN (stratified CV)...")
    gs = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=cv, scoring=SCORING,
                      n_jobs=N_JOBS, refit=True, verbose=VERBOSE)
    gs.fit(X_tr, y_tr)
    logger.info(f"Best {SCORING}: {gs.best_score_:.4f}")
    logger.info(f"Best Params: {gs.best_params_}")
    return gs, gs.best_estimator_

def save_model(model: Any, output_dir: Path, tag: str = "best_model") -> Path:
    ts = timestamp()
    out_path = output_dir / f"{tag}_{ts}.joblib"
    joblib.dump(model, out_path)
    logger.info(f"Saved model to {out_path}")
    return out_path




# coder/evaluation.py
import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from typing import Any, Dict
from .utils import timestamp
from pathlib import Path

logger = logging.getLogger(__name__)

def compute_metrics_dict(y_true, y_pred) -> Dict[str, float]:
    report = classification_report(y_true, y_pred, digits=4, output_dict=True, zero_division=0)
    # return the dict (user can inspect)
    return report

def save_classification_report(report_dict: Dict, out_path: Path) -> None:
    df = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={"index": "label"})
    df.to_csv(out_path, index=False)
    logger.info(f"Saved classification_report to: {out_path}")

def save_metrics_json(metrics: Dict, out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved metrics JSON to: {out_path}")

def save_confusion_matrix(y_true, y_pred, labels, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels, ax=ax)
    plt.xticks(rotation=90)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved confusion matrix to: {out_path}")

def save_predictions_df(df: pd.DataFrame, out_path: Path) -> None:
    df.to_csv(out_path, index=False)
    logger.info(f"Saved predictions to: {out_path}")

# High-level evaluation wrapper that mirrors your original behavior
def evaluate_and_save_all(best_estimator: Any, gs_estimator: Any,
                          X1_tr, X1_ho, y1_tr, y1_ho,
                          X1_full, y1_full,
                          X1_cvfull, y1_cvfull,
                          df2, X2_all, X2_eval, config, output_dir: Path) -> None:
    """
    best_estimator: best estimator from GridSearch (fitted on X1_tr)
    gs_estimator: the GridSearchCV object (for best_score and params)
    X1_tr / X1_ho: filtered training and holdout (for tuning)
    X1_full / y1_full: full D1 (unfiltered)
    X1_cvfull / y1_cvfull: D1 subset with labels >= CV_FOLDS for fixed-param CV
    df2: original D2 dataframe with combined_text etc.
    X2_all: features for all D2 rows (for predictions export)
    X2_eval: features for labelled subset in D2 (for evaluation)
    config: module config reference
    output_dir: Path
    """
    ts = timestamp()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) D1 Filtered Hold-out (already used to select best model)
    y1_ho_pred = best_estimator.predict(X1_ho)
    # normalize codes before metrics if needed is done upstream; here assume y1_ho are normalized strings
    report_holdout = classification_report(y1_ho.astype(str).map(str), pd.Series(y1_ho_pred, index=y1_ho.index).astype(str).map(str), digits=4, zero_division=0, output_dict=True)
    save_classification_report(report_holdout, output_dir / f"classification_report_D1_filtered_holdout_{ts}.csv")

    # 2) Refit best model on FULL (done outside normally) — best_estimator should be refit on full before calling this
    # But we'll check: if best_estimator not fitted on full, user should have refit and passed that.
    # 3) D1 FULL fixed-params CV (drop classes with <CV_FOLDS)
    metrics_cv = []
    if len(y1_cvfull) > 0:
        # cv loop (no parallel)
        from sklearn.model_selection import StratifiedKFold
        cv2 = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
        for fold, (tr, va) in enumerate(cv2.split(X1_cvfull, y1_cvfull), 1):
            est = gs_estimator.best_estimator_.__class__(**gs_estimator.best_estimator_.get_params(deep=False))
            # clone pipeline properly - simplest: clone from sklearn.base.clone
            from sklearn.base import clone
            est = clone(gs_estimator.best_estimator_)
            est.fit(X1_cvfull.iloc[tr], y1_cvfull.iloc[tr])
            pred = est.predict(X1_cvfull.iloc[va])
            y_true = pd.Series(y1_cvfull.iloc[va]).astype(str)
            y_pred = pd.Series(pred, index=y1_cvfull.iloc[va].index).astype(str)
            rep = classification_report(y_true, y_pred, digits=4, output_dict=True, zero_division=0)
            metrics_cv.append(rep)
        # aggregate weighted f1 across folds
        mean_weighted_f1 = np.mean([m["weighted avg"]["f1-score"] for m in metrics_cv])
        logger.info(f"D1 Full (unfiltered): 5-fold CV (fixed params) — mean weighted f1 across folds: {mean_weighted_f1:.4f}")
        # save aggregated summary
        save_metrics_json({"d1_full_cv_mean_weighted_f1": mean_weighted_f1}, output_dir / f"d1_full_cv_summary_{ts}.json")

    # 4) D1 In-sample
    # user should provide y1_full_pred: but here compute if best_estimator was fitted on full
    try:
        y1_full_pred = best_estimator.predict(X1_full)
        report_insample = classification_report(y1_full.astype(str), pd.Series(y1_full_pred, index=y1_full.index).astype(str), digits=4, zero_division=0, output_dict=True)
        save_classification_report(report_insample, output_dir / f"classification_report_D1_full_insample_{ts}.csv")
    except Exception:
        logger.warning("Could not compute D1 in-sample predictions with best_estimator (maybe not fitted on full). Skipping in-sample.")

    # 5) External evaluation on D2
    # Predict ALL rows (with confidence if available)
    try:
        y2_all_pred = best_estimator.predict(X2_all)
        try:
            proba = best_estimator.predict_proba(X2_all)
            proba_max = proba.max(axis=1)
        except Exception:
            proba_max = [np.nan] * len(y2_all_pred)
        out = df2.copy()
        out["pred_mjj2cclean"] = pd.Series(y2_all_pred, index=df2.index).astype(str)
        out["pred_confidence"] = proba_max
        out_path = output_dir / f"predictions_D2_from_LG_{ts}.csv"
        save_predictions_df = out.to_csv
        out.to_csv(out_path, index=False)
        logger.info(f"Saved predictions for ALL D2 rows to: {out_path}")
    except Exception as e:
        logger.exception(f"Failed to predict and save D2 predictions: {e}")

    # 6) Evaluate only rows with labels (if present)
    if config.TARGET_COL in df2.columns and df2[config.TARGET_COL].notna().sum() > 0:
        df2_eval = df2.dropna(subset=[config.TARGET_COL]).copy()
        X2_eval = X2_eval = df2_eval[["combined_text"] + config.CODE_COLS]
        y2_eval_raw = df2_eval[config.TARGET_COL].astype(str).map(lambda x: x)
        y2_pred_eval_raw = best_estimator.predict(X2_eval)
        y2_pred_eval = pd.Series(y2_pred_eval_raw, index=df2_eval.index).astype(str)
        # Label coverage
        train_labels = set(y1_full.astype(str).unique())
        d2_labels = set(y2_eval_raw.unique())
        unseen = sorted(d2_labels - train_labels)
        if unseen:
            logger.info(f"{len(unseen)} D2 classes were unseen during D1 training (e.g., {unseen[:5]} ...). These have zero recall by definition.")
        # Full labelled D2 metrics
        rpt = classification_report(y2_eval_raw, y2_pred_eval, digits=4, output_dict=True, zero_division=0)
        save_classification_report(rpt, output_dir / f"classification_report_D2_full_{ts}.csv")
        # Overlap-only
        overlap = sorted(list(d2_labels & train_labels))
        mask_overlap = y2_eval_raw.isin(overlap)
        if mask_overlap.sum() > 0:
            rpt_ov = classification_report(y2_eval_raw[mask_overlap], y2_pred_eval[mask_overlap], digits=4, output_dict=True, zero_division=0)
            save_classification_report(rpt_ov, output_dir / f"classification_report_D2_overlap_{ts}.csv")
        # Hierarchical metrics at levels 1..4
        def to_level_simple(x, k):
            digits = "".join(filter(str.isdigit, str(x)))
            return digits.zfill(4)[:k] if digits else ""
        for k in [1,2,3,4]:
            y_true_k = y2_eval_raw.map(lambda c: to_level_simple(c, k))
            y_pred_k = y2_pred_eval.map(lambda c: to_level_simple(c, k))
            rpt_k = classification_report(y_true_k, y_pred_k, digits=4, output_dict=True, zero_division=0)
            save_classification_report(rpt_k, output_dir / f"classification_report_D2_hier_{k}digit_{ts}.csv")
        # Save metrics JSON and confusion matrix
        metrics = {
            "accuracy": accuracy_score(y2_eval_raw, y2_pred_eval),
            "examples": len(y2_eval_raw)
        }
        save_metrics_json(metrics, output_dir / f"D2_eval_summary_{ts}.json")
        cm_path = output_dir / f"D2_confusion_matrix_{ts}.png"
        save_confusion_matrix(y2_eval_raw, y2_pred_eval, sorted(list(set(y2_eval_raw))), cm_path)
    else:
        logger.info("No labelled rows in D2 to compute evaluation metrics.")




# main.py
from pathlib import Path
import logging
import sys

# local modules
from coder.config import TRAIN_FILE, TRAIN_FILE_FALLBACK, TEST_FILE, TEXT_COLS, CODE_COLS, TARGET_COL, MIN_SAMPLES_PER_CLASS_TUNING, CV_FOLDS
from coder.utils import setup_logging, timestamp
from coder.preprocessing import build_preprocessor
from coder.model import run_grid_search, save_model, build_pipeline
from coder.evaluation import evaluate_and_save_all
from coder import preprocessing
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.base import clone
import numpy as np
from pathlib import Path

def resolve_train_file():
    if TRAIN_FILE.exists():
        return TRAIN_FILE
    if TRAIN_FILE_FALLBACK.exists():
        logging.warning(f"'{TRAIN_FILE}' not found. Falling back to '{TRAIN_FILE_FALLBACK}'.")
        return TRAIN_FILE_FALLBACK
    raise FileNotFoundError(f"Could not find '{TRAIN_FILE}' or fallback.")

def resolve_test_file():
    if TEST_FILE.exists():
        return TEST_FILE
    raise FileNotFoundError(f"Could not find test file '{TEST_FILE}'.")

def main():
    out_dir = Path("output")
    setup_logging(out_dir)
    ts = timestamp()
    logging.info("Starting ML pipeline...")

    train_path = resolve_train_file()
    test_path = resolve_test_file()

    logging.info(f"Loading TRAIN from: {train_path}")
    if str(train_path).lower().endswith((".xlsx", ".xls")):
        df1 = pd.read_excel(train_path, engine="openpyxl")
    else:
        df1 = pd.read_csv(train_path)

    logging.info(f"Loading TEST from: {test_path}")
    if str(test_path).lower().endswith((".csv", ".txt")):
        df2 = pd.read_csv(test_path)
    else:
        df2 = pd.read_excel(test_path, engine="openpyxl")

    # Validate columns
    required = [TARGET_COL] + TEXT_COLS + CODE_COLS
    missing_train = [c for c in required if c not in df1.columns]
    missing_test = [c for c in required if c not in df2.columns]
    if missing_train:
        raise ValueError(f"TRAIN missing expected column(s): {missing_train}")
    if missing_test:
        raise ValueError(f"TEST missing expected column(s): {missing_test}")

    # Prepare combined_text fields
    df1 = df1.dropna(subset=[TARGET_COL]).copy()
    df1[TEXT_COLS] = df1[TEXT_COLS].fillna("")
    df1["combined_text"] = df1[TEXT_COLS].apply(lambda x: " ".join(x.astype(str)), axis=1)

    df2 = df2.copy()
    df2[TEXT_COLS] = df2[TEXT_COLS].fillna("")
    df2["combined_text"] = df2[TEXT_COLS].apply(lambda x: " ".join(x.astype(str)), axis=1)

    X1_full = df1[["combined_text"] + CODE_COLS]
    y1_full = df1[TARGET_COL].astype(str)

    # Filter rare classes for tuning
    vc = y1_full.value_counts()
    rare_for_tuning = vc[vc < MIN_SAMPLES_PER_CLASS_TUNING].index.tolist()
    df1_filtered = df1[~df1[TARGET_COL].astype(str).isin(rare_for_tuning)].copy()
    X1f = df1_filtered[["combined_text"] + CODE_COLS]
    y1f = df1_filtered[TARGET_COL].astype(str)

    logging.info(f"MIN_SAMPLES_PER_CLASS_TUNING = {MIN_SAMPLES_PER_CLASS_TUNING} (derived from TRAIN requirement with TEST_SIZE)")
    logging.info(f"Filtering {len(rare_for_tuning)} classes with <{MIN_SAMPLES_PER_CLASS_TUNING} samples for tuning.")
    logging.info(f"Remaining samples for tuning: {len(df1_filtered)}")
    logging.info(f"Classes kept for tuning: {y1f.nunique()} / {y1_full.nunique()}")

    if y1f.nunique() < CV_FOLDS:
        raise ValueError(f"[Abort] Only {y1f.nunique()} classes remain after filtering, but {CV_FOLDS}-fold stratified CV requires at least {CV_FOLDS} classes.")

    # Stratified hold-out on filtered D1
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tr_idx, ho_idx = next(sss.split(X1f, y1f))
    X1_tr, X1_ho = X1f.iloc[tr_idx], X1f.iloc[ho_idx]
    y1_tr, y1_ho = y1f.iloc[tr_idx], y1f.iloc[ho_idx]

    # GridSearch on X1_tr
    gs, best_on_tr = run_grid_search(X1_tr, y1_tr, out_dir)

    # Evaluate on filtered D1 holdout (print metrics)
    y1_ho_pred = best_on_tr.predict(X1_ho)
    # normalize to 4-digit form if you want to keep same behavior; reusing simple normalization:
    def normalize_code_str(x: str) -> str:
        if x is None:
            return ""
        s = str(x).strip()
        s = s.replace(".0", "")
        digits = "".join([c for c in s if c.isdigit()])
        if digits == "":
            return s
        return digits.zfill(4)
    y1_ho_true_norm4 = y1_ho.astype(str).map(normalize_code_str)
    y1_ho_pred_norm4 = pd.Series(y1_ho_pred, index=y1_ho.index).astype(str).map(normalize_code_str)
    # compute and print
    from sklearn.metrics import classification_report
    rep = classification_report(y1_ho_true_norm4, y1_ho_pred_norm4, digits=4, zero_division=0)
    out_path = out_dir / f"classification_report_D1_filtered_holdout_{ts}.csv"
    pd.DataFrame(rep).transpose().reset_index().rename(columns={"index":"label"}).to_csv(out_path, index=False)
    logging.info(f"[Info] classification_report saved to: {out_path}")

    # Refit best_on_full on FULL unfiltered D1
    best_on_full = clone(best_on_tr)
    best_on_full.fit(X1_full, y1_full)
    # save model
    save_path = out_dir / f"best_model_{ts}.joblib"
    import joblib
    joblib.dump(best_on_full, save_path)
    logging.info(f"Saved best model to {save_path}")

    # D1 FULL fixed-params CV (drop labels with <CV_FOLDS)
    counts_full = y1_full.value_counts()
    ok_labels = counts_full[counts_full >= CV_FOLDS].index.tolist()
    mask_cv_full = y1_full.isin(ok_labels)
    X1_cvfull = X1_full[mask_cv_full]
    y1_cvfull = y1_full[mask_cv_full]

    logging.info(f"D1 FULL CV (fixed params): keeping {len(y1_cvfull)} rows, dropping {len(y1_full)-len(y1_cvfull)} with labels <{CV_FOLDS} samples.")

    # 5-fold CV loop with cloned best_on_full
    from sklearn.model_selection import StratifiedKFold
    cv2 = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    metrics_cv = []
    for fold, (tr, va) in enumerate(cv2.split(X1_cvfull, y1_cvfull), 1):
        est = clone(best_on_full)
        est.fit(X1_cvfull.iloc[tr], y1_cvfull.iloc[tr])
        pred = est.predict(X1_cvfull.iloc[va])
        y_true = pd.Series(y1_cvfull.iloc[va]).astype(str).map(normalize_code_str)
        y_pred = pd.Series(pred, index=y1_cvfull.iloc[va].index).astype(str).map(normalize_code_str)
        from sklearn.metrics import f1_score
        m_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        metrics_cv.append(m_f1)
        logging.info(f"D1 Full CV Fold {fold} weighted f1: {m_f1:.4f}")
    if metrics_cv:
        logging.info(f"D1 Full (unfiltered): 5-fold CV (fixed params) — mean weighted f1 across folds: {np.mean(metrics_cv):.4f}")

    # D1 in-sample
    y1_full_pred = best_on_full.predict(X1_full)
    y1_full_true_norm4 = y1_full.astype(str).map(normalize_code_str)
    y1_full_pred_norm4 = pd.Series(y1_full_pred, index=y1_full.index).astype(str).map(normalize_code_str)
    rep_full_in = classification_report(y1_full_true_norm4, y1_full_pred_norm4, digits=4, zero_division=0)
    out_path_full = out_dir / f"classification_report_D1_full_insample_{ts}.csv"
    pd.DataFrame(rep_full_in).transpose().reset_index().rename(columns={"index":"label"}).to_csv(out_path_full, index=False)
    logging.info(f"[Info] D1 in-sample report saved to: {out_path_full}")

    # External evaluation on D2
    df2["target_norm4"] = df2[TARGET_COL].astype(str).map(lambda x: normalize_code_str(x))
    df2_eval = df2.dropna(subset=[TARGET_COL]).copy()

    X2_all = df2[["combined_text"] + CODE_COLS]
    y2_all_pred = best_on_full.predict(X2_all)
    y2_all_pred_norm4 = pd.Series(y2_all_pred, index=df2.index).astype(str).map(normalize_code_str)
    try:
        proba = best_on_full.predict_proba(X2_all)
        proba_max = proba.max(axis=1)
    except Exception:
        proba_max = np.full(shape=(len(X2_all),), fill_value=np.nan)

    out = df2.copy()
    out["pred_mjj2cclean"] = y2_all_pred_norm4
    out["pred_confidence"] = proba_max
    pred_save = out_dir / f"predictions_D2_from_LG_{ts}.csv"
    out.to_csv(pred_save, index=False)
    logging.info(f"\n[Info] Saved predictions for ALL D2 rows to: {pred_save}")

    if len(df2_eval) == 0:
        logging.warning("[Warn] No non-missing targets in D2 to evaluate metrics.")
    else:
        X2_eval = df2_eval[["combined_text"] + CODE_COLS]
        y2_eval_raw = df2_eval["target_norm4"]
        y2_pred_eval_raw = best_on_full.predict(X2_eval)
        y2_pred_eval = pd.Series(y2_pred_eval_raw, index=df2_eval.index).astype(str).map(normalize_code_str)

        train_labels_norm4 = set(y1_full.astype(str).map(normalize_code_str).unique())
        d2_labels_norm4 = set(y2_eval_raw.unique())
        unseen_in_train = sorted(d2_labels_norm4 - train_labels_norm4)
        if unseen_in_train:
            logging.info(f"[Info] {len(unseen_in_train)} D2 classes were unseen during D1 training (e.g., {unseen_in_train[:20]})")

        logging.info(f"Rows in D2: {len(df2)} | with labels: {len(df2_eval)} | missing labels: {df2[TARGET_COL].isna().sum()}")

        # Full labelled D2 metrics
        rep_d2 = classification_report(y2_eval_raw, y2_pred_eval, digits=4, zero_division=0, output_dict=True)
        pd.DataFrame(rep_d2).transpose().reset_index().rename(columns={"index":"label"}).to_csv(out_dir / f"classification_report_D2_full_{ts}.csv", index=False)
        logging.info(f"[Info] classification_report saved to: {out_dir / f'classification_report_D2_full_{ts}.csv'}")

        # Overlap-only
        overlap_labels = sorted(d2_labels_norm4 & train_labels_norm4)
        mask_overlap = y2_eval_raw.isin(overlap_labels)
        if mask_overlap.sum() > 0:
            y2_eval_overlap = y2_eval_raw[mask_overlap]
            y2_pred_overlap = y2_pred_eval[mask_overlap]
            rep_overlap = classification_report(y2_eval_overlap, y2_pred_overlap, digits=4, zero_division=0, output_dict=True)
            pd.DataFrame(rep_overlap).transpose().reset_index().rename(columns={"index":"label"}).to_csv(out_dir / f"classification_report_D2_overlap_{ts}.csv", index=False)
            logging.info(f"[Info] classification_report saved to: {out_dir / f'classification_report_D2_overlap_{ts}.csv'}")
        else:
            logging.info("[Info] No overlap between D2 and D1 labels (unexpected).")

        # Hierarchical metrics
        def to_level(code_str: str, k: int) -> str:
            s = normalize_code_str(code_str)
            return s[:k] if s else ""

        for k in [1,2,3,4]:
            y_true_k = y2_eval_raw.map(lambda c: to_level(c, k))
            y_pred_k = y2_pred_eval.map(lambda c: to_level(c, k))
            rpt_k = classification_report(y_true_k, y_pred_k, digits=4, zero_division=0, output_dict=True)
            pd.DataFrame(rpt_k).transpose().reset_index().rename(columns={"index":"label"}).to_csv(out_dir / f"classification_report_D2_hier_{k}digit_{ts}.csv", index=False)
            logging.info(f"[Info] classification_report saved to: {out_dir / f'classification_report_D2_hier_{k}digit_{ts}.csv'}")

    logging.info(f"\n[Done] Elapsed.")
    return

if __name__ == "__main__":
    main()



