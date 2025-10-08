

# main.py
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import classification_report, f1_score
import joblib
import sys

# local modules
from coder.config import (
    TRAIN_FILE, TRAIN_FILE_FALLBACK, TEST_FILE,
    TEXT_COLS, CODE_COLS, TARGET_COL,
    MIN_SAMPLES_PER_CLASS_TUNING, CV_FOLDS
)
from coder.utils import setup_logging, timestamp
from coder.model import run_grid_search, save_model

# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------

def resolve_train_file():
    """Resolve training file with fallback."""
    if TRAIN_FILE.exists():
        return TRAIN_FILE
    if TRAIN_FILE_FALLBACK.exists():
        logging.warning(f"'{TRAIN_FILE}' not found. Falling back to '{TRAIN_FILE_FALLBACK}'.")
        return TRAIN_FILE_FALLBACK
    raise FileNotFoundError(f"Could not find '{TRAIN_FILE}' or fallback.")

def resolve_test_file():
    """Resolve test file path."""
    if TEST_FILE.exists():
        return TEST_FILE
    raise FileNotFoundError(f"Could not find test file '{TEST_FILE}'.")

def normalize_code_str(x: str) -> str:
    """Normalize occupation codes to 4-digit strings."""
    if x is None:
        return ""
    s = str(x).strip()
    s = s.replace(".0", "")
    digits = "".join([c for c in s if c.isdigit()])
    if digits == "":
        return s
    return digits.zfill(4)

# --------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------

def main():
    out_dir = Path("output")
    setup_logging(out_dir)
    ts = timestamp()
    logging.info("Starting ML pipeline...")

    # ----------------------------------------------------------------
    # Load data
    # ----------------------------------------------------------------
    train_path = resolve_train_file()
    test_path = resolve_test_file()

    logging.info(f"Loading TRAIN from: {train_path}")
    df1 = pd.read_excel(train_path, engine="openpyxl") if str(train_path).endswith((".xlsx", ".xls")) else pd.read_csv(train_path)

    logging.info(f"Loading TEST from: {test_path}")
    df2 = pd.read_csv(test_path) if str(test_path).endswith((".csv", ".txt")) else pd.read_excel(test_path, engine="openpyxl")

    # ----------------------------------------------------------------
    # Validate columns
    # ----------------------------------------------------------------
    required = [TARGET_COL] + TEXT_COLS + CODE_COLS
    missing_train = [c for c in required if c not in df1.columns]
    missing_test = [c for c in required if c not in df2.columns]
    if missing_train:
        raise ValueError(f"TRAIN missing expected column(s): {missing_train}")
    if missing_test:
        raise ValueError(f"TEST missing expected column(s): {missing_test}")

    # ----------------------------------------------------------------
    # Prepare text features
    # ----------------------------------------------------------------
    df1 = df1.dropna(subset=[TARGET_COL]).copy()
    df1[TEXT_COLS] = df1[TEXT_COLS].fillna("")
    df1["combined_text"] = df1[TEXT_COLS].apply(lambda x: " ".join(x.astype(str)), axis=1)

    df2 = df2.copy()
    df2[TEXT_COLS] = df2[TEXT_COLS].fillna("")
    df2["combined_text"] = df2[TEXT_COLS].apply(lambda x: " ".join(x.astype(str)), axis=1)

    X1_full = df1[["combined_text"] + CODE_COLS]
    y1_full = df1[TARGET_COL].astype(str)

    # ----------------------------------------------------------------
    # Filter rare classes for tuning
    # ----------------------------------------------------------------
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

    # ----------------------------------------------------------------
    # Stratified Hold-out
    # ----------------------------------------------------------------
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tr_idx, ho_idx = next(sss.split(X1f, y1f))
    X1_tr, X1_ho = X1f.iloc[tr_idx], X1f.iloc[ho_idx]
    y1_tr, y1_ho = y1f.iloc[tr_idx], y1f.iloc[ho_idx]

    # ----------------------------------------------------------------
    # Grid Search
    # ----------------------------------------------------------------
    gs, best_on_tr = run_grid_search(X1_tr, y1_tr, out_dir)

    # ----------------------------------------------------------------
    # Evaluate on filtered D1 holdout
    # ----------------------------------------------------------------
    y1_ho_pred = best_on_tr.predict(X1_ho)
    y1_ho_true_norm4 = y1_ho.astype(str).map(normalize_code_str)
    y1_ho_pred_norm4 = pd.Series(y1_ho_pred, index=y1_ho.index).astype(str).map(normalize_code_str)
    rep = classification_report(y1_ho_true_norm4, y1_ho_pred_norm4, digits=4, zero_division=0, output_dict=True)
    out_path = out_dir / f"classification_report_D1_filtered_holdout_{ts}.csv"
    pd.DataFrame(rep).transpose().reset_index().rename(columns={"index":"label"}).to_csv(out_path, index=False)
    logging.info(f"[Info] classification_report saved to: {out_path}")

    # ----------------------------------------------------------------
    # Refit best model on full D1
    # ----------------------------------------------------------------
    best_on_full = clone(best_on_tr)
    best_on_full.fit(X1_full, y1_full)
    save_path = out_dir / f"best_model_{ts}.joblib"
    joblib.dump(best_on_full, save_path)
    logging.info(f"Saved best model to {save_path}")

    # ----------------------------------------------------------------
    # D1 Full Cross-Validation
    # ----------------------------------------------------------------
    counts_full = y1_full.value_counts()
    ok_labels = counts_full[counts_full >= CV_FOLDS].index.tolist()
    mask_cv_full = y1_full.isin(ok_labels)
    X1_cvfull = X1_full[mask_cv_full]
    y1_cvfull = y1_full[mask_cv_full]

    logging.info(f"D1 FULL CV (fixed params): keeping {len(y1_cvfull)} rows, dropping {len(y1_full)-len(y1_cvfull)} with labels <{CV_FOLDS} samples.")

    cv2 = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    metrics_cv = []
    for fold, (tr, va) in enumerate(cv2.split(X1_cvfull, y1_cvfull), 1):
        est = clone(best_on_full)
        est.fit(X1_cvfull.iloc[tr], y1_cvfull.iloc[tr])
        pred = est.predict(X1_cvfull.iloc[va])
        y_true = pd.Series(y1_cvfull.iloc[va]).astype(str).map(normalize_code_str)
        y_pred = pd.Series(pred, index=y1_cvfull.iloc[va].index).astype(str).map(normalize_code_str)
        m_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        metrics_cv.append(m_f1)
        logging.info(f"D1 Full CV Fold {fold} weighted f1: {m_f1:.4f}")
    if metrics_cv:
        logging.info(f"D1 Full (unfiltered): 5-fold CV mean weighted f1: {np.mean(metrics_cv):.4f}")

    # ----------------------------------------------------------------
    # D1 In-sample
    # ----------------------------------------------------------------
    y1_full_pred = best_on_full.predict(X1_full)
    y1_full_true_norm4 = y1_full.astype(str).map(normalize_code_str)
    y1_full_pred_norm4 = pd.Series(y1_full_pred, index=y1_full.index).astype(str).map(normalize_code_str)
    rep_full_in = classification_report(y1_full_true_norm4, y1_full_pred_norm4, digits=4, zero_division=0, output_dict=True)
    out_path_full = out_dir / f"classification_report_D1_full_insample_{ts}.csv"
    pd.DataFrame(rep_full_in).transpose().reset_index().rename(columns={"index":"label"}).to_csv(out_path_full, index=False)
    logging.info(f"[Info] D1 in-sample report saved to: {out_path_full}")

    # ----------------------------------------------------------------
    # External evaluation on D2
    # ----------------------------------------------------------------
    df2["target_norm4"] = df2[TARGET_COL].astype(str).map(normalize_code_str)
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
    logging.info(f"[Info] Saved predictions for ALL D2 rows to: {pred_save}")

    # ----------------------------------------------------------------
    # D2 Evaluation
    # ----------------------------------------------------------------
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
            logging.info(f"[Info] {len(unseen_in_train)} D2 classes unseen in D1 training (e.g., {unseen_in_train[:20]})")

        rep_d2 = classification_report(y2_eval_raw, y2_pred_eval, digits=4, zero_division=0, output_dict=True)
        pd.DataFrame(rep_d2).transpose().reset_index().rename(columns={"index":"label"}).to_csv(out_dir / f"classification_report_D2_full_{ts}.csv", index=False)

        overlap_labels = sorted(d2_labels_norm4 & train_labels_norm4)
        mask_overlap = y2_eval_raw.isin(overlap_labels)
        if mask_overlap.sum() > 0:
            y2_eval_overlap = y2_eval_raw[mask_overlap]
            y2_pred_overlap = y2_pred_eval[mask_overlap]
            rep_overlap = classification_report(y2_eval_overlap, y2_pred_overlap, digits=4, zero_division=0, output_dict=True)
            pd.DataFrame(rep_overlap).transpose().reset_index().rename(columns={"index":"label"}).to_csv(out_dir / f"classification_report_D2_overlap_{ts}.csv", index=False)
        else:
            logging.info("[Info] No overlap between D2 and D1 labels.")

        def to_level(code_str: str, k: int) -> str:
            s = normalize_code_str(code_str)
            return s[:k] if s else ""

        for k in [1, 2, 3, 4]:
            y_true_k = y2_eval_raw.map(lambda c: to_level(c, k))
            y_pred_k = y2_pred_eval.map(lambda c: to_level(c, k))
            rpt_k = classification_report(y_true_k, y_pred_k, digits=4, zero_division=0, output_dict=True)
            pd.DataFrame(rpt_k).transpose().reset_index().rename(columns={"index":"label"}).to_csv(out_dir / f"classification_report_D2_hier_{k}digit_{ts}.csv", index=False)
            logging.info(f"[Info] classification_report saved to: classification_report_D2_hier_{k}digit_{ts}.csv")

    logging.info("[Done] Pipeline completed successfully.")

# --------------------------------------------------------------------
if __name__ == "__main__":
    main()


























