from pathlib import Path
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import classification_report
import joblib

# local modules
from coder.config import (
    TRAIN_FILE, TRAIN_FILE_FALLBACK, TEST_FILE,
    TEXT_COLS, CODE_COLS, TARGET_COL,
    MIN_SAMPLES_PER_CLASS_TUNING, CV_FOLDS
)
from coder.utils import setup_logging, timestamp
from coder.model import run_grid_search, save_model


# -------------------------------------------------------------------
# 🧩 Helper Functions
# -------------------------------------------------------------------

def resolve_train_file():
    """Resolve training file path with fallback."""
    if TRAIN_FILE.exists():
        return TRAIN_FILE
    if TRAIN_FILE_FALLBACK.exists():
        logging.warning(f"'{TRAIN_FILE}' not found. Using fallback '{TRAIN_FILE_FALLBACK}'.")
        return TRAIN_FILE_FALLBACK
    raise FileNotFoundError(f"Could not find training file '{TRAIN_FILE}' or fallback.")


def resolve_test_file():
    """Resolve test file path."""
    if TEST_FILE.exists():
        return TEST_FILE
    raise FileNotFoundError(f"Could not find test file '{TEST_FILE}'.")


def normalize_code_str(x: str) -> str:
    """Normalize occupation codes to 4-digit zero-padded strings."""
    if x is None:
        return ""
    s = str(x).strip().replace(".0", "")
    digits = "".join([c for c in s if c.isdigit()])
    return digits.zfill(4) if digits else s


# -------------------------------------------------------------------
# 🚀 Main Pipeline
# -------------------------------------------------------------------

def main():
    out_dir = Path("output")
    setup_logging(out_dir)
    ts = timestamp()
    logging.info("🚀 Starting Reproducible ML Pipeline...")

    # --- Load Data ---
    train_path, test_path = resolve_train_file(), resolve_test_file()
    logging.info(f"Loading TRAIN: {train_path}")
    logging.info(f"Loading TEST: {test_path}")

    df1 = pd.read_excel(train_path, engine="openpyxl") if str(train_path).endswith((".xlsx", ".xls")) else pd.read_csv(train_path)
    df2 = pd.read_csv(test_path) if str(test_path).endswith((".csv", ".txt")) else pd.read_excel(test_path, engine="openpyxl")

    # --- Validate Columns ---
    required = [TARGET_COL] + TEXT_COLS + CODE_COLS
    for df, name in [(df1, "TRAIN"), (df2, "TEST")]:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")

    # --- Prepare Combined Text Feature ---
    df1 = df1.dropna(subset=[TARGET_COL]).copy()
    df1[TEXT_COLS] = df1[TEXT_COLS].fillna("")
    df1["combined_text"] = df1[TEXT_COLS].apply(lambda x: " ".join(x.astype(str)), axis=1)

    df2 = df2.copy()
    df2[TEXT_COLS] = df2[TEXT_COLS].fillna("")
    df2["combined_text"] = df2[TEXT_COLS].apply(lambda x: " ".join(x.astype(str)), axis=1)

    X1_full = df1[["combined_text"] + CODE_COLS]
    y1_full = df1[TARGET_COL].astype(str)

    # --- Filter Rare Classes for Tuning ---
    vc = y1_full.value_counts()
    rare = vc[vc < MIN_SAMPLES_PER_CLASS_TUNING].index.tolist()
    df1_filtered = df1[~df1[TARGET_COL].astype(str).isin(rare)].copy()

    X1f = df1_filtered[["combined_text"] + CODE_COLS]
    y1f = df1_filtered[TARGET_COL].astype(str)

    logging.info(f"Filtered to {len(df1_filtered)} samples across {y1f.nunique()} classes.")

    # --- Split Train/Holdout ---
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tr_idx, ho_idx = next(sss.split(X1f, y1f))
    X_train, X_holdout = X1f.iloc[tr_idx], X1f.iloc[ho_idx]
    y_train, y_holdout = y1f.iloc[tr_idx], y1f.iloc[ho_idx]

    # --- Run Grid Search ---
    gs, best_model = run_grid_search(X_train, y_train, out_dir)

    # --- Evaluate on D1 Holdout ---
    y_pred_ho = best_model.predict(X_holdout)
    rep_d1 = classification_report(
        y_holdout.map(normalize_code_str),
        pd.Series(y_pred_ho, index=y_holdout.index).map(normalize_code_str),
        digits=4,
        zero_division=0,
        output_dict=True
    )
    out_d1 = out_dir / f"classification_report_D1_{ts}.csv"
    pd.DataFrame(rep_d1).transpose().to_csv(out_d1, index=False)
    logging.info(f"[Info] Saved D1 report: {out_d1}")

    # --- Refit on Full D1 ---
    final_model = clone(best_model)
    final_model.fit(X1_full, y1_full)
    save_model(final_model, out_dir, ts)

    # --- Evaluate on D2 ---
    df2["target_norm"] = df2[TARGET_COL].astype(str).map(normalize_code_str)
    X2 = df2[["combined_text"] + CODE_COLS]
    preds = final_model.predict(X2)

    df2["predicted"] = pd.Series(preds, index=df2.index).map(normalize_code_str)
    pred_path = out_dir / f"predictions_D2_{ts}.csv"
    df2.to_csv(pred_path, index=False)
    logging.info(f"[Info] Predictions saved: {pred_path}")

    # --- Classification Report for D2 ---
    df2_eval = df2.dropna(subset=[TARGET_COL])
    rep_d2 = classification_report(
        df2_eval["target_norm"],
        df2_eval["predicted"],
        digits=4,
        zero_division=0,
        output_dict=True
    )
    out_d2 = out_dir / f"classification_report_D2_{ts}.csv"
    pd.DataFrame(rep_d2).transpose().to_csv(out_d2, index=False)
    logging.info(f"[Info] Saved D2 report: {out_d2}")

    logging.info("[✅ Done] Pipeline completed successfully.")


if __name__ == "__main__":
    main()

