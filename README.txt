
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
    """
    Prepare training and test data for the ML pipeline.

    Args:
        df_train (pd.DataFrame): Raw training dataset
        df_test (pd.DataFrame): Raw test dataset

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
            - X_train: Prepared training features
            - y_train: Target labels for training
            - X_test: Prepared test features
            - y_test: Target labels for test (if available)
    """

    # --- 🧩 Validate presence of the target column ---
    if TARGET_COL not in df_train.columns:
        possible_targets = [c for c in df_train.columns if "target" in c.lower() or "code" in c.lower()]
        msg = f"'{TARGET_COL}' not found in training data. Possible alternatives: {possible_targets}"
        logging.error(msg)
        raise KeyError(msg)

    # --- 🧹 Clean training data ---
    df_train = df_train.dropna(subset=[TARGET_COL]).copy()
    df_train[TEXT_COLS] = df_train[TEXT_COLS].fillna("")
    df_train["combined_text"] = df_train[TEXT_COLS].apply(lambda x: " ".join(x.astype(str)), axis=1)

    # --- 🧹 Clean test data ---
    df_test = df_test.copy()
    df_test[TEXT_COLS] = df_test[TEXT_COLS].fillna("")
    df_test["combined_text"] = df_test[TEXT_COLS].apply(lambda x: " ".join(x.astype(str)), axis=1)

    # --- 🔍 Extract features and labels ---
    X_train = df_train[["combined_text"] + CODE_COLS]
    y_train = df_train[TARGET_COL].astype(str)

    # Handle missing target column in test
    if TARGET_COL in df_test.columns:
        y_test = df_test[TARGET_COL].astype(str)
    else:
        y_test = pd.Series([None] * len(df_test), name="target")
        logging.warning("Test data has no target column — proceeding with prediction-only mode.")

    X_test = df_test[["combined_text"] + CODE_COLS]

    # --- ✅ Log summary ---
    logging.info(f"Prepared data successfully.")
    logging.info(f"Train samples: {len(X_train)}, unique classes: {y_train.nunique()}")
    logging.info(f"Test samples: {len(X_test)}, target available: {y_test.notna().sum()}")

    return X_train, y_train, X_test, y_test






















