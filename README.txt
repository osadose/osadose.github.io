from pathlib import Path

# === PATHS ===
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

TRAIN_FILE = DATA_DIR / "NLFS_2024Q1_INDIVIDUAL.xlsx"
TEST_FILE = DATA_DIR / "NLFS_2024_Q2.csv"

# === MODEL & CV SETTINGS ===
RANDOM_STATE = 42
CV_FOLDS = 5
MIN_SAMPLES_PER_CLASS_TUNING = 5

# === FEATURES ===
TEXT_COLS = ["occupation_title", "description"]
CODE_COLS = ["industry_code"]
TARGET_COL = "target"



import logging
from pathlib import Path

def setup_logger(output_dir: Path):
    """Configure logging to file and console."""
    output_dir.mkdir(exist_ok=True)
    log_file = output_dir / "run.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_file}")



import pandas as pd
import logging
from pathlib import Path
from coder.config import TEXT_COLS, TARGET_COL, CODE_COLS

def load_data(train_path: Path, test_path: Path):
    """Load training and test datasets."""
    logging.info(f"Loading TRAIN from: {train_path}")
    df_train = pd.read_excel(train_path, engine="openpyxl") if train_path.suffix in [".xlsx", ".xls"] else pd.read_csv(train_path)

    logging.info(f"Loading TEST from: {test_path}")
    df_test = pd.read_csv(test_path) if test_path.suffix == ".csv" else pd.read_excel(test_path, engine="openpyxl")

    return df_train, df_test

def prepare_data(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """Clean, combine text fields, and ensure consistency."""
    df_train = df_train.dropna(subset=[TARGET_COL])
    df_train[TEXT_COLS] = df_train[TEXT_COLS].fillna("")
    df_train["combined_text"] = df_train[TEXT_COLS].agg(" ".join, axis=1)

    df_test[TEXT_COLS] = df_test[TEXT_COLS].fillna("")
    df_test["combined_text"] = df_test[TEXT_COLS].agg(" ".join, axis=1)

    X_train = df_train[["combined_text"] + CODE_COLS]
    y_train = df_train[TARGET_COL].astype(str)

    X_test = df_test[["combined_text"] + CODE_COLS]
    y_test = df_test.get(TARGET_COL, None)

    logging.info(f"Prepared data: TRAIN={len(X_train)} rows, TEST={len(X_test)} rows")
    return X_train, y_train, X_test, y_test



import logging
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from coder.config import RANDOM_STATE, CV_FOLDS

def build_model():
    """Build ML pipeline for text + categorical data."""
    preprocessor = ColumnTransformer([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2)), "combined_text"),
        ("onehot", OneHotEncoder(handle_unknown="ignore"), ["industry_code"])
    ])

    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
    return pipeline

def tune_model(X, y):
    """Run GridSearchCV for hyperparameter optimization."""
    pipeline = build_model()
    param_grid = {
        "classifier__C": [0.1, 1, 10],
        "classifier__penalty": ["l2"],
        "classifier__solver": ["lbfgs"]
    }

    gs = GridSearchCV(
        pipeline,
        param_grid,
        scoring="f1_weighted",
        cv=CV_FOLDS,
        n_jobs=-1,
        verbose=1
    )
    gs.fit(X, y)
    logging.info(f"Best f1_weighted: {gs.best_score_:.4f} | Params: {gs.best_params_}")
    return gs.best_estimator_

def save_model(model, path):
    """Save model to output directory."""
    joblib.dump(model, path)
    logging.info(f"Model saved to {path}")



import pandas as pd
import numpy as np
import logging
from sklearn.metrics import classification_report

def evaluate_and_save(model, X_test, y_test, output_dir):
    """Generate predictions, save classification report and predictions."""
    preds = model.predict(X_test)

    report = classification_report(
        y_test, preds, digits=4, zero_division=0, output_dict=True
    )
    report_path = output_dir / "classification_report.csv"
    pd.DataFrame(report).transpose().to_csv(report_path, index=True)

    try:
        probas = model.predict_proba(X_test)
        confidence = np.max(probas, axis=1)
    except Exception:
        confidence = np.nan

    preds_df = X_test.copy()
    preds_df["predicted"] = preds
    preds_df["confidence"] = confidence
    preds_path = output_dir / "predictions_D2_from_LG.csv"
    preds_df.to_csv(preds_path, index=False)

    logging.info(f"Saved classification report to {report_path}")
    logging.info(f"Saved predictions to {preds_path}")



import logging
from coder.logger import setup_logger
from coder.config import TRAIN_FILE, TEST_FILE, OUTPUT_DIR
from coder.data_utils import load_data, prepare_data
from coder.model_utils import tune_model, save_model
from coder.evaluation import evaluate_and_save

def main():
    setup_logger(OUTPUT_DIR)
    logging.info("🚀 Starting Reproducible ML Pipeline")

    # 1️⃣ Load and prepare data
    df_train, df_test = load_data(TRAIN_FILE, TEST_FILE)
    X_train, y_train, X_test, y_test = prepare_data(df_train, df_test)

    # 2️⃣ Train + tune model
    model = tune_model(X_train, y_train)
    save_model(model, OUTPUT_DIR / "best_model.joblib")

    # 3️⃣ Evaluate on test data
    if y_test is not None:
        evaluate_and_save(model, X_test, y_test, OUTPUT_DIR)
    else:
        logging.warning("No target column found in test data — predictions only.")

    logging.info("✅ Pipeline complete. All outputs in /output folder.")

if __name__ == "__main__":
    main()


