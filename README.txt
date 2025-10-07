
random_state: 42
test_size: 0.2
cv_folds: 5
scoring: f1_weighted

paths:
  data_dir: "data"
  output_dir: "output"
  train_file: "NLFS_2024Q1_INDIVIDUAL.xlsx"
  test_file: "NLFS_2024_Q2.csv"

model:
  type: "logreg"
  c_values: [0.5, 1.0, 2.0, 5.0, 10.0]
  solver: ["lbfgs", "liblinear"]
  penalty: ["l2"]



import os
import yaml
from typing import Dict, Any

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load YAML configuration."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    os.makedirs(config["paths"]["output_dir"], exist_ok=True)
    return config



import logging
import time
import os
from typing import Callable, Any
from functools import wraps

def setup_logging(output_dir: str) -> str:
    """Configure file + console logging."""
    log_file = os.path.join(output_dir, "run.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_file}")
    return log_file

def time_it(func: Callable) -> Callable:
    """Decorator to measure execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        logging.info(f"{func.__name__} executed in {time.time() - start:.2f}s")
        return result
    return wrapper



import os
import pandas as pd
import logging
from typing import Tuple
from src.config_loader import load_config

def load_data(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test datasets from configured paths."""
    paths = config["paths"]
    train_path = os.path.join(paths["data_dir"], paths["train_file"])
    test_path = os.path.join(paths["data_dir"], paths["test_file"])
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file missing: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file missing: {test_path}")
    logging.info(f"Loading TRAIN from: {train_path}")
    logging.info(f"Loading TEST  from: {test_path}")
    train_df = pd.read_excel(train_path) if train_path.endswith(".xlsx") else pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df



import pandas as pd
import logging
from typing import Tuple

def prepare_data(df: pd.DataFrame, text_cols: list, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Combine text fields and prepare X, y."""
    df = df.dropna(subset=[target_col]).copy()
    df[text_cols] = df[text_cols].fillna("")
    df["combined_text"] = df[text_cols].apply(lambda x: " ".join(x.astype(str)), axis=1)
    X = df["combined_text"]
    y = df[target_col].astype(str)
    logging.info(f"Prepared data with {len(df)} samples and {y.nunique()} unique classes.")
    return X, y



import os
import logging
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple

def build_pipeline(random_state: int) -> Pipeline:
    """Construct a text + logistic regression pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 3))),
        ("clf", LogisticRegression(max_iter=800, class_weight="balanced", random_state=random_state))
    ])

def train_model(X, y, config) -> Tuple[Pipeline, dict]:
    """Run GridSearchCV and save the best model."""
    model_cfg = config["model"]
    cv_folds = config["cv_folds"]
    scoring = config["scoring"]
    random_state = config["random_state"]
    output_dir = config["paths"]["output_dir"]

    pipe = build_pipeline(random_state)
    param_grid = {
        "clf__solver": model_cfg["solver"],
        "clf__C": model_cfg["c_values"],
        "clf__penalty": model_cfg["penalty"]
    }

    grid = GridSearchCV(pipe, param_grid, scoring=scoring, cv=cv_folds, n_jobs=-1, verbose=1)
    grid.fit(X, y)

    best_params = grid.best_params_
    best_score = grid.best_score_
    logging.info(f"Best model F1_weighted={best_score:.4f} with params {best_params}")

    # Save model
    timestamp = time_tag()
    model_path = os.path.join(output_dir, f"best_model_{timestamp}.joblib")
    joblib.dump(grid.best_estimator_, model_path)
    logging.info(f"Saved best model to {model_path}")
    return grid.best_estimator_, {"best_score": best_score, "best_params": best_params}

def time_tag() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")



import os
import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from typing import Any

def evaluate_and_save(model: Any, X_test, y_test, output_dir: str, prefix: str = "eval") -> dict:
    """Compute metrics, confusion matrix, and save results."""
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, digits=4, zero_division=0)

    # Save CSV + JSON
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"{prefix}_classification_report_{timestamp}.csv")
    json_path = os.path.join(output_dir, f"{prefix}_metrics_{timestamp}.json")
    pd.DataFrame(report).transpose().to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(report, f, indent=4)

    # Save confusion matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
    cm_path = os.path.join(output_dir, f"{prefix}_confusion_matrix_{timestamp}.png")
    plt.savefig(cm_path)
    plt.close(fig)

    logging.info(f"Saved report to {csv_path}, JSON metrics to {json_path}, and confusion matrix to {cm_path}")
    return report



import sys
import logging
import argparse
from src.config_loader import load_config
from src.utils import setup_logging, time_it
from src.data_loader import load_data
from src.preprocessing import prepare_data
from src.model import train_model
from src.evaluation import evaluate_and_save

@time_it
def main(args):
    try:
        config = load_config(args.config)
        setup_logging(config["paths"]["output_dir"])
        logging.info("Starting ML pipeline...")

        # Load data
        train_df, test_df = load_data(config)

        # Prepare data
        text_cols = ["mjj2a", "mjj2b", "mjj3a", "mjj3b"]
        target_col = "mjj2cclean"
        X_train, y_train = prepare_data(train_df, text_cols, target_col)
        X_test, y_test = prepare_data(test_df, text_cols, target_col)

        # Train model
        model, meta = train_model(X_train, y_train, config)

        # Evaluate on test
        evaluate_and_save(model, X_test, y_test, config["paths"]["output_dir"], prefix="D2")

        logging.info("Pipeline completed successfully ✅")

    except Exception as e:
        logging.exception(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Logistic Regression on D2")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    args = parser.parse_args()
    main(args)



__pycache__/
output/
*.log
*.csv
*.xlsx
.venv/



