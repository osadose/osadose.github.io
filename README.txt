import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_data(train_path: str, test_path: str):
    """
    Load training and testing datasets.
    """
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    logger.info(f"Loading TRAIN from: {train_path}")
    df_train = pd.read_excel(train_path) if train_path.endswith(".xlsx") else pd.read_csv(train_path)

    logger.info(f"Loading TEST from: {test_path}")
    df_test = pd.read_excel(test_path) if test_path.endswith(".xlsx") else pd.read_csv(test_path)

    return df_train, df_test




import logging
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
import joblib
import os

logger = logging.getLogger(__name__)

def train_model(X_train, y_train, config, timestamp):
    """
    Train a Logistic Regression model with 5-fold CV GridSearch.
    """
    logger.info("Starting model training with 5-fold cross-validation...")

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs"))
    ])

    param_grid = {
        'clf__C': [0.1, 1.0, 10.0],
        'clf__penalty': ['l2']
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, param_grid, scoring='f1_weighted', cv=cv, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    logger.info(f"Best model F1_weighted={grid.best_score_:.4f} with params {grid.best_params_}")

    # Save best model
    output_dir = config["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"best_model_{timestamp}.joblib")
    joblib.dump(grid.best_estimator_, model_path)
    logger.info(f"Saved best model to {model_path}")

    return grid.best_estimator_




import os
import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

logger = logging.getLogger(__name__)

def save_confusion_matrix(y_true, y_pred, labels, output_dir, timestamp):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation="vertical", colorbar=False)
    plt.title("Confusion Matrix")
    cm_path = os.path.join(output_dir, f"confusion_matrix_{timestamp}.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved confusion matrix to {cm_path}")

def evaluate_model(model, X_test, y_test, config, timestamp, prefix="D2"):
    """
    Evaluate a trained model and save all reports.
    """
    logger.info(f"Evaluating model on {prefix} dataset...")

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    output_dir = config["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Save CSV report
    report_path = os.path.join(output_dir, f"{prefix}_classification_report_{timestamp}.csv")
    pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "label"}).to_csv(report_path, index=False)

    # Save metrics JSON
    metrics_path = os.path.join(output_dir, f"{prefix}_metrics_{timestamp}.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"]
        }, f, indent=4)

    # Save predictions
    pred_path = os.path.join(output_dir, f"{prefix}_predictions_{timestamp}.csv")
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(pred_path, index=False)

    # Save confusion matrix
    save_confusion_matrix(y_test, y_pred, sorted(list(set(y_test))), output_dir, timestamp)

    logger.info(f"Saved all evaluation outputs for {prefix} to {output_dir}")




import os
import yaml
import logging
from datetime import datetime

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "run.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )
    logging.info(f"Logging initialized. Log file: {log_path}")

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")




paths:
  train_data: data/NLFS_2024Q1_INDIVIDUAL.xlsx
  test_data: data/NLFS_2024_Q2.csv
  output_dir: output



import logging
from coder.data_loader import load_data
from coder.model import train_model
from coder.evaluation import evaluate_model
from coder.utils import setup_logging, load_config, get_timestamp

def main():
    config = load_config()
    setup_logging(config["paths"]["output_dir"])
    timestamp = get_timestamp()

    logging.info("Starting ML pipeline...")

    # Load data
    df_train, df_test = load_data(config["paths"]["train_data"], config["paths"]["test_data"])

    # Basic example — replace 'X' and 'y' with your actual feature/target columns
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_test = df_test.drop(columns=["target"])
    y_test = df_test["target"]

    # Train model
    best_model = train_model(X_train, y_train, config, timestamp)

    # Evaluate model
    evaluate_model(best_model, X_test, y_test, config, timestamp, prefix="D2")

    logging.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()

