import os
import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

logger = logging.getLogger(__name__)


# =====================================================
# 🧩 Helper: Save Confusion Matrix
# =====================================================
def save_confusion_matrix(y_true, y_pred, labels, output_dir, timestamp):
    """
    Save confusion matrix as PNG image.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation="vertical", colorbar=False)
    plt.title("Confusion Matrix")

    cm_path = os.path.join(output_dir, f"D2_confusion_matrix_{timestamp}.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved confusion matrix to {cm_path}")
    return cm_path


# =====================================================
# 🧩 Helper: Save Predictions
# =====================================================
def save_predictions(y_true, y_pred, config, timestamp):
    """
    Save raw D2 predictions (true vs predicted labels) to CSV.
    """
    output_dir = config["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    df_pred = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred
    })

    pred_path = os.path.join(output_dir, f"predictions_D2_{timestamp}.csv")
    df_pred.to_csv(pred_path, index=False)
    logger.info(f"Saved raw D2 predictions to {pred_path}")

    return pred_path


# =====================================================
# 🧩 Evaluate D1 (holdout)
# =====================================================
def evaluate_on_D1(model, X_holdout, y_holdout, config, timestamp):
    """
    Evaluate trained model on D1 holdout set and save report.
    """
    logger.info("Evaluating model on D1 (holdout)...")

    y_pred = model.predict(X_holdout)
    report = classification_report(y_holdout, y_pred, output_dict=True)

    out_dir = config["paths"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    report_path = os.path.join(out_dir, f"D1_classification_report_{timestamp}.csv")
    pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "label"}).to_csv(report_path, index=False)

    logger.info(f"Saved D1 classification report to {report_path}")
    return report


# =====================================================
# 🧩 Evaluate D2 (test/out-of-domain)
# =====================================================
def evaluate_on_D2(model, X_test, y_test, config, timestamp):
    """
    Evaluate best model on D2 test dataset, saving report, metrics, confusion matrix, and predictions.
    """
    logger.info("Evaluating best model on D2...")

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    output_dir = config["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Save CSV report
    report_path = os.path.join(output_dir, f"D2_classification_report_{timestamp}.csv")
    pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "label"}).to_csv(report_path, index=False)
    logger.info(f"Saved D2 classification report to {report_path}")

    # Save metrics summary (JSON)
    metrics = {
        "accuracy": report["accuracy"],
        "macro_avg_f1": report["macro avg"]["f1-score"],
        "weighted_avg_f1": report["weighted avg"]["f1-score"]
    }

    metrics_path = os.path.join(output_dir, f"D2_metrics_{timestamp}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved D2 metrics summary to {metrics_path}")

    # Save confusion matrix
    labels = sorted(list(set(y_test)))
    save_confusion_matrix(y_test, y_pred, labels, output_dir, timestamp)

    # Save predictions
    save_predictions(y_test, y_pred, config, timestamp)

    logger.info(f"Saved all D2 evaluation artifacts to {output_dir}")
    return report


# =====================================================
# 🧩 Unified Evaluate & Save (for main.py)
# =====================================================
def evaluate_and_save(model, data_splits, config, timestamp):
    """
    Run full evaluation on D1 (holdout) and D2 (test) datasets.
    """
    try:
        X_holdout, y_holdout = data_splits["X_holdout"], data_splits["y_holdout"]
        X_test, y_test = data_splits["X_test"], data_splits["y_test"]

        logger.info("=== Starting model evaluation phase ===")

        # D1 Evaluation
        evaluate_on_D1(model, X_holdout, y_holdout, config, timestamp)

        # D2 Evaluation
        evaluate_on_D2(model, X_test, y_test, config, timestamp)

        logger.info("=== Evaluation completed successfully ===")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise

