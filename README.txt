GitHub\ML-coder\.venv\Lib\site-packages\sklearn\model_selection\_split.py:811: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=3.
  warnings.warn(
2025-10-07 14:08:50,227 [INFO] Best model F1_weighted=0.4872 with params {'clf__C': 10.0, 'clf__penalty': 'l2', 'clf__solver': 'lbfgs'}
2025-10-07 14:08:50,455 [INFO] Saved best model to output\best_model_20251007_140850.joblib
2025-10-07 14:09:44,303 [INFO] Saved report to output\D2_classification_report_20251007_140851.csv, JSON metrics to output\D2_metrics_20251007_140851.json, and confusion matrix to output\D2_confusion_matrix_20251007_140851.png
2025-10-07 14:09:44,305 [INFO] Pipeline completed successfully
2025-10-



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
# 🧩 D1 Evaluation (training holdout performance)
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
# 🧩 Helper: Save Confusion Matrix Plot
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
# 🧩 Helper: Save Predictions (NEW)
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
# 🧩 D2 Evaluation (out-of-domain generalization)
# =====================================================
def evaluate_on_D2(model, X_test, y_test, config, timestamp):
    """
    Evaluate the best model on D2 test dataset.
    Saves classification report, metrics (JSON), confusion matrix, and predictions.
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

    # Save JSON metrics summary
    metrics = {
        "accuracy": report["accuracy"],
        "macro avg": report["macro avg"]["f1-score"],
        "weighted avg": report["weighted avg"]["f1-score"]
    }

    metrics_path = os.path.join(output_dir, f"D2_metrics_{timestamp}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved D2 metrics summary to {metrics_path}")

    # Save Confusion Matrix
    labels = sorted(list(set(y_test)))
    save_confusion_matrix(y_test, y_pred, labels, output_dir, timestamp)

    # ✅ NEW: Save raw predictions
    save_predictions(y_test, y_pred, config, timestamp)

    logger.info(
        f"Saved report to {report_path}, JSON metrics to {metrics_path}, "
        f"and confusion matrix + predictions to {output_dir}"
    )

    return report
