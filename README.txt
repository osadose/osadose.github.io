Documents/GitHub/ML-coder/main.py
2025-10-07 16:01:54,748 [INFO] Logging initialized. Log file: output\run.log
2025-10-07 16:01:54,749 [INFO] Starting ML pipeline...
2025-10-07 16:01:54,750 [INFO] Loading TRAIN from: data\NLFS_2024Q1_INDIVIDUAL.xlsx
2025-10-07 16:01:54,751 [INFO] Loading TEST  from: data\NLFS_2024_Q2.csv
2025-10-07 16:02:33,542 [INFO] Prepared data with 21665 samples and 329 unique classes.
2025-10-07 16:02:34,105 [INFO] Prepared data with 12455 samples and 280 unique classes.
Fitting 3 folds for each of 5 candidates, totalling 15 fits
C:\Users\osadoo\Documents\GitHub\ML-coder\.venv\Lib\site-packages\sklearn\model_selection\_split.py:811: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=3.
  warnings.warn(
2025-10-07 16:04:31,544 [INFO] Best model F1_weighted=0.4872 with params {'clf__C': 10.0, 'clf__penalty': 'l2', 'clf__solver': 'lbfgs'}
2025-10-07 16:04:31,756 [INFO] Saved best model to output\best_model_20251007_160431.joblib
2025-10-07 16:04:31,757 [ERROR] Pipeline failed: evaluate_and_save() got an unexpected keyword argument 'prefix'
Traceback (most recent call last):
  File "c:\Users\osadoo\Documents\GitHub\ML-coder\main.py", line 31, in main
    evaluate_and_save(model, X_test, y_test, config["paths"]["output_dir"], prefix="D2")
TypeError: evaluate_and_save() got an unexpected keyword argument 'prefix'




# assuming you have these already prepared:
# X_holdout, y_holdout, X_test, y_test, config, timestamp
data_splits = {
    "X_holdout": X_holdout,
    "y_holdout": y_holdout,
    "X_test": X_test,
    "y_test": y_test
}

evaluate_and_save(best_model, data_splits, config, timestamp)

