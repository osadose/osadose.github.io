Documents/GitHub/ML-coder/main.py
2025-10-08 10:21:48,017 [INFO] Logging initialized. Log file: output\run.log
2025-10-08 10:21:48,017 [INFO] Starting ML pipeline...
2025-10-08 10:21:48,018 [WARNING] 'data\NLFS_2024Q1_INDIVIDUAL 2.xlsx' not found. Falling back to 'data\NLFS_2024Q1_INDIVIDUAL.xlsx'.
2025-10-08 10:21:48,018 [INFO] Loading TRAIN from: data\NLFS_2024Q1_INDIVIDUAL.xlsx
2025-10-08 10:22:11,459 [INFO] Loading TEST from: data\NLFS_2024_Q2.csv
2025-10-08 10:22:12,936 [INFO] MIN_SAMPLES_PER_CLASS_TUNING = 7 (derived from TRAIN requirement with TEST_SIZE)
2025-10-08 10:22:12,936 [INFO] Filtering 174 classes with <7 samples for tuning.
2025-10-08 10:22:12,937 [INFO] Remaining samples for tuning: 21219
2025-10-08 10:22:12,939 [INFO] Classes kept for tuning: 155 / 329
2025-10-08 10:22:12,999 [INFO] [Info] Running GridSearchCV for Logistic Regression on D1-TRAIN (stratified CV)...
Fitting 5 folds for each of 5 candidates, totalling 25 fits
2025-10-08 10:24:32,703 [INFO] Best f1_weighted: 0.7665
2025-10-08 10:24:32,703 [INFO] Best Params: {'classifier__C': 10.0, 'classifier__penalty': 'l2', 'classifier__solver': 'lbfgs'}
Traceback (most recent call last):
  File "c:\Users\osadoo\Documents\GitHub\ML-coder\main.py", line 232, in <module>
    main()
  File "c:\Users\osadoo\Documents\GitHub\ML-coder\main.py", line 116, in main
    pd.DataFrame(rep).transpose().reset_index().rename(columns={"index":"label"}).to_csv(out_path, index=False)
    ^^^^^^^^^^^^^^^^^
  File "C:\Users\osadoo\Documents\GitHub\ML-coder\.venv\Lib\site-packages\pandas\core\frame.py", line 890, in __init__
    raise ValueError("DataFrame constructor not properly called!")
ValueError: DataFrame constructor not properly called!



























