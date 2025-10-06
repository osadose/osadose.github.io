[Warn] 'data\NLFS_2024Q1_INDIVIDUAL 2.xlsx' not found. Falling back to 'data\NLFS_2024Q1_INDIVIDUAL.xlsx'.
[Info] Loading TRAIN from: data\NLFS_2024Q1_INDIVIDUAL.xlsx
[Info] Loading TEST  from: data\NLFS_2024_Q2.csv
[Info] MIN_SAMPLES_PER_CLASS_TUNING = 7 (derived from TRAIN requirement 5 with TEST_SIZE=0.2)
[Info] Filtering 174 classes with <7 samples for tuning.
[Info] Remaining samples for tuning: 21219
[Info] Classes kept for tuning: 155 / 329
[Info] Running GridSearchCV for Logistic Regression on D1-TRAIN (stratified CV)...
Fitting 5 folds for each of 10 candidates, totalling 50 fits

=== Best Model (Logistic Regression on D1-TRAIN) ===
Best f1_weighted: 0.8356
Best Params: {'classifier__C': 10.0, 'classifier__penalty': 'l2', 'classifier__solver': 'liblinear'}

=== D1 (Filtered) Stratified Hold-out (in-domain) ===
Accuracy:             0.8351
Precision (micro):    0.8351
Precision (macro):    0.7480
Precision (weighted): 0.8410
Recall (micro):       0.8351
Recall (macro):       0.7380
Recall (weighted):    0.8351
F1-score (micro):     0.8351
F1-score (macro):     0.7229
F1-score (weighted):  0.8342
Balanced Accuracy:    0.7380
Traceback (most recent call last):
  File "c:\Users\Documents\GitHub\ML-coder\evaluate_LG_on_D2", line 571, in <module>
    main()
  File "c:\Users\Documents\GitHub\ML-coder\evaluate_LG_on_D2", line 441, in main
    pd.DataFrame(report_holdout).transpose().reset_index().rename(columns={"index":"label"}).to_csv(out_path, index=False) 
                 ^^^^^^^^^^^^^^
NameError: name 'report_holdout' is not defined

