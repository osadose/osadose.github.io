
GitHub/ML-coder - Copy/main.py"
2025-11-17 12:41:49,295 [INFO] Logging initialized. Log file: output\run.log
2025-11-17 12:41:49,295 [INFO] Starting multi-output ISCO+ISIC pipeline...
Traceback (most recent call last):
  File "c:\Users\osadoo\Documents\GitHub\ML-coder - Copy\main.py", line 136, in <module>
    main()
  File "c:\Users\osadoo\Documents\GitHub\ML-coder - Copy\main.py", line 37, in main
    df_train, df_test = load_data()
                        ^^^^^^^^^^^
  File "c:\Users\osadoo\Documents\GitHub\ML-coder - Copy\coder\data_utils.py", line 21, in load_data
    train_path = resolve_train_file()
                 ^^^^^^^^^^^^^^^^^^^^
  File "c:\Users\osadoo\Documents\GitHub\ML-coder - Copy\coder\data_utils.py", line 13, in resolve_train_file
    raise FileNotFoundError(f"Could not find {TRAIN_FILE} or fallback in current working directory.")
FileNotFoundError: Could not find NLFS_2024Q1_INDIVIDUAL 2.xlsx or fallback in current working directory.






























