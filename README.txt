
Documents/GitHub/ML-coder/main.py
2025-10-08 10:02:11,447 [INFO] Logging initialized. Log file: output\run.log
2025-10-08 10:02:11,447 [INFO] Starting ML pipeline...
2025-10-08 10:02:11,447 [INFO] Loading TRAIN from: data/NLFS_2024Q1_INDIVIDUAL.xlsx
2025-10-08 10:02:33,440 [INFO] Loading TEST from: data/NLFS_2024_Q2.csv
Traceback (most recent call last):
  File "c:\Users\osadoo\Documents\GitHub\ML-coder\main.py", line 32, in <module>
    main()
  File "c:\Users\osadoo\Documents\GitHub\ML-coder\main.py", line 18, in main
    X_train = df_train.drop(columns=["target"])
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\osadoo\Documents\GitHub\ML-coder\.venv\Lib\site-packages\pandas\core\frame.py", line 5603, in drop
    return super().drop(
           ^^^^^^^^^^^^^
  File "C:\Users\osadoo\Documents\GitHub\ML-coder\.venv\Lib\site-packages\pandas\core\generic.py", line 4810, in drop
    obj = obj._drop_axis(labels, axis, level=level, errors=errors)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\osadoo\Documents\GitHub\ML-coder\.venv\Lib\site-packages\pandas\core\generic.py", line 4852, in _drop_axis
    new_axis = axis.drop(labels, errors=errors)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\osadoo\Documents\GitHub\ML-coder\.venv\Lib\site-packages\pandas\core\indexes\base.py", line 7136, in drop
    raise KeyError(f"{labels[mask].tolist()} not found in axis")
KeyError: "['target'] not found in axis"






















