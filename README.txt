
Documents\GitHub\ML-coder - Copy\output\run.log
--- Logging error ---
Traceback (most recent call last):
  File "c:\ONSapps\My_Spyder\Lib\logging\__init__.py", line 1113, in emit
    stream.write(msg + self.terminator)
  File "c:\ONSapps\My_Spyder\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f680' in position 31: character maps to <undefined>      
Call stack:
  File "c:\Users\osadoo\Documents\GitHub\ML-coder - Copy\main.py", line 29, in <module>
    main()
  File "c:\Users\osadoo\Documents\GitHub\ML-coder - Copy\main.py", line 10, in main
    logging.info("🚀 Starting Reproducible ML Pipeline")
Message: '🚀 Starting Reproducible ML Pipeline'
Arguments: ()
2025-10-08 15:24:14,467 [INFO] 🚀 Starting Reproducible ML Pipeline
2025-10-08 15:24:14,584 [INFO] Loading TRAIN from: C:\Users\osadoo\Documents\GitHub\ML-coder - Copy\data\NLFS_2024Q1_INDIVIDUAL.xlsx
2025-10-08 15:24:53,484 [INFO] Loading TEST from: C:\Users\osadoo\Documents\GitHub\ML-coder - Copy\data\NLFS_2024_Q2.csv
Traceback (most recent call last):
  File "c:\Users\osadoo\Documents\GitHub\ML-coder - Copy\main.py", line 29, in <module>
    main()
  File "c:\Users\osadoo\Documents\GitHub\ML-coder - Copy\main.py", line 14, in main
    X_train, y_train, X_test, y_test = prepare_data(df_train, df_test)
                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "c:\Users\osadoo\Documents\GitHub\ML-coder - Copy\coder\data_utils.py", line 18, in prepare_data
    df_train = df_train.dropna(subset=[TARGET_COL])
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\osadoo\Documents\GitHub\ML-coder - Copy\.venv\Lib\site-packages\pandas\core\frame.py", line 6692, in dropna
    raise KeyError(np.array(subset)[check].tolist())
KeyError: ['target']





















