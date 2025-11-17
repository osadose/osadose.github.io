Traceback (most recent call last):
  File "c:\Users\osadoo\Documents\GitHub\ML-coder - Copy\main.py", line 43, in <module>
    main()
  File "c:\Users\osadoo\Documents\GitHub\ML-coder - Copy\main.py", line 17, in main
    X_train, y_train, X_test, y_test = prepare_data(df_train, df_test)
                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "c:\Users\osadoo\Documents\GitHub\ML-coder - Copy\coder\preprocessing.py", line 15, in prepare_data
    df_train[TEXT_COLS] = df_train[TEXT_COLS].fillna("")
                          ~~~~~~~~^^^^^^^^^^^
  File "C:\Users\osadoo\Documents\GitHub\ML-coder - Copy\.venv\Lib\site-packages\pandas\core\frame.py", line 4119, in __getitem__
    indexer = self.columns._get_indexer_strict(key, "columns")[1]
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\osadoo\Documents\GitHub\ML-coder - Copy\.venv\Lib\site-packages\pandas\core\indexes\base.py", line 6212, in _get_indexer_strict
    self._raise_if_missing(keyarr, indexer, axis_name)
  File "C:\Users\osadoo\Documents\GitHub\ML-coder - Copy\.venv\Lib\site-packages\pandas\core\indexes\base.py", line 6261, in _raise_if_missing
    raise KeyError(f"None of [{key}] are in the [{axis_name}]")
KeyError: "None of [Index(['why', 'what'], dtype='object')] are in the [columns]"

























