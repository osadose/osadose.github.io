
PS C:\Users\osadoo\Documents\GitHub\ISCO-ISIC-OCCUPATIONCODER> & c:/Users/osadoo/Documents/GitHub/ISCO-ISIC-OCCUPATIONCODER/.venv/Scripts/Activate.ps1
(.venv) PS C:\Users\osadoo\Documents\GitHub\ISCO-ISIC-OCCUPATIONCODER> python -m coder.cli --input inputs\test.csv --output outputs\cleaned_test.csv --isco "coder\data\ISCO-08 EN Structure and definitions.xlsx" --isic "coder\data\ISIC5_Exp_Notes_11Mar2024.xlsx" --config coder\data\config.yaml
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Users\osadoo\Documents\GitHub\ISCO-ISIC-OCCUPATIONCODER\coder\cli.py", line 27, in <module>
    main()
  File "C:\Users\osadoo\Documents\GitHub\ISCO-ISIC-OCCUPATIONCODER\.venv\Lib\site-packages\click\core.py", line 1442, in __call__
    return self.main(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\osadoo\Documents\GitHub\ISCO-ISIC-OCCUPATIONCODER\.venv\Lib\site-packages\click\core.py", line 1363, in main
    rv = self.invoke(ctx)
         ^^^^^^^^^^^^^^^^
  File "C:\Users\osadoo\Documents\GitHub\ISCO-ISIC-OCCUPATIONCODER\.venv\Lib\site-packages\click\core.py", line 1226, in invoke
    return ctx.invoke(self.callback, **ctx.params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\osadoo\Documents\GitHub\ISCO-ISIC-OCCUPATIONCODER\.venv\Lib\site-packages\click\core.py", line 794, in invoke
    return callback(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\osadoo\Documents\GitHub\ISCO-ISIC-OCCUPATIONCODER\coder\cli.py", line 14, in main
    df, review_path = run_pipeline(
                      ^^^^^^^^^^^^^
  File "C:\Users\osadoo\Documents\GitHub\ISCO-ISIC-OCCUPATIONCODER\coder\pipeline.py", line 27, in run_pipeline
    with open(config_path, "r") as f:
         ^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: 'coder\\data\\config.yaml'
(.venv) PS C:\Users\osadoo\Documents\GitHub\ISCO-ISIC-OCCUPATIONCODER> 
