
(.venv) PS C:\Users\osadoo\Documents\GitHub\ISCO-ISIC-OCCUPATIONCODER> python -m coder.cli --input inputs\test.csv --output outputs\cleaned_test.csv --isco "coder\data\ISCO-08 EN Structure and definitions.xlsx" --isic "coder\data\ISIC Rev. 5 Explanatory Notes.xlsx" --config coder\config.yaml
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
  File "C:\Users\osadoo\Documents\GitHub\ISCO-ISIC-OCCUPATIONCODER\coder\pipeline.py", line 139, in run_pipeline
    isco_titles = [t for t in (e.title for e in isco_catalog.entries)]
                                                ^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\osadoo\Documents\GitHub\ISCO-ISIC-OCCUPATIONCODER\.venv\Lib\site-packages\pandas\core\generic.py", line 6318, in __getattr__
    return object.__getattribute__(self, name)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'DataFrame' object has no attribute 'entries'
