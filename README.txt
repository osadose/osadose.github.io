
PS C:\Users\osadoo\Documents\GitHub\ISCO-ISIC-OCCUPATIONCODER> & c:/Users/osadoo/Documents/GitHub/ISCO-ISIC-OCCUPATIONCODER/.venv/Scripts/Activate.ps1
(.venv) PS C:\Users\osadoo\Documents\GitHub\ISCO-ISIC-OCCUPATIONCODER> python -m coder.cli --input inputs\test.csv --output outputs\cleaned_test.csv --isco "coder\data\ISCO-08 EN Structure and definitions.xlsx" --isic "coder\data\ISIC Rev. 5 Explanatory Notes.xlsx" --config coder\config.yaml
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Users\osadoo\Documents\GitHub\ISCO-ISIC-OCCUPATIONCODER\coder\cli.py", line 2, in <module>
    from .pipeline import run_pipeline
  File "C:\Users\osadoo\Documents\GitHub\ISCO-ISIC-OCCUPATIONCODER\coder\pipeline.py", line 8, in <module>
    from coder.loader import load_isco_catalog, load_isic_catalog
ImportError: cannot import name 'load_isco_catalog' from 'coder.loader' (C:\Users\osadoo\Documents\GitHub\ISCO-ISIC-OCCUPATIONCODER\coder\loader.py)
(.venv) PS C:\Users\osadoo\Documents\GitHub\ISCO-ISIC-OCCUPATIONCODER> 
