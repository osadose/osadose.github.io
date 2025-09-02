Hi! This is Ose!


PS C:\Users\osadoo\Documents\GitHub\ISCO-ISIC-OCCUPATIONCODER> & c:/Users/osadoo/Documents/GitHub/ISCO-ISIC-OCCUPATIONCODER/.venv/Scripts/Activate.ps1
(.venv) PS C:\Users\osadoo\Documents\GitHub\ISCO-ISIC-OCCUPATIONCODER> 
(.venv) PS C:\Users\osadoo\Documents\GitHub\ISCO-ISIC-OCCUPATIONCODER> 
(.venv) PS C:\Users\osadoo\Documents\GitHub\ISCO-ISIC-OCCUPATIONCODER> python -m coder.cli \
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Users\osadoo\Documents\GitHub\ISCO-ISIC-OCCUPATIONCODER\coder\cli.py", line 2, in <module>
    from .pipeline import run_pipeline
ModuleNotFoundError: No module named 'coder.pipeline'
(.venv) PS C:\Users\osadoo\Documents\GitHub\ISCO-ISIC-OCCUPATIONCODER>   --input inputs\test.csv \
At line:1 char:5
+   --input inputs\test.csv \
+     ~
Missing expression after unary operator '--'.
At line:1 char:5
+   --input inputs\test.csv \
+     ~~~~~
Unexpected token 'input' in expression or statement.
    + CategoryInfo          : ParserError: (:) [], ParentContainsErrorRecordException
    + FullyQualifiedErrorId : MissingExpressionAfterOperator
 
(.venv) PS C:\Users\osadoo\Documents\GitHub\ISCO-ISIC-OCCUPATIONCODER>   --output outputs\cleaned_test.csv \
At line:1 char:5
+   --output outputs\cleaned_test.csv \
+     ~
Missing expression after unary operator '--'.
At line:1 char:5
+   --output outputs\cleaned_test.csv \
+     ~~~~~~
Unexpected token 'output' in expression or statement.
    + CategoryInfo          : ParserError: (:) [], ParentContainsErrorRecordException
    + FullyQualifiedErrorId : MissingExpressionAfterOperator
 
(.venv) PS C:\Users\osadoo\Documents\GitHub\ISCO-ISIC-OCCUPATIONCODER>   --isco "coder\data\ISCO-08 EN Structure and definitions.xlsx" \
At line:1 char:5
+   --isco "coder\data\ISCO-08 EN Structure and definitions.xlsx" \
+     ~
Missing expression after unary operator '--'.
At line:1 char:5
+   --isco "coder\data\ISCO-08 EN Structure and definitions.xlsx" \
+     ~~~~
Unexpected token 'isco' in expression or statement.
    + CategoryInfo          : ParserError: (:) [], ParentContainsErrorRecordException
    + FullyQualifiedErrorId : MissingExpressionAfterOperator

(.venv) PS C:\Users\osadoo\Documents\GitHub\ISCO-ISIC-OCCUPATIONCODER>   --isic "coder\data\ISIC5_Exp_Notes_11Mar2024.xlsx" \
At line:1 char:5
+   --isic "coder\data\ISIC5_Exp_Notes_11Mar2024.xlsx" \
+     ~
Missing expression after unary operator '--'.
At line:1 char:5
+   --isic "coder\data\ISIC5_Exp_Notes_11Mar2024.xlsx" \
+     ~~~~
Unexpected token 'isic' in expression or statement.
    + CategoryInfo          : ParserError: (:) [], ParentContainsErrorRecordException
    + FullyQualifiedErrorId : MissingExpressionAfterOperator

(.venv) PS C:\Users\osadoo\Documents\GitHub\ISCO-ISIC-OCCUPATIONCODER>   --config coder/config.yaml
 *  History restored 

PS C:\Users\osadoo\Documents\GitHub\ISCO-ISIC-OCCUPATIONCODER> 








 





