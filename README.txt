grep -r "class.*ectoris" .venv/lib/python3.11/site-packages/classifai/


python -c "import classifai.vectorisers; print(dir(classifai.vectorisers))"


# Change from:
query_isco = query_isco[~query_isco['isco_code_clean'].str.contains('l')]

# To:
query_isco['isco_code_clean'] = query_isco['isco_code_clean'].astype(str)
query_isco = query_isco[~query_isco['isco_code_clean'].str.contains('l')]
