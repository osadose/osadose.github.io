grep -r "class.*ectoris" .venv/lib/python3.11/site-packages/classifai/


python -c "import classifai.vectorisers; print(dir(classifai.vectorisers))"


# Change from:
query_isco = query_isco[~query_isco['isco_code_clean'].str.contains('l')]

# To:
query_isco['isco_code_clean'] = query_isco['isco_code_clean'].astype(str)
query_isco = query_isco[~query_isco['isco_code_clean'].str.contains('l')]


    query_isco['isco_code_clean'] = query_isco['isco_code_clean'].astype(str)
    query_isco = query_isco[~query_isco['isco_code_clean'].str.contains('l')]
    query_isco['isco'] = query_isco['isco'].astype(str)
    query_isco['isco_code'] = query_isco['isco'].str.extract('(\d+)')



    query_isic['isic_code_clean'] = query_isic['isic_code_clean'].astype(str)
    query_isic['isic'] = query_isic['isic'].astype(str)
    query_isic['isic_code'] = query_isic['isic'].str.extract('(\d+)')
