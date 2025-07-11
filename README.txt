Hi there, I'm Ose

import pandas as pd
import string

# Original dirty data
dirty_df = pd.DataFrame({
    "person_id": [1101, 1102, 1103, 1104, 1105],
    "occupation_response": [
        "Secondary Teacher, PUBLIC school",
        "REGISTERED NURSE  ",
        "sells fruit on roadside (informal)",
        "small-scale gold miner",
        "CIVIL ENGR. constr.site"
    ]
})

# Improved cleaning function
def clean_text(text):
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove punctuation (keeping apostrophes)
    text = text.translate(str.maketrans('', '', string.punctuation.replace("'", "")))
    
    # Standardize abbreviations
    replacements = {
        'engr': 'engineer',
        'constr': 'construction',
        'site': ' site'  # Ensure space before 'site'
    }
    for wrong, right in replacements.items():
        text = text.replace(wrong, right)
    
    # Fix all spacing issues (including removing hyphens)
    text = text.replace('-', ' ')  # Replace hyphens with spaces
    text = ' '.join(text.split())  # This handles all whitespace normalization
    
    return text

# Apply cleaning
dirty_df["cleaned_occupation"] = dirty_df["occupation_response"].apply(clean_text)

# Show results
dirty_df[["person_id", "occupation_response", "cleaned_occupation"]]

python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

import pandas as pd

# Dirty version of the sample data
df = pd.DataFrame({
    "person_id": [1101, 1102, 1103, 1104, 1105],
    "occupation_response": [
        "Secondary Teacher, PUBLIC school",  # Mixed case, punctuation
        "REGISTERED NURSE  ",  # All caps, trailing spaces
        "sells fruit on roadside (informal)",  # Parentheses, missing 'the'
        "small-scale gold miner",  # Hyphenated
        "CIVIL ENGR. constr.site"  # Abbreviations, mixed case, punctuation
    ]
})

df.head()

import pandas as pd
import string

# 1. Create the dirty dataset
dirty_df = pd.DataFrame({
    "person_id": [1101, 1102, 1103, 1104, 1105],
    "occupation_response": [
        "Secondary Teacher, PUBLIC school",
        "REGISTERED NURSE  ",
        "sells fruit on roadside (informal)",
        "small-scale gold miner",
        "CIVIL ENGR. constr.site"
    ]
})

# 2. Cleaning function
def clean_text(text):
    # Convert to string in case of non-string values
    text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation (keeping apostrophes)
    text = text.translate(str.maketrans('', '', string.punctuation.replace("'", "")))
    
    # Standardize common abbreviations
    replacements = {
        'engr': 'engineer',
        'constr': 'construction',
        'tchr': 'teacher',
        'nrse': 'nurse'
    }
    for wrong, right in replacements.items():
        text = text.replace(wrong, right)
    
    # Fix spacing issues
    text = ' '.join(text.split())  # Removes extra whitespace
    
    return text

# 3. Apply cleaning
dirty_df["cleaned_occupation"] = dirty_df["occupation_response"].apply(clean_text)

# 4. Display before/after comparison
dirty_df[["person_id", "occupation_response", "cleaned_occupation"]]