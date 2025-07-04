Hi there, I'm Ose



---
title: "NLP for Occupation Coding (Using NLFS Data)"
format:
  html:
    self-contained: true
jupyter: python3
---

## 🎯 Learning Objectives

- Load and examine free-text occupation responses from Nigeria Labour Force Survey (NLFS)
- Clean and normalize text
- Tokenize and lemmatize using spaCy
- Map cleaned tokens to ISCO codes using keyword and fuzzy matching
- Understand how NLP helps clean survey data for labour market insights

---

## 📊 Load Survey Data (Mock NLFS Sample)

Assuming you have cleaned NLFS microdata that includes free-text occupation:

```python
import pandas as pd

# Simulated sample based on real structure from NLFS Q2 2024 microdata
df = pd.DataFrame({
    "person_id": [1101, 1102, 1103, 1104, 1105],
    "occupation_response": [
        "secondary teacher public",
        "registered nurse",
        "sells fruit on the roadside",
        "small scale gold miner",
        "civil engr construction site"
    ]
})

df.head()
```

---

## 🧼 Clean and Normalize the Text

Remove punctuation and convert to lowercase:

```python
import string

def clean_text(text):
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation))

df["job_clean"] = df["occupation_response"].apply(clean_text)
df[["occupation_response", "job_clean"]]
```

---

## 🔠 Tokenization + Lemmatization (spaCy)

Break into words and reduce to their base forms:

```python
import spacy
nlp = spacy.load("en_core_web_sm")

def tokenize_lemmatize(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

df["tokens"] = df["job_clean"].apply(tokenize_lemmatize)
df[["job_clean", "tokens"]]
```

---

## 🗂 ISCO Code Matching (Keyword Method)

Define known keywords and their ISCO-08 major group codes:

```python
isco_keywords = {
    "teacher": "2341",        # Secondary school teachers
    "nurse": "2221",          # Nursing professionals
    "vendor": "5211",         # Street and market vendors
    "miner": "8111",          # Miners and quarriers
    "engineer": "2142"        # Civil engineers
}

def keyword_match(tokens):
    for token in tokens:
        if token in isco_keywords:
            return isco_keywords[token]
    return "0000"  # No match

df["isco_keyword"] = df["tokens"].apply(keyword_match)
df[["job_clean", "tokens", "isco_keyword"]]
```

---

## 🔍 Fuzzy Matching for Spelling Variants

Support imperfect matches using RapidFuzz:

```python
from rapidfuzz import process

def fuzzy_match(text, mapping, threshold=80):
    match, score = process.extractOne(text, mapping.keys())
    if score >= threshold:
        return mapping[match]
    return "0000"

df["isco_fuzzy"] = df["job_clean"].apply(lambda x: fuzzy_match(x, isco_keywords))
df[["job_clean", "isco_keyword", "isco_fuzzy"]]
```

---

## 🧮 Final ISCO Assignment

Use keyword match unless unavailable, then fall back to fuzzy match:

```python
df["isco_final"] = df["isco_keyword"].where(df["isco_keyword"] != "0000", df["isco_fuzzy"])
df[["person_id", "occupation_response", "isco_final"]]
```

---

## ✅ Summary

You have:

- Used a realistic NLFS-like dataset
- Cleaned and lemmatized text responses
- Mapped jobs to ISCO-08 codes using keyword and fuzzy techniques
- Built a reproducible pipeline for occupation coding

---

## 🔄 What Next?

- Expand the keyword list with NBS-aligned ISCO mappings  
- Use more sophisticated models for improved classification  
- Aggregate and analyze labour data across occupation groups  

