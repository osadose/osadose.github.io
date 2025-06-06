Hi there, I'm Ose

[
```markdown
---
title: "Week 4: Importing and Cleaning Data"
format: revealjs
theme: simple
logo: template/logo.png
---

## 1. Welcome to Week 4 🎉

**Theme:** Importing and Cleaning Data

This week combines:
- 📥 Loading raw data from different sources
- 🧹 Cleaning and preparing it for analysis

This is the "real world" part of data science — messy, essential, and very hands-on!

---

## 2. Why This Matters 💡

> "80% of data science is cleaning data. The other 20% is complaining about cleaning data." — *Anonymous*

You need clean, structured data to:
- Avoid wrong conclusions
- Work with machine learning tools
- Build reproducible workflows

---

## 3. Sources of Data 📂

You'll commonly work with:
- `.csv`, `.xlsx` (Excel), `.json`
- Databases (SQL, SQLite)
- APIs
- Web scraping

📌 This week focuses on CSV and Excel — the most common file types in data projects.

---

## 4. Tools We’ll Use 🛠️

- `pandas`: for importing, cleaning, and analyzing tabular data  
- `openpyxl`, `xlrd`: for Excel file support  
- Python’s `os` and `pathlib`: for managing file paths

You'll use Jupyter Notebooks or VS Code to run your scripts.

---

## 5. The Import Process 📥

With `pandas`, importing is simple:

```python
import pandas as pd

df = pd.read_csv("data/sales.csv")
```

For Excel files:

```python
df = pd.read_excel("data/finance.xlsx")
```

🔍 After loading, inspect with:
```python
df.head()
df.info()
```

---

## 6. What is “Dirty” Data? 🧼

Common problems include:
- Missing values
- Inconsistent formats (e.g. dates, case)
- Duplicates
- Wrong data types
- Outliers

🛠️ Cleaning = fixing or removing these issues.

---

## 7. Common Cleaning Techniques 🧹

Examples:

```python
df.dropna()                 # Remove missing rows
df.fillna(0)                # Fill missing values
df['Date'] = pd.to_datetime(df['Date'])  # Fix dates
df.drop_duplicates()        # Remove duplicates
```

We'll go deeper into these in the notebook.

---

## 8. Real-Life Example 🏥

You receive a spreadsheet with patient data:
- Names in **UPPERCASE**
- Dates in **text format**
- Some rows are **empty**
- Income listed as **text like "$45,000"**

🎯 Your job: Make it usable for analysis.

---

## 9. This Week's Goals ✅

By the end of the week, you should be able to:
- Import data from CSV and Excel
- Inspect and understand its structure
- Clean missing, invalid, or inconsistent values
- Save cleaned datasets for future use

---

## 10. Let’s Get Started! 🚀

📌 During the live session:
- We'll explore real messy datasets
- Practice cleaning step by step
- Build confidence with pandas tools

👩‍💻 Start your notebook:  
```bash
notebooks/week4-cleaning.ipynb
```

Ready to clean some data? Let’s go!

```
]
