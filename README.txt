Hi there, I'm Ose


---
title: "Importing and Cleaning Data"
format:
  html:
    self-contained: true
jupyter: python3
---

## Learning Objectives

- Read CSV files using pandas
- Inspect and understand data structure
- Clean column names and data types
- Handle missing data
- Save cleaned datasets for reuse

---

## Loading the Dataset

We will use the same Gapminder dataset you've worked with in previous weeks.

```{python}
import pandas as pd

# Load data
df = pd.read_csv("../../data/gapminder.csv")

# Preview
df.head()
```

---

## Inspecting the Data

Use pandas functions to understand the structure and quality of the dataset.

```{python}
df.info()
df.describe()
df.columns
```

---

## Filtering for Nigeria

Focus on a subset of the data for practical cleaning:

```{python}
nigeria = df[df["country"] == "Nigeria"].copy()
nigeria.head()
```

---

## Renaming Columns

Clean and standardize column names for clarity and consistency:

```{python}
nigeria.columns = nigeria.columns.str.lower().str.replace(" ", "_")
nigeria = nigeria.rename(columns={"gdpPercap": "gdp_per_capita"})
nigeria.head()
```

---

## Checking for Missing Data

It's important to check for missing or null values:

```{python}
nigeria.isnull().sum()
```

(For this dataset, there are typically no missing values — but this is a best practice step.)

---

## Data Type Checks and Conversion

Make sure each column is the correct type:

```{python}
nigeria.dtypes
```

If needed, convert `year` to integer or datetime:

```{python}
nigeria["year"] = nigeria["year"].astype(int)
```

---

## Creating New Columns

Add a simple new column: total GDP (approximate) = population × gdp_per_capita

```{python}
nigeria["total_gdp"] = nigeria["pop"] * nigeria["gdp_per_capita"]
nigeria[["year", "gdp_per_capita", "pop", "total_gdp"]].head()
```

---

## Saving the Cleaned Dataset

Once cleaned, export the dataset for future use:

```{python}
nigeria.to_csv("../../data/gapminder_nigeria_cleaned.csv", index=False)
```

Check that the file was saved correctly.

---

## Summary

In this live coding session, we:

- Loaded and inspected the Gapminder dataset
- Focused on Nigeria for simplicity
- Cleaned column names and checked types
- Added a derived column (`total_gdp`)
- Saved the cleaned version for future analysis


```
