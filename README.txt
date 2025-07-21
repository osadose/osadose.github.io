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
- (Optional) Visualize trends in cleaned data  

---

## Loading the Dataset

We’ll use the same Gapminder dataset from previous weeks.

```{python}
import pandas as pd

# Load data
df = pd.read_csv("../../data/gapminder.csv")

# Preview the dataset
df.head()
```

---

## Quick Overview of the Dataset

```{python}
df.shape  # number of rows and columns
df.sample(5)  # random sample
```

---

## Inspecting the Data

Use pandas tools to understand the structure and quality of the data.

```{python}
df.info()
df.describe()
df.columns
```

---

## Why This Matters

> Clean, well-structured data reduces errors and makes analysis and communication more effective.  
> It’s often where most time is spent in real-world data work.

---

## Filtering for Nigeria

Focus on data for Nigeria to keep things simple.

```{python}
nigeria = df[df["country"] == "Nigeria"].copy()
nigeria.head()
```

---

## Renaming Columns

Standardize column names for clarity and consistency.

```{python}
nigeria.columns = nigeria.columns.str.lower().str.replace(" ", "_")
nigeria = nigeria.rename(columns={"gdpPercap": "gdp_per_capita"})
nigeria.head()
```

---

## Checking for Missing Data

Look for missing values — even if we expect none.

```{python}
nigeria.isnull().sum()
```

---

## Checking and Converting Data Types

Ensure each column uses the appropriate data type.

```{python}
nigeria.dtypes
```

Convert `year` to integer if needed:

```{python}
nigeria["year"] = nigeria["year"].astype(int)
```

---

## Creating a New Column

Calculate approximate total GDP: `population × gdp_per_capita`.

```{python}
nigeria["total_gdp"] = nigeria["pop"] * nigeria["gdp_per_capita"]
nigeria[["year", "gdp_per_capita", "pop", "total_gdp"]].head()
```

---

## Visualizing GDP per Capita Over Time

```{python}
import matplotlib.pyplot as plt

nigeria.plot(x="year", y="gdp_per_capita", kind="line", title="GDP per capita in Nigeria")
plt.ylabel("GDP per capita")
plt.xlabel("Year")
plt.grid(True)
plt.tight_layout()
```

---

## Saving the Cleaned Dataset

Export the cleaned data for future use.

```{python}
nigeria.to_csv("../../data/gapminder_nigeria_cleaned.csv", index=False)
```

---

## Confirm File Save

Check that the file exists:

```{python}
import os
os.path.exists("../../data/gapminder_nigeria_cleaned.csv")
```

---

## Optional: Advanced Preview (Stretch)

Group by decade and calculate average GDP per capita (preview of `groupby`):

```{python}
nigeria["decade"] = (nigeria["year"] // 10) * 10
nigeria.groupby("decade")["gdp_per_capita"].mean().reset_index()
```

---

## Summary

In this session, we:

- Loaded and explored the Gapminder dataset  
- Filtered for Nigeria  
- Cleaned and renamed columns  
- Checked for missing data and adjusted types  
- Added a derived column (`total_gdp`)  
- Created a simple line plot  
- Saved a clean version for reuse  
- (Optional) Introduced grouping for future analysis  

---

