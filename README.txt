Hi there, I'm Ose


---
title: "Data Types and Structures in Python"
format:
  html:
    self-contained: true
jupyter: python3
---

## Learning Objectives

- Understand Python's core data types (int, float, str, bool)
- Use built-in data structures: lists, tuples, dictionaries
- Work with pandas Series and DataFrames
- Apply indexing, slicing, and data filtering
- Use Gapminder data to practice these concepts

---

## Introduction to Python Data Types

Basic examples of common types:

```{python}
age = 30               # int
height = 1.75          # float
name = "Nigeria"       # str
is_african = True      # bool

print(type(age), type(height), type(name), type(is_african))
```

---

## Working with Lists and Tuples

Lists are ordered and mutable:

```{python}
countries = ['Nigeria', 'Ghana', 'Kenya']
print(countries[0])
countries.append('Ethiopia')
print(countries)
```

Tuples are ordered and immutable:

```{python}
coords = (9.05785, 7.49508)  # Abuja's coordinates
print(coords)
```

---

## Using Dictionaries

Dictionaries store key-value pairs:

```{python}
gdp_dict = {
    "Nigeria": 2229,
    "Ghana": 2183,
    "Kenya": 1801
}

print(gdp_dict["Nigeria"])
```

Update a value:

```{python}
gdp_dict["Nigeria"] = 2300
print(gdp_dict)
```

---

## Loading the Gapminder Dataset

We use pandas to load and structure the data:

```{python}
import pandas as pd

df = pd.read_csv("data/gapminder.csv")
print(df.head())
```

Check column types:

```{python}
df.dtypes
```

---

## Filtering Rows and Selecting Columns

Focus on data for Nigeria:

```{python}
nigeria = df[df['country'] == 'Nigeria']
print(nigeria[['year', 'lifeExp', 'gdpPercap']])
```

Select a column as a Series:

```{python}
nigeria_gdp = nigeria['gdpPercap']
print(type(nigeria_gdp))
print(nigeria_gdp.head())
```

---

## Summary Statistics and Data Types

Explore types and describe numerical data:

```{python}
print(nigeria.dtypes)
print(nigeria.describe())
```

Check if values match expected types:

```{python}
print(isinstance(nigeria['year'].iloc[0], int))   # Should be True
print(isinstance(nigeria['lifeExp'].mean(), float))  # Should be True
```

---

## Combining Data Types and Structures

Example: Create a summary dictionary:

```{python}
summary = {
    "country": "Nigeria",
    "years": list(nigeria['year']),
    "avg_gdp": nigeria['gdpPercap'].mean(),
    "has_nulls": nigeria.isnull().any().any()
}

print(summary)
```

---

## Summary

In this live session, you:

- Worked with basic data types in Python
- Used lists, tuples, and dictionaries
- Loaded and filtered a real-world dataset
- Explored data using pandas structures

🎯 You now understand how Python structures help organize and manipulate real data.

```



