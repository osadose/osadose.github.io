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
df = pd.read_csv("data/gapminder.csv")

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
nigeria.to_csv("data/gapminder_nigeria_cleaned.csv", index=False)
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

🎉 Learners now have a tidy, well-structured dataset they can use in later weeks!

```



---
title: "Control Flow, Loops, and Functions in Python"
format:
  html:
    self-contained: true
jupyter: python3
---

## 🎯 Learning Objectives

- Use `if` statements to control code execution
- Use `for` loops to repeat actions
- Write custom Python functions
- Apply control flow to Gapminder data (focus: Nigeria)

---

## 📥 Load and Filter Gapminder Data

```{python}
import pandas as pd

# Load dataset and filter for Nigeria
df = pd.read_csv("data/gapminder.csv")
nigeria = df[df['country'] == 'Nigeria']
nigeria.head()
```

---

## ❓ Using Conditional Statements

Check if GDP per capita was ever below $1,000.

```{python}
for _, row in nigeria.iterrows():
    if row['gdpPercap'] < 1000:
        print(f"{row['year']}: Low GDP (${row['gdpPercap']:.2f})")
```

---

## 🔁 Classifying GDP Levels with a Function

Define a function to label GDP as **low**, **medium**, or **high**.

```{python}
def classify_gdp(gdp):
    if gdp < 1000:
        return "low"
    elif gdp < 3000:
        return "medium"
    else:
        return "high"
```

Apply it to Nigeria data:

```{python}
nigeria['gdp_level'] = nigeria['gdpPercap'].apply(classify_gdp)
nigeria[['year', 'gdpPercap', 'gdp_level']]
```

---

## 🔄 Loop with Function: Print Yearly GDP Category

```{python}
for _, row in nigeria.iterrows():
    category = classify_gdp(row['gdpPercap'])
    print(f"{row['year']}: GDP was {category}")
```

---

## 📊 Comparing Two Years with a Function

Create a function to compare GDP between two years.

```{python}
def compare_gdp(year1, year2, data):
    gdp1 = data[data['year'] == year1]['gdpPercap'].values[0]
    gdp2 = data[data['year'] == year2]['gdpPercap'].values[0]

    if gdp2 > gdp1:
        return f"GDP increased from {year1} to {year2} by ${gdp2 - gdp1:.2f}"
    elif gdp2 < gdp1:
        return f"GDP decreased from {year1} to {year2} by ${gdp1 - gdp2:.2f}"
    else:
        return f"GDP stayed the same between {year1} and {year2}"
```

Try the function:

```{python}
compare_gdp(1982, 2002, nigeria)
```

---

## 🧾 Add a Summary Table with Changes

Add GDP change and growth labels:

```{python}
nigeria['gdp_change'] = nigeria['gdpPercap'].diff().round(2)
nigeria['growth'] = nigeria['gdp_change'].apply(
    lambda x: "increase" if x > 0 else ("decrease" if x < 0 else "no change")
)
nigeria[['year', 'gdpPercap', 'gdp_level', 'gdp_change', 'growth']]
```

---

## ✅ Summary

You’ve learned how to:

- Use `if` to evaluate conditions  
- Use `for` to repeat actions  
- Create functions to reuse logic  
- Analyze real data dynamically  

These tools allow you to write smart, reusable, and readable Python code.

```


