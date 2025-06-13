Hi there, I'm Ose


---
title: "Git Basics and First Python Steps"
format:
  html:
    self-contained: true
jupyter: python3
---

## Learning Objectives

- Understand what Git is and why it’s useful
- Create and manage a local Git repository
- Write a basic Python script using pandas
- Explore the Gapminder dataset, focusing on Nigeria
- Push a project to GitHub

---

## Setting Up Git

Start by setting your Git identity (only once per machine):

```{bash}
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

This ensures Git tracks changes under your name.

---

## Creating the Project Folder

Use your terminal to create a new project:

```{bash}
mkdir week1-gapminder
cd week1-gapminder
git init
```

You now have a Git-tracked folder ready to work in.

---

## Adding the Gapminder Dataset

Create a `data/` folder and save the dataset as `gapminder.csv`.

You can download it from:

📎 https://github.com/resbaz/r-novice-gapminder-files/raw/master/data/gapminder-FiveYearData.csv

---

## Writing Your First Python Script

Create a file called `gapminder_nigeria.py`:

```{python}
# gapminder_nigeria.py

import pandas as pd

df = pd.read_csv("data/gapminder.csv")
nigeria = df[df['country'] == 'Nigeria']
print(nigeria.head())
```

This loads the dataset and prints rows related to Nigeria.

---

## Saving the Script with Git

Track and commit your script:

```{bash}
git add gapminder_nigeria.py
git commit -m "Initial script to explore Nigeria data"
```

You’ve saved your first version!

---

## Updating and Re-Committing

Let’s improve the script by narrowing focus to GDP:

```{python}
# Updated gapminder_nigeria.py

import pandas as pd

df = pd.read_csv("data/gapminder.csv")
nigeria = df[df['country'] == 'Nigeria']

print("Nigeria GDP per capita (first 5 rows):")
print(nigeria[['year', 'gdpPercap']].head())
```

Then commit the change:

```{bash}
git add gapminder_nigeria.py
git commit -m "Display Nigeria's GDP per capita"
```

---

## Creating a GitHub Repository

1. Go to [github.com](https://github.com)
2. Click **New repository**
3. Name it `week1-gapminder`
4. Leave it empty (don’t add a README or .gitignore)

---

## Connecting Local Git to GitHub

Back in your terminal:

```{bash}
git remote add origin https://github.com/YOUR-USERNAME/week1-gapminder.git
git branch -M main
git push -u origin main
```

Now your local code is backed up online.

---

## Verifying Your Project

Check that:
- Your script appears on GitHub
- Commit messages are visible
- The script runs successfully:

```{bash}
python gapminder_nigeria.py
```

---

## Summary

You have:
- Created a Git-tracked project folder
- Written and updated a Python script using pandas
- Used Git to track changes and history
- Pushed your project to GitHub
- Started exploring real data with code

🎉 Well done — this is the foundation for all future sessions!

```

