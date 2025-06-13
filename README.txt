Hi there, I'm Ose


---
title: "Git Branches and Collaboration"
format:
  html:
    self-contained: true
jupyter: python3
---

## Learning Objectives

- Understand Git branching: what it is and why it matters
- Create and switch between branches in a Git repository
- Use branches to isolate feature changes
- Merge branches using Git
- Push and pull branches on GitHub

---

## Setting the Stage

We’ll continue working in the existing `week1-gapminder` project.

Make sure you’re in the project folder and have a working Python script:

```{bash}
cd week1-gapminder
```

---

## Checking Current Branch

Let’s confirm which branch we’re on:

```{bash}
git status
git branch
```

You should see you're on `main`.

---

## Creating a New Branch

We want to add a simple enhancement (e.g. filtering by year).  
Let’s create a new Git branch for it:

```{bash}
git checkout -b filter-year
```

---

## Editing the Script in a Branch

Open `gapminder_nigeria.py` and make a small change:  
Filter data from **2000 onward**.

```{python}
import pandas as pd

df = pd.read_csv("data/gapminder.csv")
nigeria = df[df['country'] == 'Nigeria']
recent = nigeria[nigeria['year'] >= 2000]

print("Nigeria GDP (from 2000):")
print(recent[['year', 'gdpPercap']])
```

---

## Committing in a Feature Branch

Now save and commit your changes:

```{bash}
git add gapminder_nigeria.py
git commit -m "Filter Nigeria data from year 2000"
```

This is now safely stored in the `filter-year` branch.

---

## Merging Your Feature Branch

Switch back to `main`:

```{bash}
git checkout main
```

Merge your feature branch into `main`:

```{bash}
git merge filter-year
```

If no conflicts occur, the changes are now part of `main`.

---

## Cleaning Up (Optional)

You can delete the branch after merging:

```{bash}
git branch -d filter-year
```

---

## Pushing to GitHub

Make sure your latest work is saved online:

```{bash}
git push origin main
```

If you want to preserve the branch on GitHub before merging:

```{bash}
git push origin filter-year
```

Then create a **Pull Request** on GitHub if collaborating.

---

## Summary

In this session, you learned how to:

- Create a Git branch to safely test changes
- Isolate changes before merging
- Push branches to GitHub
- Use `main` as a clean, working version of your project

🧠 This practice supports confident experimentation and teamwork.

```


