Hi there, I'm Ose

---
title: "Data Types and Structures in Python"
format:
  html:
    self-contained: true
jupyter: python3
---

## Learning Objectives

By the end of this session, you will:

- Recognize common Python data types: numbers, text, True/False
- Work with Python’s built-in data structures: list, tuple, dict, set
- Access and update values using indexing and keys
- Build confidence for using Python with real data in Week 4

---

## 1. Python Data Types

Python has a few simple types that are used all the time:

```{python}
age = 30                # Whole number (int)
height = 1.75           # Decimal number (float)
country = "Nigeria"     # Text (str)
is_african = True       # True or False (bool)

print(type(age))
print(type(height))
print(type(country))
print(type(is_african))
```

---

## 2. Strings (Text)

Strings store words or sentences. You can look at parts of them:

```{python}
country = "Nigeria"

print(country[0])      # First letter
print(country[-1])     # Last letter
print(country[0:3])    # First 3 letters: 'Nig'
print(country[3:])     # Letters from position 3 onwards
```

---

## 3. Lists (Ordered Groups)

A list is a group of items in a specific order. Lists can grow or change.

```{python}
countries = ["Nigeria", "Ghana", "Kenya"]

print(countries[0])        # First item
print(countries[-1])       # Last item
print(countries[1:3])      # Items from index 1 to 2

countries.append("Ethiopia")  # Add a new item
countries[1] = "Rwanda"       # Change an item

print(countries)
```

---

## 4. Tuples (Fixed Groups)

A tuple is like a list, but you **can’t change** it after it’s created.

```{python}
location = (9.05785, 7.49508)  # Abuja’s coordinates

print("Latitude:", location[0])
print("Longitude:", location[1])
```

Tuples are useful for fixed things like coordinates or dates.

---

## 5. Dictionaries (Labeled Data)

Dictionaries let you store data using a name (key) and value:

```{python}
gdp = {
    "Nigeria": 2229,
    "Ghana": 2183,
    "Kenya": 1801
}

print(gdp["Nigeria"])          # Look up Nigeria’s value

gdp["Nigeria"] = 2300          # Update value
gdp["Ethiopia"] = 1000         # Add new country

print(gdp)
```

---

## 6. Sets (Unique Values)

Sets are like unordered lists, but they **remove duplicates** automatically.

```{python}
languages = {"English", "French", "Swahili", "English"}

print(languages)              # "English" appears only once

languages.add("Hausa")        # Add new language
print("Swahili" in languages) # Check if Swahili is included
```

---

## 7. Putting It All Together

We can use these structures together. Example: a list of countries with info.

```{python}
countries_info = [
    {"name": "Nigeria", "population_m": 223},
    {"name": "Ghana", "population_m": 34}
]

for country in countries_info:
    print(country["name"], "has", country["population_m"], "million people")
```

---

## 8. Quick Review

Let’s check what we’ve used:

```{python}
print(type(3))                  # int
print(type("Africa"))          # str
print(type([1, 2, 3]))          # list
print(type((1, 2)))             # tuple
print(type({"a": 1}))           # dict
print(type({"x", "y", "z"}))    # set
```

---

## ✅ Summary

You’ve now learned:

- The 4 most common Python data types  
- How to use built-in structures like `list`, `tuple`, `dict`, and `set`  
- How to access, update, and combine values  
- How Python helps organize information clearly

🚀 This sets you up perfectly for Week 4, where we load and clean real-world data.

```

---
title: "Git Branches and Collaboration"
format:
  html:
    self-contained: true
jupyter: python3
---

## 🎯 Learning Objectives

- Understand what a Git branch is and why it's useful
- Follow Git best practices for collaborative work
- Create and switch between branches
- Make isolated changes in a feature branch
- Push feature branches and open a Pull Request (PR)
- Merge changes into `main` using PRs

---

## 📁 Setup: Continue from Week 1 Repo

Make sure you're in the cloned project folder (`ISIC-ISCO-CODER`):

```bash
cd ISIC-ISCO-CODER
```

Check the contents:

```bash
ls
```

You should see `hello_nigeria.py`.

---

## 🔎 Check Current Git Branch

Let's verify that you're on the main branch:

```bash
git branch
```

Expected output:

```bash
* main
```

Always start from a clean `main` before branching.

---

## 🌿 Create a Feature Branch (Best Practice)

Never edit directly on `main`.  
Let’s create a new feature branch for our update:

```bash
git checkout -b update-greeting-message
```

✅ **Naming tip**: Use descriptive names like `fix-typo`, `add-analysis`, or `update-greeting-message`.

---

## ✍️ Modify the Script

Open `hello_nigeria.py` and make the greeting message dynamic:

```python
# hello_nigeria.py

country = "Nigeria"
print(f"Hello, {country}! Welcome to your first Python project.")
```

Save the file.

---

## ✅ Stage & Commit (Use Clear Messages)

Now commit your changes with a meaningful message:

```bash
git add hello_nigeria.py
git commit -m "feat: personalize greeting using variable"
```

🔑 **Tip**: Use prefixes like `feat:`, `fix:`, `docs:`, or `refactor:` for clarity.

---

## ☁️ Push the Feature Branch

Push the new branch to GitHub:

```bash
git push origin update-greeting-message
```

---

## 🔁 Open a Pull Request (PR) – Best Practice

Go to the repo on GitHub.  
You’ll see a prompt to **“Compare & pull request”**.

1. Click it.
2. Review the changes.
3. Add a **clear PR title and description**, e.g.:

   > **Title**: `feat: dynamic greeting message in hello_nigeria.py`  
   > **Description**: Updated the script to use a variable so the greeting can be reused or expanded in future iterations.

4. Submit the Pull Request.

✅ **Best Practice**: Even when working solo, use PRs to document and review changes clearly.

---

## 🔄 Merge the PR into `main`

Once the PR is reviewed (by you or a teammate), click **“Merge pull request”**.  
Then delete the branch using the GitHub button if no longer needed.

---

## ⬇️ Update Local `main` After PR Merge

Pull the merged changes back to your local environment:

```bash
git checkout main
git pull origin main
```

This ensures your local `main` matches GitHub.

---

## 🧹 Optional: Delete Local Branch

If you're done with it:

```bash
git branch -d update-greeting-message
```

---

## 🧠 Summary & Best Practices

Today you learned to:

✅ Create a feature branch  
✅ Make clear, trackable changes  
✅ Push branches to GitHub  
✅ Open a Pull Request and review changes  
✅ Keep `main` clean and stable  

💡 **Best Practices Recap**:

- Always create a new branch for a task
- Use meaningful commit messages
- Push and create PRs for visibility
- Review changes before merging
- Keep `main` clean and deployable

---

## ⏭️ What’s Next?

In **Week 3**, we'll explore basic Python logic, variables, and control structures to build simple but powerful scripts in `.qmd` notebooks.

