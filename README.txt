Hi there, I'm Ose


---
title: "Git + Python: Coding Setup with ISIC-ISCO-CODER"
format:
  html:
    self-contained: true
jupyter: python3
---

## Learning Objectives

- Clone an existing GitHub repo
- Set up Git config in VS Code
- Create and run a basic Python script
- Add, commit, and push code to GitHub

---

## 1. Clone the GitHub Repository

Go to: https://github.com/NBS-Nigeria/ISIC-ISCO-CODER  
Click the green **Code** button → Copy the **HTTPS link**

In **VS Code**, open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`) and type:

```
Git: Clone
```

Then:
- Paste the repo URL
- Select the destination folder
- Choose **Open Repository** when prompted

---

## 2. Set Your Git Identity

In the terminal (`Ctrl+``), run:

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

---

## 3. Create Your First Python File

In the root folder of the cloned repo, create a new file:

**`hello_nigeria.py`**

Paste the following:

```python
# hello_nigeria.py

print("Hello Nigeria! Welcome to your first Python script.")
```

Save the file.

---

## 4. Stage the File

In the terminal:

```bash
git status
git add hello_nigeria.py
```

---

## 5. Commit the File

```bash
git commit -m "Add initial Python script: Hello Nigeria"
```

---

## 6. Push to GitHub

```bash
git push origin main
```

---

## 7. Confirm on GitHub

Visit: https://github.com/NBS-Nigeria/ISIC-ISCO-CODER  
You should now see the `hello_nigeria.py` file in the repo.

---

## Summary

You’ve successfully:
- Cloned a repo
- Set up Git
- Written and saved a Python file
- Committed and pushed code

🎉 **Welcome to Git + Python for data science!**

---
title: "Week 1 Live Coding: GitHub + Python Setup"
format: 
  html:
    self-contained: true
jupyter: python3
---

## 🎯 Session Overview

In this session, we’ll walk through setting up your Python development environment, connecting to GitHub, and making your **first code push**.  
We’ll use the `ISIC-ISCO-CODER` repo as our base.

---

## 1. 🔗 Clone the GitHub Repository

**Goal:** Get the project from GitHub into VS Code.

- Go to [ISIC-ISCO-CODER](https://github.com/NBS-Nigeria/ISIC-ISCO-CODER)
- Click the green **Code** button and **copy the URL**

> 🔵 **Live Demo:**  
> In VS Code → `Ctrl+Shift+P` → select `Git: Clone`  
> Paste the URL  
> Choose your **local folder**  
> Click **Open Repository** when done.

---

## 2. 🧾 Configure Git (One-Time)

**Goal:** Tell Git who you are.

In the terminal:

```bash
git config --global user.name "Your Full Name"
git config --global user.email "you@example.com"
```

✅ Confirm with:

```bash
git config --list
```

> 🔵 **Tip:** These details will appear in every Git commit you make.

---

## 3. 🐍 Create Your First Python File

**Goal:** Run a simple script to test your Python setup.

- In the file explorer, click **New File**
- Name it: `hello_nigeria.py`

Paste this inside:

```python
# hello_nigeria.py

print("Hello Nigeria! Welcome to your first Python script.")
```

Save the file. Then:

> 🔵 **Live Demo:**  
> Right-click → Run Python File in Terminal  
> You should see: `Hello Nigeria!`

---

## 4. 🧮 Git Status Check

In the terminal:

```bash
git status
```

You should see your new file listed as **untracked**.

---

## 5. ✅ Stage the Change

```bash
git add hello_nigeria.py
```

---

## 6. 📌 Commit the Change

```bash
git commit -m "Add hello_nigeria.py script"
```

This saves the change *locally*.

---

## 7. 📤 Push to GitHub

```bash
git push origin main
```

> 🔵 **Live Check:**  
> Go to your GitHub repo — refresh — the file should now appear.

---

## 8. 🧠 Recap

You’ve just:

- Cloned a real GitHub repo
- Set up your Git identity
- Written and run Python in VS Code
- Committed your first script
- Synced your code back to GitHub

---

## 🔚 What’s Next

In Week 2, we’ll build on this by introducing **Markdown**, **Jupyter notebooks**, and **Python variables and logic**.

👏 Well done — your Git + Python setup is complete!
