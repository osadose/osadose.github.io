Hi there, I'm Ose


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

