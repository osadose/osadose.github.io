Hi there, I'm Ose



[
```markdown
---
title: "Week 2: Git Branches and Collaboration"
format: revealjs
theme: simple
logo: template/logo.png
---

## 1. Welcome to Week 2! 🚀

**Theme:** Git Branching & Collaboration

This week, you’ll:
- Understand how teams collaborate using Git
- Use branches to safely make changes
- Learn how to merge code and resolve conflicts

🧠 Key Skill: Working on shared projects without chaos

---

## 2. Why Use Branches? 🌿

Git branches let you:
- Experiment without breaking the main code
- Work in parallel with your teammates
- Organize your project into logical feature units

🧪 Example:
> Main branch = stable version  
> New feature → create a new branch → test it → merge it back

---

## 3. Git Branch Basics 🧱

Common branch commands:

```bash
git branch               # List branches
git checkout -b new-feature   # Create + switch
git switch main          # Go back to main
```

📌 Each branch is a snapshot of the project at a point in time.

✅ Practice: Create a branch, switch between branches, delete one

---

## 4. Merging Branches 🔀

When you're ready to combine work:

```bash
git merge feature-branch
```

This merges changes from `feature-branch` into the branch you're on.

🧠 Git tries to merge automatically. But if changes overlap…

⚠️ **Merge conflict** happens!

---

## 5. Resolving Merge Conflicts 🩹

Git will show files with conflicts:

```bash
<<<<<<< HEAD
your change
=======
their change
>>>>>>> feature-branch
```

🛠️ You resolve it manually → save → then:

```bash
git add conflicted-file
git commit
```

💡 Tip: Use VS Code or GitHub Desktop to resolve conflicts visually

---

## 6. Collaboration with GitHub 🤝

To work with others, you’ll:
1. Clone a repo
2. Make changes in a new branch
3. Push your branch to GitHub
4. Open a **pull request**

🎯 Pull Requests = Propose changes + get feedback + merge cleanly

---

## 7. Pull Request Workflow 📤

### On GitHub:
1. Push your branch:
```bash
git push -u origin new-feature
```
2. Click “Compare & Pull Request”
3. Add description → click “Create Pull Request”

💬 Team members can now review, comment, and approve.

---

## 8. Working as a Team 👥

**Key habits for collaboration:**
- Pull the latest changes often:  
  `git pull origin main`
- Communicate: use comments, issues
- Keep commits small and meaningful
- Don’t fear conflicts — learn to resolve them

✅ Good Git habits = less stress for your team

---

## 9. This Week’s Practice 🔁

📌 By the end of this week, you should:
- Create and switch branches
- Merge without conflicts
- Resolve merge conflicts when they occur
- Push branches to GitHub
- Open and review pull requests

🧪 Optional: Collaborate with a classmate on a small script

---

## 10. Wrap-Up & What’s Next 🎯

**Week 2 Summary:**
- Branching keeps work organized
- Collaboration means communication
- GitHub is your team’s central hub

📅 Next week: **Python Data Structures!**

🎓 Tip: If you're confused, simulate collaboration with yourself in two folders!

```
]
