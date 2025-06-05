Hi there, I'm Ose


{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🧪 Week 1 Lab: Git Basics & Your First Python Script\n",
    "\n",
    "Welcome to your first lab session! 🎉 This notebook will guide you through:\n",
    "\n",
    "- Setting up Git on your computer\n",
    "- Creating a Git repository and committing files\n",
    "- Connecting your local project to GitHub\n",
    "- Writing and running your first Python script\n",
    "\n",
    "No prior experience is needed — just follow along and try things out!"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 📁 Step 1: Create a Local Project Folder\n",
    "\n",
    "Open a terminal or command line and run the following:\n",
    "\n",
    "```bash\n",
    # Create a new folder and navigate into it\n",
    "mkdir week1-lab\n",
    "cd week1-lab\n",
    "```\n",
    "\n",
    "If you're using VS Code, you can also open this folder with:\n",
    "```bash\n",
    "code .\n",
    "```\n",
    "\n",
    "This folder will be your working Git repository."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🛠️ Step 2: Initialize Git in Your Project\n",
    "\n",
    "Still in the terminal, initialize Git with:\n",
    "\n",
    "```bash\n",
    "git init\n",
    "```\n",
    "\n",
    "You should see:\n",
    "> Initialized empty Git repository in ...\n",
    "\n",
    "### 🔧 One-time Git Configuration\n",
    "If this is your first time using Git:\n",
    "```bash\n",
    "git config --global user.name \"Your Full Name\"\n",
    "git config --global user.email \"your@email.com\"\n",
    "```\n",
    "\n",
    "✅ Git is now set up!"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🧑‍💻 Step 3: Create Your First Python Script\n",
    "\n",
    "In your project folder, create a file named `hello.py`. Paste in this code:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# hello.py\n",
    "print(\"Hello, Data Science World!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Run the script from terminal:\n",
    "\n",
    "```bash\n",
    "python hello.py\n",
    "```\n",
    "\n",
    "Expected output:\n",
    "```\n",
    "Hello, Data Science World!\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 📝 Step 4: Track Your Changes with Git\n",
    "\n",
    "You can now track and save your Python script with Git.\n",
    "\n",
    "```bash\n",
    "git status        # shows untracked files\n",
    "git add hello.py  # stages the file\n",
    "git commit -m \"Add hello.py script\"  # saves the snapshot\n",
    "```\n",
    "\n",
    "You now have a versioned script! 🧠"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## ☁️ Step 5: Create a GitHub Repository\n",
    "\n",
    "Go to [https://github.com](https://github.com) and:\n",
    "- Click **\"New Repository\"**\n",
    "- Name it `week1-lab`\n",
    "- Keep it public or private as you like\n",
    "- Don’t initialize with a README\n",
    "\n",
    "Now link your local project to GitHub (replace URL):\n",
    "```bash\n",
    "git remote add origin https://github.com/YOUR_USERNAME/week1-lab.git\n",
    "git branch -M main\n",
    "git push -u origin main\n",
    "```\n",
    "✅ Your code is now live on GitHub!"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🧠 Optional Python Challenge\n",
    "\n",
    "Update your script to include your name, today’s date, and a fun fact about Python:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import datetime\n",
    "\n",
    "name = \"[Your Name]\"\n",
    "today = datetime.date.today()\n",
    "fact = \"Python was named after Monty Python, not the snake.\"\n",
    "\n",
    "print(f\"Hello from {name}!\")\n",
    "print(f\"Today's date is {today}.\")\n",
    "print(\"Fun fact:\", fact)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 📌 Bonus Git Practice (Advanced)\n",
    "\n",
    "- Modify the Python file\n",
    "- Commit your changes with a new message\n",
    "- Try `git log` to view commit history\n",
    "\n",
    "```bash\n",
    "git add hello.py\n",
    "git commit -m \"Update script with name and date\"\n",
    "git push\n",
    "git log\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## ✅ Lab Summary\n",
    "\n",
    "**What you did today:**\n",
    "- Set up Git and GitHub\n",
    "- Created and committed your first Python file\n",
    "- Connected your project to a remote repository\n",
    "- Explored basic scripting and Git workflows\n",
    "\n",
    "🎉 Great job! You’ve completed Week 1's technical foundation.\n",
    "\n",
    "**Deliverable:** Submit your GitHub repo URL or upload a screenshot showing the final terminal commands and script output."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

