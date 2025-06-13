Hi there, I'm Ose


# Software List for the Course

In order to engage fully with the course and achieve all of the goals of the
training course, it is necessary that you set-up the following software tools.

These are all free to use and should be relatively easy to set up.
If you have any issues in the set-up, please do not hesitate to ask a question
using the GitHub issues, or reach out to one of the Data Science Campus team.

<h2 style="display: flex; align-items: center; gap: 10px;">
  <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub Logo" width="28">
  GitHub Account
</h2>

[GitHub](https://github.com) is a developer platform that allows developers to create, store, manage and share their code. This is also the platform we will be using to share the course
materials and host the discussion. This can be a great place to see what other
data scientists are up to and what cool projects are being worked on.

In order to set up an account 
Go to [GitHub](https://github.com) and Click Sign Up (top right).

Enter:
Your email address, a username, a secure password and Click Create Account.

You’ll be asked to verify your email—check your inbox and follow the instructions.
Choose Free Plan when prompted.

Once your account is set up, you can customize your profile, add a profile picture, etc.

Once you are set up on GitHub, you will need to ask for access to NBS GitHub organisation

If you envounter any issue, please refer to [these instructions](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account).

<h2 style="display: flex; align-items: center; gap: 10px;">
  <img src="https://git-scm.com/images/logos/downloads/Git-Icon-1788C.png" alt="Git Logo" width="28">
  Git
</h2>

Git is a programming language, we will use it to share our work with colleagues
as well as to download the latest changes to repositories.

Using Git effectively is a foundation to be able to work with others in an effective
way. It can be used to automatically check for changes, make sure that any conflicts in work
are resolved or allow different people in the team work on different parts of a
pipeline without interfering with each other.

It also leaves a trace of changes made to a pipeline, this means that if something
goes wrong it is easy to recover an old version. In more advanced settings, and
in collaboration with GitHub features it can be used to automate parts of your day
to day, like testing the code works as expected.

In order to allow your computer to understand the Git language, you will need
to download it. This can be done by getting the latest source release from
[the official git website](https://git-scm.com/).

For Windows:

Click the Download for Windows button from the official git website.

Open the .exe installer and follow the setup wizard.

Use the default options unless advised otherwise.

Once complete, open Git Bash from your Start Menu to test.

For macOS:

Open the Terminal app.

Type git and press Enter. If Git is not installed, macOS will prompt you to install the developer tools (Xcode Command Line Tools). Then click Install.

Alternatively, install via Homebrew with:
```brew install git
```

For Linux (Debian/Ubuntu):

```sudo apt update
```

```sudo apt install git
```

To configure Git (after install):

```git config --global user.name "Your Name"
```

```git config --global user.email "your@email.com"
```

Test installation:

```git --version
```

<h2 style="display: flex; align-items: center; gap: 10px;">
  <img src="https://www.python.org/static/opengraph-icon-200x200.png" alt="Python Logo" width="28">
  Python
</h2>

Python is a programming language, we will use it to code the data science that
we will learn in the course. It is really flexible and powerful.

It is also open-source, which means free to use, and there is a very large
community working with it and sharing their work. This means that there
are always new projects that let you do new things with Python as well as a large
group of people that can help if you have any problems.

In order to use python effectively, you will need to make sure that your computer
can _"speak and interpret"_ Python. In order to do that visit the [official Python site](https://www.python.org/downloads/) and download the latest version for your machine.

Windows:

Download and run the installer.

VERY IMPORTANT: Check the box that says "Add Python to PATH" before clicking "Install Now".

After installation, open Command Prompt and run:

```python --version
```

```pip --version
```
macOS:

Download the .pkg installer from the Python website. Run the installer and follow the instructions.

Verify in Terminal:

```python3 --version
```

```pip3 --version
```

Linux (Ubuntu):

```sudo apt update
```

```sudo apt install python3 python3-pip
```

This isn't an app on its own, in order to use Python, we will use VS Code.

<h2 style="display: flex; align-items: center; gap: 10px;">
  <img src="https://code.visualstudio.com/assets/favicon.ico" alt="VS Code Logo" width="28">
  Visual Studio Code
</h2>

In order to use Python to write code and check that this works as expected, we
need a computer application to interact with.

At the same time, it would be convenient if this application would let us use Git
to connect to the repository and make sure that our work is kept updated.

There are a variety of tools that we could use for this. In this course, we will
be using Visual Studio Code (also known as VS Code). This is an application that
is free to use, is the industry standard and provides a flexible space, with loads
of features to do our coding in.

In order to get started, download the application from [the official website](https://code.visualstudio.com/).

Run the installer and follow the prompts.

After Installation: Open VS Code.

#### Step 1: Install Essential Extensions

Go to the Extensions tab (left sidebar or press Ctrl+Shift+X).

Install the following extensions: Python by Microsoft, Jupyter (for notebooks), GitHub Pull Requests and Issues, GitLens

#### Step 2: Set Up Python Interpreter

Before creating a virtual environment:

1. Check if Python is installed:
   ```bash
   python --version  # or `python3 --version` on macOS/Linux
   ```
   - If not installed, download from [python.org](https://www.python.org/downloads/)
   
   - **Critical**: Check **"Add Python to PATH"** during installation

---

#### Step 3: Create a Virtual Environment
1. Open your project folder (`File > Open Folder`)
2. Open the terminal (`Ctrl+`` `)
3. Create venv:
   ```bash
   python -m venv .venv
   ```
4. Activate it:
   - **Windows**:
     ```bash
     .venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source .venv/bin/activate
     ```
   → Terminal prompt should show `(.venv)`

5. Select the venv in VS Code:
   - `Ctrl+Shift+P` → **"Python: Select Interpreter"**
   - Choose the Python executable from `.venv`

The set-up steps might be a bit more complicated, so we will cover this during
the sessions, so don't worry too much about getting it perfect.


# Software List for the Course

In order to engage fully with the course and achieve all of the goals of the training course, it is necessary that you set-up the following software tools.

These are all free to use and should be relatively easy to set up. If you have any issues in the set-up, please do not hesitate to ask a question using the GitHub issues, or reach out to one of the Data Science Campus team.

---

## <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub Logo" width="28"> GitHub Account

[GitHub](https://github.com) is a developer platform that allows developers to create, store, manage and share their code. This is also the platform we will be using to share the course materials and host the discussion. This can be a great place to see what other data scientists are up to and what cool projects are being worked on.

### Setup Instructions:
1. Go to [GitHub](https://github.com) and Click Sign Up (top right)
2. Enter:
   - Your email address
   - A username
   - A secure password
3. Click Create Account
4. Verify your email (check your inbox)
5. Choose Free Plan when prompted

### After Setup:
- Customize your profile (add picture, bio etc.)
- Request access to NBS GitHub organisation

**Troubleshooting**: [GitHub Account Help](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account)

---

## <img src="https://git-scm.com/images/logos/downloads/Git-Icon-1788C.png" alt="Git Logo" width="28"> Git

Git is a version control system we will use to share our work with colleagues and manage code changes. It enables collaboration, change tracking, and recovery of previous versions.

### Installation:

#### Windows:
1. Download from [git-scm.com](https://git-scm.com/)
2. Run the .exe installer
3. Use default options unless advised otherwise
4. Test by opening Git Bash from Start Menu

#### macOS:
1. Open Terminal
2. Type `git` and press Enter
3. If not installed, follow prompts to install Xcode Command Line Tools
   - OR install via Homebrew:
     ```bash
     brew install git
     ```

#### Linux (Debian/Ubuntu):
```bash
sudo apt update
sudo apt install git
```

### Configuration:
```bash
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

### Verify Installation:
```bash
git --version
```

---

## <img src="https://www.python.org/static/opengraph-icon-200x200.png" alt="Python Logo" width="28"> Python

Python is the programming language we'll use for data science. It's open-source with a large community and extensive libraries.

### Installation:

#### Windows:
1. Download from [python.org](https://www.python.org/downloads/)
2. **Critical**: Check "Add Python to PATH" during installation
3. Verify in Command Prompt:
   ```bash
   python --version
   pip --version
   ```

#### macOS:
1. Download .pkg installer from Python website
2. Run installer and follow instructions
3. Verify in Terminal:
   ```bash
   python3 --version
   pip3 --version
   ```

#### Linux (Ubuntu):
```bash
sudo apt update
sudo apt install python3 python3-pip
```

---

## <img src="https://code.visualstudio.com/assets/favicon.ico" alt="VS Code Logo" width="28"> Visual Studio Code

VS Code is our free, industry-standard code editor with Git integration and Python support.

### Installation:
1. Download from [code.visualstudio.com](https://code.visualstudio.com/)
2. Run installer (default settings recommended)
3. Launch VS Code

### Setup:

#### Step 1: Install Essential Extensions
Go to Extensions tab (`Ctrl+Shift+X`) and install:
- Python (by Microsoft)
- Jupyter (for notebooks)
- GitHub Pull Requests and Issues
- GitLens

#### Step 2: Set Up Python Interpreter
1. Check Python installation:
   ```bash
   python --version  # or `python3 --version` on macOS/Linux
   ```
   - If missing, install from [python.org](https://www.python.org/downloads/)
   - **Remember**: Check "Add Python to PATH"

#### Step 3: Create a Virtual Environment
1. Open project folder (`File > Open Folder`)
2. Open terminal (`Ctrl+`` `)
3. Create venv:
   ```bash
   python -m venv .venv
   ```
4. Activate it:
   - **Windows**:
     ```bash
     .venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source .venv/bin/activate
     ```
   → Terminal prompt should show `(.venv)`

5. Select the venv in VS Code:
   - `Ctrl+Shift+P` → "Python: Select Interpreter"
   - Choose the Python executable from `.venv`

**Note**: We'll cover setup details during sessions - don't worry about perfection!
