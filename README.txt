Hi there, I'm Ose

https://datasciencecampus.github.io/NBS-data-science-course/

Software List for the Course

In order to engage fully with the course and achieve all of the goals of the training course, it is necessary that you set up the following software tools.

These are all free to use and should be relatively easy to set up. If you have any issues in the setup, please do not hesitate to ask a question using the GitHub Issues, or reach out to one of the Data Science Campus team.

GitHub Account
GitHub is a developer platform that allows developers to create, store, manage, and share their code. This is also the platform we will be using to share the course materials and host the discussion. This can be a great place to see what other data scientists are up to and what cool projects are being worked on.

✅ Step-by-Step: Set Up a GitHub Account

Go to https://github.com
Click Sign Up (top right).
Enter:
Your email address
A username
A secure password
Click Create Account.
You’ll be asked to verify your email—check your inbox and follow the instructions.
Choose Free Plan when prompted.
Once your account is set up, you can customize your profile, add a profile picture, etc.
Finally, request access to the NBS GitHub organization. You may need to send your GitHub username to your course instructor or project coordinator.
Git
Git is not a programming language but a version control system. It allows you to track changes in your code and collaborate with others. We will use it to download course content and submit updates.

✅ Step-by-Step: Install Git

For Windows:
Go to https://git-scm.com
Click the Download for Windows button.
Open the .exe installer and follow the setup wizard.
Use the default options unless advised otherwise.
Once complete, open Git Bash from your Start Menu to test.
For macOS:
Open the Terminal app.
Type git and press Enter. If Git is not installed, macOS will prompt you to install the developer tools (Xcode Command Line Tools). Click Install.
Alternatively, install via Homebrew with:
brew install git
For Linux (Debian/Ubuntu):
sudo apt update
sudo apt install git
🔧 Configure Git (after install):

git config --global user.name "Your Name"
git config --global user.email "your@email.com"
Test installation:

git --version
Python
Python is a popular programming language for data science. We'll use it extensively in this course.

✅ Step-by-Step: Install Python

For All Systems:
Visit https://www.python.org/downloads/
Download the latest version (e.g., Python 3.12+) for your OS.
Windows:
Download and run the installer.
VERY IMPORTANT: Check the box that says "Add Python to PATH" before clicking "Install Now".
After installation, open Command Prompt and run:
python --version
pip --version
macOS:
Download the .pkg installer from the Python website.
Run the installer and follow the instructions.
Verify in Terminal:
python3 --version
pip3 --version
Linux (Ubuntu):
sudo apt update
sudo apt install python3 python3-pip
VS Code (Visual Studio Code)
VS Code is a flexible and powerful code editor with support for Python, Git, and many extensions. It's ideal for development during this course.

✅ Step-by-Step: Install VS Code

Go to https://code.visualstudio.com
Click Download for your operating system (Windows/macOS/Linux).
Run the installer and follow the prompts.
After Installation:
Open VS Code.
Go to the Extensions tab (left sidebar or press Ctrl+Shift+X).
Install the following extensions:
Python by Microsoft
Jupyter (for notebooks)
GitHub Pull Requests and Issues
GitLens
Open the Command Palette (Ctrl+Shift+P or Cmd+Shift+P) and run:
Shell Command: Install 'code' command in PATH
(This allows you to open folders in VS Code from your terminal.)
Open a folder:
File > Open Folder and choose a working directory for your projects.
Anaconda
Anaconda is a distribution of Python that includes many scientific and data science packages pre-installed. It’s especially useful for managing environments and dependencies.

✅ Step-by-Step: Install Anaconda

Go to https://www.anaconda.com/products/distribution
Scroll down and download the Individual Edition for your OS.
Run the installer:
Accept default settings
On Windows: Check "Add Anaconda to my PATH environment variable" (recommended)
After installation, open Anaconda Navigator or use the terminal:
conda --version
Optional but useful:

Use conda environments to manage project dependencies:
conda create --name myenv python=3.11
conda activate myenv
Let me know if you’d like a downloadable PDF of this or help creating a checklist version for students.
