# Automation Scripts

This folder contains a set of small but practical Python scripts that automate frequent Git and Jupyter tasks. Each script is tailored for everyday machine learning and coding workflow.

---

## Scripts

### 1. `pokreniJupyter.py` – Jupyter Notebook Launcher

A graphical interface (Tkinter-based) for selecting and opening Jupyter Notebooks from a specific project folder.

- Recursively scans a fixed directory (`notebook_dir`) for `.ipynb` files, including those inside subfolders
- Displays all found notebooks in a dropdown menu
- Opens the selected notebook with a single click using `jupyter notebook`

**Customization Tip:**  
Change the `notebook_dir` and `jupyter.exe` path to fit your system setup.

---

### 2. `proveriGit.py` – Git Status & Pull Helper

Automates the routine of checking Git repository status and optionally pulling changes.

- Runs `git fetch`, `git status`, and `git diff --stat`
- Prompts user: "Do you want to pull the latest changes?"
- Executes `git pull` if the user confirms

**Customization Tip:**  
Update `repo_path` to point to your Git repository location.

--- 

### 3. `pushujGit.py` – Add, Commit, and Push in One Go

Streamlines the process of saving and pushing changes:

- Automatically stages all changes (`git add .`)
- Prompts for a commit message
- Adds a date prefix to the message (e.g., `26/3 your_message`)
- Pushes to the current branch

**Customization Tip:**  
Make sure the `repoPath` points to your project folder.
