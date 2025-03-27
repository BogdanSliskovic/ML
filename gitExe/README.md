
## Git Automation Scripts (proveriGit.py & pushujGit.py)

This project contains two simple scripts that automate common Git tasks from the command line. Both scripts are designed for speed and convenience — ideal for quick checks and updates in your Git workflow.

---

### proveriGit.py — Check Git Status & Pull

This script checks the status of a local Git repository and asks whether you'd like to pull the latest changes.

**What it does:**
- Navigates to your local Git project
- Runs `git fetch`, `git status`, and `git diff --stat`
- Asks if you want to run `git pull`

Useful for quickly checking if the remote branch has new commits without manually typing multiple Git commands.

---

### pushujGit.py — Add, Commit, Push

This script adds all modified files, asks for a commit message, and pushes everything to the remote repository.

**How it works:**
- Runs `git add .`
- Prompts you for a commit message
- Appends today’s date to the message
- Runs `git commit` and `git push`

Perfect for speeding up daily Git operations and avoiding repetitive typing.