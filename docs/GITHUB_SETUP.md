# GitHub setup — Windows PowerShell (once per project)

**Current owner:** `Expcon`. This project must be a NEW repository, not `expcon.github.io`.

## 1. Create the remote repository (owner must click this)

Open https://github.com/new, signed in as Expcon.

- Repository name: `het-foodai`.
- Visibility: **Private** for now (raw images/license constraints have not been cleared).
- **Do not** tick README, .gitignore, or license. The starter already contains README and .gitignore. Creating the remote empty avoids a first-push conflict.
- Click `Create repository`.

## 2. Download/unzip starter into D:\Projects\HeT_FoodAI

Verify files such as README.md, AGENTS.md, src/, scripts/, docs/, tests/ are directly inside `D:\Projects\HeT_FoodAI` (not nested in another folder).

## 3. Run locally before pushing

```powershell
cd D:\Projects\HeT_FoodAI
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\inspect_contract.py
```

## 4. Publish starter into remote (no raw food photos)

```powershell
cd D:\Projects\HeT_FoodAI
git init -b main
git add .
git status
git commit -m "chore: bootstrap shared HeT FoodAI project"
git remote add origin https://github.com/Expcon/het-foodai.git
git push -u origin main
```

On first push, Git Credential Manager may open a GitHub login in the browser. Use your own GitHub account. Never paste any password or token into the chat.

If `git remote add origin` says it already exists, do not re-add: check `git remote -v` before using a correct URL. If the remote repo was accidentally initialized with a README and push fails, do not force-push blindly; ask the lead to reconcile histories.

## 5. Invite teammates

Repository -> Settings -> Collaborators -> Add people. Invite each member using their **exact GitHub username**, granting only the access needed. Their WeChat names are not GitHub usernames.

## 6. Collaboration rules

Each member clones the same repository, creates their branch:

```powershell
git clone https://github.com/Expcon/het-foodai.git
cd het-foodai
git switch -c feat/data-inventory
```

Branch examples: `feat/model-baseline` / `feat/data-inventory` / `feat/app-shell`. Do not all commit directly to main. Open pull requests, review changed files and evidence, then merge.
