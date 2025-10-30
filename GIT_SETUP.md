# Git Setup Guide

## Initial Setup

```bash
# Initialize repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: NAS-HGM v3.0.0 with unbounded primitive synthesis"

# Add remote
git remote add origin https://github.com/Somebodyhere101/NAS-HGM.git

# Push to remote
git push -u origin main
```

## What's Included

**Core Files** (tracked):
- All Python source files (*.py)
- Documentation (README.md, docs/)
- Configuration (requirements.txt, .gitignore)
- Tests (tests/)

**Excluded** (in .gitignore):
- Python cache (__pycache__/)
- Data files (data/)
- Checkpoints (checkpoints/)
- Results (results/, *.pkl, *.json)
- Virtual environments (venv/, env/)

## Recommended Workflow

```bash
# Create new branch for features
git checkout -b feature/new-primitive

# Make changes and commit
git add .
git commit -m "Add new primitive synthesis method"

# Push to remote
git push origin feature/new-primitive

# Merge via pull request
```

## Clean Repository

Total tracked files: ~15
Repository size: ~150KB (without data/checkpoints)
All generated files properly ignored
