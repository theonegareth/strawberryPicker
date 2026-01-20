# GitHub Setup and Push Guide

Follow these steps to push your strawberry picker ML project to GitHub.

## Prerequisites

1. **Git installed** on your system
   ```bash
   git --version
   ```

2. **GitHub account** (create at https://github.com if you don't have one)

3. **Git configured** with your credentials
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

## Step 1: Create GitHub Repository

### Option A: Using GitHub Website (Recommended)
1. Go to https://github.com/new
2. Repository name: `strawberry-picker-robot`
3. Description: `Machine learning vision system for robotic strawberry picking`
4. Choose: **Public** or **Private**
5. Check: **Add a README file** (we'll overwrite it)
6. Click: **Create repository**

### Option B: Using GitHub CLI (if installed)
```bash
gh repo create strawberry-picker-robot --public --source=. --remote=origin --push
```

## Step 2: Initialize Local Repository

Open terminal in your project folder:

```bash
cd "G:\My Drive\University Files\5th Semester\Kinematics and Dynamics\strawberryPicker"
```

Initialize git repository:
```bash
git init
```

## Step 3: Add Files to Git

Add all files (except those in .gitignore):
```bash
git add .
```

Check what will be committed:
```bash
git status
```

You should see files like:
- `requirements.txt`
- `train_yolov8.py`
- `train_yolov8_colab.ipynb`
- `setup_training.py`
- `TRAINING_README.md`
- `README.md`
- `.gitignore`
- `ArduinoCode/`
- `assets/`

## Step 4: Create First Commit

```bash
git commit -m "Initial commit: YOLOv8 training pipeline for strawberry detection

- Add YOLOv8 training scripts (local, Colab, WSL)
- Add environment setup and validation
- Add comprehensive training documentation
- Add .gitignore for ML project
- Support multiple training environments"
```

## Step 5: Connect to GitHub Repository

If you created repo on GitHub website, link it:

```bash
git remote add origin https://github.com/YOUR_USERNAME/strawberry-picker-robot.git
```

Replace `YOUR_USERNAME` with your actual GitHub username.

## Step 6: Rename Default Branch (if needed)

```bash
git branch -M main
```

## Step 7: Push to GitHub

First push (sets up remote tracking):
```bash
git push -u origin main
```

Enter your GitHub credentials when prompted.

## Step 8: Verify Push

Go to https://github.com/YOUR_USERNAME/strawberry-picker-robot
You should see all your files!

## Step 9: Add .gitignore for Large Files

If you want to add dataset or large model files later, use Git LFS:

```bash
# Install Git LFS
git lfs install

# Track large file types
git lfs track "*.pt"
git lfs track "*.onnx"
git lfs track "*.tflite"
git lfs track "*.h5"

# Add .gitattributes
git add .gitattributes
git commit -m "Add Git LFS tracking for large model files"
git push
```

## Step 10: Create .gitattributes File

Create `.gitattributes` file in project root:

```
*.pt filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.tflite filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
```

## Quick Push Commands (Summary)

```bash
# One-time setup
git init
git add .
git commit -m "Initial commit: YOLOv8 training pipeline"
git remote add origin https://github.com/YOUR_USERNAME/strawberry-picker-robot.git
git branch -M main
git push -u origin main

# Future updates
git add .
git commit -m "Your commit message"
git push
```

## Troubleshooting

### Authentication Issues

If you get authentication errors:

**Option 1: Use Personal Access Token**
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate new token (classic) with `repo` scope
3. Use token as password when prompted

**Option 2: Use SSH (Recommended)**
```bash
# Check if SSH key exists
ls ~/.ssh/id_rsa.pub

# If not, create one
ssh-keygen -t rsa -b 4096 -C "your.email@example.com"

# Add to GitHub
cat ~/.ssh/id_rsa.pub
# Copy output and add to GitHub → Settings → SSH and GPG keys

# Change remote to SSH
git remote set-url origin git@github.com:YOUR_USERNAME/strawberry-picker-robot.git
```

### Large File Issues

If files are too large for GitHub (max 100MB):
- Use Git LFS (see Step 9)
- Or add to `.gitignore` and upload separately to Google Drive/Dropbox

### Proxy Issues (if behind firewall)

```bash
git config --global http.proxy http://proxy.company.com:8080
git config --global https.proxy https://proxy.company.com:8080
```

## Best Practices

### Commit Messages
Write clear, descriptive commit messages:
```bash
git commit -m "Add YOLOv8 training script with Colab support

- Auto-detects training environment
- Supports local, WSL, and Google Colab
- Includes dataset validation
- Exports to ONNX format"
```

### Branching Strategy
```bash
# Create feature branch
git checkout -b feature/add-ripeness-detection

# Work on changes
git add .
git commit -m "Add ripeness classification dataset collection"

# Push branch
git push -u origin feature/add-ripeness-detection

# Create pull request on GitHub
```

### Regular Pushes
Push frequently to avoid losing work:
```bash
# Daily push
git add .
git commit -m "Training progress: epoch 50/100, loss: 0.123"
git push
```

## GitHub Repository Settings

### Protect Main Branch
1. Go to Settings → Branches
2. Add rule for `main` branch:
   - Require pull request reviews
   - Require status checks
   - Include administrators

### Add Description and Topics
1. Go to repository page
2. Click "Edit" next to description
3. Add topics: `machine-learning`, `yolov8`, `raspberry-pi`, `robotics`, `computer-vision`

### Enable Issues and Projects
- Use Issues to track bugs and features
- Use Projects to organize development phases

## Continuous Integration (Optional)

Add `.github/workflows/train.yml` for automated training:

```yaml
name: Train Model

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Validate dataset
      run: |
        python train_yolov8.py --validate-only
```

## Next Steps After Push

1. **Share repository** with teammates/collaborators
2. **Create issues** for Phase 2, 3, 4 tasks
3. **Set up project board** to track progress
4. **Add documentation** to wiki if needed
5. **Enable GitHub Pages** for documentation (optional)

## Getting Help

- GitHub Docs: https://docs.github.com
- Git Cheat Sheet: https://education.github.com/git-cheat-sheet-education.pdf
- Git LFS Docs: https://git-lfs.github.com

## Repository URL

Your repository will be at:
`https://github.com/YOUR_USERNAME/strawberry-picker-robot`

## Clone Command (for others)

```bash
git clone https://github.com/YOUR_USERNAME/strawberry-picker-robot.git
cd strawberry-picker-robot
pip install -r requirements.txt
```

---

**Ready to push?** Run the commands in Step 1-7 above!