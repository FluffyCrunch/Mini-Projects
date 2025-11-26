# Guide: Uploading DL Project to GitHub

This guide will walk you through uploading your Deep Learning colorization project to GitHub.

## Prerequisites

1. A GitHub account (create one at https://github.com if you don't have one)
2. Git installed on your computer
3. Your project files ready in the `DL` folder

## Step-by-Step Instructions

### Step 1: Create Repository on GitHub

1. Go to https://github.com and sign in
2. Click the **"+"** icon in the top right corner
3. Select **"New repository"**
4. Repository name: `Mini-Projects`
5. Description: "Collection of mini projects including DL, NLP, HPC, and BI"
6. Choose **Public** or **Private** (your choice)
7. **DO NOT** initialize with README, .gitignore, or license (we already have files)
8. Click **"Create repository"**

### Step 2: Initialize Git in Your Project (If Not Already Done)

Open PowerShell/Terminal in your `Mini-Project` folder (parent folder, not DL folder):

```bash
# Navigate to your project root
cd D:\Mini-Project

# Initialize git repository (if not already initialized)
git init
```

### Step 3: Add Remote Repository

```bash
# Add your GitHub repository as remote
# Replace YOUR_USERNAME with your actual GitHub username
git remote add origin https://github.com/YOUR_USERNAME/Mini-Projects.git
```

### Step 4: Create Folder Structure on GitHub

GitHub doesn't have a direct "create folder" button. Folders are created automatically when you add files to them.

**Option A: Using Git Commands (Recommended)**

```bash
# Stage the DL folder
git add DL/

# Commit the changes
git commit -m "Add DL project: Colorizing B&W Images using Deep Learning"

# Push to GitHub (creates the folder automatically)
git branch -M main
git push -u origin main
```

**Option B: Using GitHub Web Interface**

1. Go to your repository on GitHub
2. Click **"Add file"** → **"Upload files"**
3. Drag and drop the entire `DL` folder
4. Or click **"Create new file"**
5. Type `DL/README.md` (the `/` creates the folder)
6. Paste your README content
7. Click **"Commit changes"**

### Step 5: Verify Upload

1. Go to your GitHub repository: `https://github.com/YOUR_USERNAME/Mini-Projects`
2. You should see the `DL` folder
3. Click on it to see:
   - `DL.py`
   - `README.md`
   - `requirements.txt`
   - `.gitignore`

## Adding Images to README

To add images to your README:

1. Create a folder `DL/README_images/` in your repository
2. Upload your images there
3. Reference them in README.md using:

```markdown
![Description](README_images/image_name.png)
```

**Example:**
```markdown
## Results

![Before and After](README_images/comparison.png)
```

## Adding More Projects Later

When you want to add other projects (NLP, HPC, BI):

1. Create folders: `NLP/`, `HPC/`, `BI/`
2. Add project files and README to each
3. Commit and push:

```bash
git add NLP/ HPC/ BI/
git commit -m "Add NLP, HPC, and BI projects"
git push
```

## Common Git Commands

```bash
# Check status
git status

# Add all changes
git add .

# Commit changes
git commit -m "Your commit message"

# Push to GitHub
git push

# Pull latest changes
git pull

# View remote URL
git remote -v
```

## Troubleshooting

### If you get "repository not found" error:
- Check your GitHub username is correct
- Verify the repository exists on GitHub
- Make sure you have access permissions

### If you get authentication errors:
- Use GitHub Personal Access Token instead of password
- Or set up SSH keys for authentication

### If folder doesn't appear:
- Make sure you've added at least one file to the folder
- Refresh the GitHub page
- Check you're in the correct branch (usually `main` or `master`)

## Next Steps

1. ✅ Upload DL project
2. Add sample images to `README_images/` folder
3. Update README with actual results/images
4. Add other projects (NLP, HPC, BI) in separate folders
5. Create a main README.md in the root explaining all projects

## Creating Main README for Repository

Create a `README.md` in the root `Mini-Projects` folder:

```markdown
# Mini Projects

Collection of mini projects demonstrating various technologies and concepts.

## Projects

- [DL - Colorizing B&W Images](./DL/) - Deep Learning project for automatic image colorization
- [NLP - Real-time Search Engine](./NLP/) - Natural Language Processing project
- [HPC - High Performance Computing](./HPC/) - HPC project
- [BI - Business Intelligence](./BI/) - BI project

## Overview

This repository contains various mini-projects completed as part of coursework/learning.
```

---

**Need Help?** Check GitHub documentation: https://docs.github.com

