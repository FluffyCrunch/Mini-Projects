# Quick Upload Commands for NLP Project

## Upload NLP folder to GitHub

Run these commands in PowerShell from your `D:\Mini-Project` directory:

### Step 1: Navigate to your project root
```powershell
cd D:\Mini-Project
```

### Step 2: Add the NLP folder to Git
```powershell
git add NLP/
```

### Step 3: Commit the changes
```powershell
git commit -m "Add NLP project: Real-Time Search Engine"
```

### Step 4: Push to GitHub
```powershell
git push
```

---

## What's included in the upload:

- ✅ `RealTimeSearchEngine.py` - Main code (with syntax error fixed)
- ✅ `README.md` - Complete documentation
- ✅ `requirements.txt` - Python dependencies
- ✅ `.gitignore` - Git ignore rules
- ✅ `Data/ChatLog.json` - Sample chat log (if you want to include it)

## What's excluded (by .gitignore):

- ❌ `.env` file (contains API keys - should never be uploaded!)
- ❌ `__pycache__/` folders
- ❌ `.pptx` and `.docx` files (optional, can be changed in .gitignore)

---

## Important Notes:

1. **Never upload `.env` file** - It contains your API keys!
2. The `.gitignore` is configured to exclude sensitive files
3. If you want to include ChatLog.json, comment out that line in `.gitignore`
4. If you want to include report files, remove `*.pptx` and `*.docx` from `.gitignore`

---

## Verify Upload

After pushing, check your GitHub repository:
- Go to: `https://github.com/FluffyCrunch/Mini-Projects/tree/main/NLP`
- You should see all the files listed above

