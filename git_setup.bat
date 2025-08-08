@echo off
REM 🌿 Parsely AI - Git Setup and Push to GitHub (Windows)

echo 🌿 Setting up Parsely AI for GitHub...

REM Check if git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Git is not installed. Please install Git first.
    pause
    exit /b 1
)

REM Initialize git repository if not already initialized
if not exist .git (
    echo 📁 Initializing Git repository...
    git init
    echo ✅ Git repository initialized
)

REM Add all files to git
echo 📦 Adding files to Git...
git add .

REM Create initial commit
echo 💾 Creating initial commit...
git commit -m "🌿 Initial commit: Parsely AI - Fresh take on document processing" -m "- Complete Gemini-powered document processing system" -m "- Multi-format document support (PDF, DOCX, EML, TXT, HTML)" -m "- Intelligent claim evaluation with semantic search" -m "- Production-ready Docker deployment" -m "- Comprehensive API with authentication" -m "- Full test suite and monitoring capabilities"

REM Set main branch
git branch -M main

echo ✅ Git setup complete!
echo.
echo 🚀 Next steps:
echo 1. Create a new repository on GitHub named 'parsely-ai'
echo 2. Copy the repository URL
echo 3. Run the following commands:
echo.
echo git remote add origin https://github.com/yourusername/parsely-ai.git
echo git push -u origin main
echo.
echo Replace 'yourusername' with your actual GitHub username
echo.
echo 📋 Repository is ready with:
echo ✅ .gitignore (protects sensitive files)
echo ✅ LICENSE (MIT)
echo ✅ README.md (with badges and documentation)
echo ✅ CONTRIBUTING.md (contribution guidelines)
echo ✅ GitHub Actions (CI/CD workflows)
echo ✅ Issue templates
echo ✅ Pull request template
echo.
echo 🎉 Your Parsely AI project is ready for GitHub!

pause