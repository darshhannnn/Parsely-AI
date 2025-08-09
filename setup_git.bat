@echo off
echo 📦 Setting up Git repository for deployment
echo ==========================================

echo Initializing Git repository...
git init

echo Adding all files...
git add .

echo Creating initial commit...
git commit -m "Initial commit - Hackathon API ready for deployment"

echo.
echo ✅ Git repository ready!
echo.
echo Next steps:
echo 1. Create a new repository on GitHub
echo 2. Copy the repository URL
echo 3. Run: git remote add origin YOUR_GITHUB_URL
echo 4. Run: git push -u origin main
echo.
pause