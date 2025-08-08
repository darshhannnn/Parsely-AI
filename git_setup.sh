#!/bin/bash
# 🌿 Parsely AI - Git Setup and Push to GitHub

echo "🌿 Setting up Parsely AI for GitHub..."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Git is not installed. Please install Git first.${NC}"
    exit 1
fi

# Initialize git repository if not already initialized
if [ ! -d ".git" ]; then
    echo -e "${YELLOW}📁 Initializing Git repository...${NC}"
    git init
    echo -e "${GREEN}✅ Git repository initialized${NC}"
fi

# Add all files to git
echo -e "${YELLOW}📦 Adding files to Git...${NC}"
git add .

# Create initial commit
echo -e "${YELLOW}💾 Creating initial commit...${NC}"
git commit -m "🌿 Initial commit: Parsely AI - Fresh take on document processing

- Complete Gemini-powered document processing system
- Multi-format document support (PDF, DOCX, EML, TXT, HTML)
- Intelligent claim evaluation with semantic search
- Production-ready Docker deployment
- Comprehensive API with authentication
- Full test suite and monitoring capabilities"

# Set main branch
git branch -M main

echo -e "${GREEN}✅ Git setup complete!${NC}"
echo ""
echo -e "${YELLOW}🚀 Next steps:${NC}"
echo "1. Create a new repository on GitHub named 'parsely-ai'"
echo "2. Copy the repository URL"
echo "3. Run the following commands:"
echo ""
echo -e "${GREEN}git remote add origin https://github.com/yourusername/parsely-ai.git${NC}"
echo -e "${GREEN}git push -u origin main${NC}"
echo ""
echo "Replace 'yourusername' with your actual GitHub username"
echo ""
echo -e "${YELLOW}📋 Repository is ready with:${NC}"
echo "✅ .gitignore (protects sensitive files)"
echo "✅ LICENSE (MIT)"
echo "✅ README.md (with badges and documentation)"
echo "✅ CONTRIBUTING.md (contribution guidelines)"
echo "✅ GitHub Actions (CI/CD workflows)"
echo "✅ Issue templates"
echo "✅ Pull request template"
echo ""
echo -e "${GREEN}🎉 Your Parsely AI project is ready for GitHub!${NC}"