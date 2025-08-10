# 🚀 Hackathon API - Clean Project Structure

## 📁 Essential Files Only

### **🔧 Core Application**
```
src/api/hackathon_main.py          # Main FastAPI application
main.py                            # Railway deployment entry point
run.py                             # Alternative startup script
start_hackathon_api.py            # Local development server
```

### **⚙️ Configuration**
```
.env                               # Environment variables
requirements.txt                   # Python dependencies
```

### **🚀 Deployment**
```
Dockerfile                         # Docker container configuration
railway.json                       # Railway deployment config
Procfile                          # Process configuration
render.yaml                       # Render deployment config (alternative)
fly.toml                          # Fly.io deployment config (alternative)
vercel.json                       # Vercel deployment config (alternative)
```

### **🧪 Testing & Validation**
```
validate_hackathon_requirements.py # Comprehensive API validation
test_api_simple.py                 # Simple API testing script
test_deployed_api.py               # Deployed API testing
tests/test_hackathon_api.py        # Unit tests for hackathon API
```

### **🌐 Web Interface**
```
web_interface.html                 # Complete testing web interface
```

### **📋 Utilities**
```
update_api_key.py                  # Google API key update utility
deploy.py                          # Multi-platform deployment helper
API_KEY_UPDATE_GUIDE.md           # API key update instructions
```

### **📚 Documentation**
```
README.md                          # Project documentation
LICENSE                           # License file
```

## 🎯 **Your Deployed Application**

**Webhook URL:** `https://parsely-ai-production-f2ad.up.railway.app/hackrx/run`

**Status:** ✅ **READY FOR HACKATHON SUBMISSION**

## 🧪 **How to Test**

1. **Web Interface:** Open `web_interface.html`
2. **API Docs:** https://parsely-ai-production-f2ad.up.railway.app/docs
3. **Validation:** `python validate_hackathon_requirements.py`
4. **Simple Test:** `python test_api_simple.py`

## 🔧 **Development Commands**

```bash
# Start local server
python start_hackathon_api.py

# Update API key
python update_api_key.py

# Validate API compliance
python validate_hackathon_requirements.py

# Deploy to Railway
npx @railway/cli up
```

## 📦 **Removed Files**

The following non-essential files were removed to clean up the project:
- GitHub templates and workflows
- Unused modules (semantic_search, document_processing, etc.)
- Development specs and planning documents
- Duplicate deployment files
- Example and template files
- Brand and contributing guidelines

**Result:** Clean, focused project with only essential files for hackathon deployment! 🎉