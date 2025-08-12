# 🌐 Ngrok Setup for Hackathon Submission

## Quick Setup Steps

### 1. **Install Ngrok**
- Go to [ngrok.com](https://ngrok.com) and create free account
- Download ngrok for Windows
- Extract to a folder (e.g., `C:\ngrok\`)
- Add to PATH or use full path

### 2. **Authenticate Ngrok**
```bash
# Get your auth token from ngrok dashboard
ngrok config add-authtoken YOUR_NGROK_AUTH_TOKEN
```

### 3. **Start Your Server**

**Option A: Use the batch file**
```bash
# Double-click or run
start_ngrok_server.bat
```

**Option B: Manual setup**
```bash
# Terminal 1: Start API server
python src/api/hackathon_main.py

# Terminal 2: Start ngrok tunnel
ngrok http 8000
```

### 4. **Get Your Webhook URL**
After running ngrok, you'll see output like:
```
Forwarding  https://abc123-def456.ngrok-free.app -> http://localhost:8000
```

**Your hackathon webhook URL:**
```
https://abc123-def456.ngrok-free.app/hackrx/run
```

## 🧪 **Test Your Setup**

### Health Check
```bash
curl https://your-ngrok-url.ngrok-free.app/health
```

### API Documentation
```
https://your-ngrok-url.ngrok-free.app/docs
```

### Sample API Call
```bash
curl -X POST "https://your-ngrok-url.ngrok-free.app/hackrx/run" \
  -H "Authorization: Bearer hackrx_2024_parsely_ai_token" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
    "questions": ["What type of document is this?"]
  }'
```

## 📋 **Hackathon Submission Info**

**Project:** Parsely AI - LLM Document Processing  
**Webhook URL:** `https://your-ngrok-url.ngrok-free.app/hackrx/run`  
**Method:** POST  
**Authentication:** Bearer Token  
**Token:** `hackrx_2024_parsely_ai_token`  

## 🔧 **Troubleshooting**

### Common Issues:
1. **Port already in use**: Change port in hackathon_main.py
2. **Ngrok not found**: Add ngrok to PATH or use full path
3. **Auth token**: Make sure to authenticate ngrok first
4. **Firewall**: Allow Python and ngrok through Windows firewall

### Keep Ngrok Running:
- Don't close the ngrok terminal window
- The URL will change if you restart ngrok
- For permanent URL, upgrade to ngrok paid plan

## 🎯 **Ready for Submission!**

Once ngrok is running, your webhook URL is ready for hackathon submission. The URL will look like:
```
https://[random-string].ngrok-free.app/hackrx/run
```

Copy this exact URL for your hackathon submission! 🚀