# 🚀 Quick Start - Hackathon Webhook Setup

## Step-by-Step Instructions

### 1. **Start Your API Server**
```bash
python start_hackathon_server.py
```

You should see:
```
🌿 Parsely AI - Hackathon Server Startup
🚀 Starting Parsely AI Hackathon API Server...
✅ Server is running successfully!
📋 Server Info:
   Health Check: http://localhost:8001/health
   API Docs: http://localhost:8001/docs
   Main Endpoint: http://localhost:8001/hackrx/run

🌐 Now run ngrok in another terminal:
   ngrok http 8001
```

### 2. **Install & Setup Ngrok**
- Download from [ngrok.com](https://ngrok.com)
- Create free account and get auth token
- Run: `ngrok config add-authtoken YOUR_TOKEN`

### 3. **Start Ngrok Tunnel**
In a **new terminal window**:
```bash
ngrok http 8001
```

You'll see output like:
```
Forwarding  https://abc123-def456.ngrok-free.app -> http://localhost:8001
```

### 4. **Your Hackathon Webhook URL**
```
https://abc123-def456.ngrok-free.app/hackrx/run
```

## 🧪 **Test Your Setup**

### Quick Health Check
```bash
curl https://your-ngrok-url.ngrok-free.app/health
```

### API Test
```bash
curl -X POST "https://your-ngrok-url.ngrok-free.app/hackrx/run" \
  -H "Authorization: Bearer hackrx_2024_parsely_ai_token" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
    "questions": ["What is this document about?"]
  }'
```

## 📋 **Hackathon Submission**

**Project:** Parsely AI - LLM Document Processing  
**Webhook URL:** `https://your-ngrok-url.ngrok-free.app/hackrx/run`  
**Method:** POST  
**Authentication:** Bearer Token (`hackrx_2024_parsely_ai_token`)  
**Status:** ✅ Ready for submission!  

## 🔧 **Troubleshooting**

### Server Won't Start
- Check if port 8001 is free
- Make sure GOOGLE_API_KEY is set in .env file

### Ngrok Issues
- Make sure you authenticated with your token
- Try different port if 8001 is busy

### API Errors
- Check server logs in the first terminal
- Verify GOOGLE_API_KEY is valid

## 🎯 **You're Ready!**

Once both terminals are running:
1. ✅ API Server on localhost:8001
2. ✅ Ngrok tunnel providing public URL
3. ✅ Webhook URL ready for hackathon submission

**Copy your ngrok URL and submit it to the hackathon!** 🚀