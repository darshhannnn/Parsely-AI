# 🔧 Railway Deployment Troubleshooting

## Current Issue: 404 Application Not Found

The Railway deployment is showing a 404 error, which typically means:

### **Possible Causes:**
1. **Deployment in Progress** - Railway is still building/deploying
2. **Build Failed** - Check Railway logs for build errors
3. **Environment Variables Missing** - Required vars not set
4. **Port Configuration** - App not binding to correct port
5. **Domain Not Ready** - DNS propagation in progress

### **Quick Fixes:**

#### **1. Check Railway Dashboard**
1. Go to [railway.app](https://railway.app)
2. Open your project: `parsely-ai`
3. Check **Deployments** tab for status
4. Look for any error messages

#### **2. Check Environment Variables**
Ensure these are set in Railway → Variables:
```
HACKATHON_API_TOKEN=your_token_here
GOOGLE_API_KEY=your_gemini_key_here
LLM_MODEL=gemini-2.0-flash
```

#### **3. Check Logs**
In Railway Dashboard → Deployments → View Logs
Look for:
- Build errors
- Runtime errors
- Port binding issues

#### **4. Redeploy**
If needed, trigger a new deployment:
- Railway Dashboard → Deployments → Redeploy

### **Alternative Testing Methods:**

#### **Local Testing**
```bash
# Test locally first
python src/api/hackathon_main.py

# Then test endpoints
curl http://localhost:8000/health
```

#### **Ngrok Backup**
If Railway issues persist:
```bash
# Terminal 1: Start local server
python src/api/hackathon_main.py

# Terminal 2: Expose via ngrok
ngrok http 8000
```

### **Expected Railway Behavior:**

When working correctly, you should see:
```bash
# Health check
curl https://parsely-ai.up.railway.app/health
# Response: {"status": "ok", "service": "LLM Document Processing - Hackathon API", ...}

# API docs
curl https://parsely-ai.up.railway.app/docs
# Response: HTML page with API documentation
```

### **Common Railway Issues:**

#### **Build Timeout**
- **Solution:** Optimize requirements.txt, remove unused dependencies

#### **Memory Limit**
- **Solution:** Railway Dashboard → Settings → Increase memory

#### **Port Binding**
- **Solution:** Ensure app uses `PORT` environment variable

#### **Environment Variables**
- **Solution:** Double-check all required vars are set

### **Backup Deployment Options:**

If Railway continues to have issues:

1. **Render:** `https://render.com`
2. **Heroku:** `https://heroku.com`
3. **Google Cloud Run:** `https://cloud.google.com/run`
4. **Ngrok (temporary):** For immediate testing

### **Status Check Commands:**

```bash
# Check if domain resolves
nslookup parsely-ai.up.railway.app

# Check HTTP response
curl -I https://parsely-ai.up.railway.app/

# Check with verbose output
curl -v https://parsely-ai.up.railway.app/health
```

### **Next Steps:**

1. **Check Railway Dashboard** for deployment status
2. **Review logs** for any error messages
3. **Verify environment variables** are set correctly
4. **Try redeployment** if needed
5. **Use backup deployment** if Railway issues persist

### **Hackathon Submission:**

**Primary URL:** `https://parsely-ai.up.railway.app/hackrx/run`
**Backup Plan:** Have ngrok ready as fallback
**Status:** Monitor Railway dashboard for deployment completion