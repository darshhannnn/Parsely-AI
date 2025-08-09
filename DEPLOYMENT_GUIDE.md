# 🚀 Hackathon API Deployment Guide

## Option 1: Render (Recommended - No CLI needed)

### Step 1: Prepare Your Code
1. Make sure all files are saved
2. Push your code to GitHub (if not already done)

### Step 2: Deploy to Render
1. Go to [render.com](https://render.com)
2. Sign up/Login with GitHub
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Use these settings:

**Basic Settings:**
- Name: `hackathon-api`
- Environment: `Python 3`
- Build Command: `pip install -r requirements.txt`
- Start Command: `python -m uvicorn src.api.hackathon_main:app --host 0.0.0.0 --port $PORT`

**Environment Variables:**
Add these in the "Environment" section:
- `GOOGLE_API_KEY` = `AIzaSyBtgCpWhnVVRknJvvKbpSJ-MzGCniEXLuM`
- `HACKATHON_API_TOKEN` = `8e6a11e26a0e51d768ce7fb55743017cb25ee7c6891e15c4ab2f1bf971bf9d63`
- `LLM_MODEL` = `gemini-1.5-flash`
- `PYTHONPATH` = `/opt/render/project/src`

6. Click "Create Web Service"
7. Wait for deployment (5-10 minutes)

### Step 3: Get Your Webhook URL
After deployment, your URL will be: `https://your-app-name.onrender.com`
Your webhook URL: `https://your-app-name.onrender.com/hackrx/run`

---

## Option 2: Railway (If CLI works)

### Step 1: Install Railway CLI
```bash
npm install -g @railway/cli
```

### Step 2: Login and Deploy
```bash
railway login
railway new
railway variables set GOOGLE_API_KEY=AIzaSyBtgCpWhnVVRknJvvKbpSJ-MzGCniEXLuM
railway variables set HACKATHON_API_TOKEN=8e6a11e26a0e51d768ce7fb55743017cb25ee7c6891e15c4ab2f1bf971bf9d63
railway up
```

---

## Option 3: Fly.io

### Step 1: Install Fly CLI
Download from: https://fly.io/docs/getting-started/installing-flyctl/

### Step 2: Deploy
```bash
fly auth login
fly launch
fly secrets set GOOGLE_API_KEY=AIzaSyBtgCpWhnVVRknJvvKbpSJ-MzGCniEXLuM
fly secrets set HACKATHON_API_TOKEN=8e6a11e26a0e51d768ce7fb55743017cb25ee7c6891e15c4ab2f1bf971bf9d63
fly deploy
```

---

## Testing Your Deployment

After deployment, test your API:

```bash
python test_deployed_api.py
```

Or manually test:
```bash
curl -X POST "https://your-app.onrender.com/hackrx/run" \
  -H "Authorization: Bearer 8e6a11e26a0e51d768ce7fb55743017cb25ee7c6891e15c4ab2f1bf971bf9d63" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
    "questions": ["What is this document about?"]
  }'
```

## 🎯 Final Submission

Your hackathon webhook URL will be:
**https://your-app-name.onrender.com/hackrx/run**

## Troubleshooting

### Common Issues:
1. **Build fails**: Check requirements.txt
2. **App crashes**: Check environment variables
3. **Timeout**: Increase timeout in Render settings
4. **Import errors**: Check PYTHONPATH setting

### Support:
- Render: https://render.com/docs
- Railway: https://docs.railway.app
- Fly.io: https://fly.io/docs