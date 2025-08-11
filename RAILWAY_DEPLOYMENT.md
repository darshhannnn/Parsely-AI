# 🚂 Railway Deployment Guide

## Quick Deployment Steps

### 1. **Prepare Your Repository**
```bash
# Make sure all files are committed
git add .
git commit -m "Prepare for Railway deployment"
git push origin main
```

### 2. **Deploy to Railway**

**Option A: GitHub Integration (Recommended)**
1. Go to [railway.app](https://railway.app)
2. Sign up/Login with GitHub
3. Click "New Project"
4. Select "Deploy from GitHub repo"
5. Choose this repository
6. Railway will auto-detect Python and deploy

**Option B: Railway CLI**
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway link
railway up
```

### 3. **Set Environment Variables**
In Railway dashboard → Your Project → Variables:

```
HACKATHON_API_TOKEN=hackrx_2024_your_secure_token_here
GOOGLE_API_KEY=your_gemini_api_key_here
LLM_MODEL=gemini-2.0-flash
```

### 4. **Get Your Webhook URL**
After deployment, Railway will provide a URL like:
```
https://parsely-ai-production-abc123.up.railway.app
```

Your hackathon webhook URL will be:
```
https://parsely-ai-production-abc123.up.railway.app/hackrx/run
```

## 🧪 Testing Your Deployment

### Health Check
```bash
curl https://your-railway-url.up.railway.app/health
```

### API Test
```bash
curl -X POST "https://your-railway-url.up.railway.app/hackrx/run" \
  -H "Authorization: Bearer your_token" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.blob.core.windows.net/docs/sample.pdf",
    "questions": ["What is this document about?"]
  }'
```

### API Documentation
Visit: `https://your-railway-url.up.railway.app/docs`

## 📋 Hackathon Submission Info

**Service:** Parsely AI - LLM Document Processing
**Webhook URL:** `https://your-railway-url.up.railway.app/hackrx/run`
**Method:** POST
**Authentication:** Bearer Token
**Health Check:** `https://your-railway-url.up.railway.app/health`

## 🔧 Troubleshooting

### Common Issues:
1. **Build Fails**: Check `requirements.txt` is complete
2. **Environment Variables**: Ensure all required vars are set in Railway
3. **Port Issues**: Railway sets PORT automatically
4. **API Key Issues**: Verify GOOGLE_API_KEY is valid

### Logs:
```bash
railway logs
```

## 🚀 Production Optimizations

### Custom Domain (Optional)
1. Railway Dashboard → Settings → Domains
2. Add your custom domain
3. Update DNS records as instructed

### Scaling
Railway auto-scales based on traffic. For high-load hackathons:
1. Dashboard → Settings → Resources
2. Increase memory/CPU if needed

## 📊 Monitoring

Railway provides built-in monitoring:
- **Metrics**: CPU, Memory, Network usage
- **Logs**: Real-time application logs  
- **Deployments**: Deployment history and rollbacks

Access via Railway Dashboard → Your Project → Observability