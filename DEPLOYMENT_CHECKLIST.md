# ✅ Railway Deployment Checklist

## Pre-Deployment
- [ ] All code committed to GitHub
- [ ] `requirements.txt` is complete
- [ ] Environment variables ready
- [ ] API tested locally

## Railway Setup
1. [ ] Go to [railway.app](https://railway.app)
2. [ ] Login with GitHub
3. [ ] Click "New Project" → "Deploy from GitHub repo"
4. [ ] Select your repository
5. [ ] Wait for initial deployment

## Environment Variables
Set these in Railway Dashboard → Variables:
- [ ] `HACKATHON_API_TOKEN=your_secure_token`
- [ ] `GOOGLE_API_KEY=your_gemini_key`
- [ ] `LLM_MODEL=gemini-2.0-flash`

## Testing
- [ ] Health check: `https://your-url.up.railway.app/health`
- [ ] API docs: `https://your-url.up.railway.app/docs`
- [ ] Main endpoint: `https://your-url.up.railway.app/hackrx/run`

## Hackathon Submission
**Webhook URL:** `https://your-url.up.railway.app/hackrx/run`

## Example Test Command
```bash
curl -X POST "https://your-url.up.railway.app/hackrx/run" \
  -H "Authorization: Bearer your_token" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.blob.core.windows.net/docs/sample.pdf",
    "questions": ["What is this document about?"]
  }'
```

## 🎯 Final Webhook URL Format
```
https://[your-app-name].up.railway.app/hackrx/run
```

This is what you submit to the hackathon!