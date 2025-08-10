# 🔑 Google API Key Update Guide

## Quick Update (Recommended)

1. **Run the update script:**
   ```bash
   python update_api_key.py
   ```
   - Enter your new Google API key when prompted
   - Script will update both local and Railway environments

## Manual Update

### Step 1: Get New Google API Key
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create/select a project
3. Enable "Generative Language API"
4. Create API Key in "APIs & Services" → "Credentials"

### Step 2: Update Local Environment
1. Open `.env` file
2. Replace the line:
   ```
   GOOGLE_API_KEY=YOUR_NEW_API_KEY_HERE
   ```
   With your actual API key

### Step 3: Update Railway Deployment
```bash
npx @railway/cli variables --set "GOOGLE_API_KEY=your_actual_api_key_here"
```

### Step 4: Test the Update
```bash
# Test locally
python test_api_simple.py

# Or test deployed version
curl -X POST "https://parsely-ai-production-f2ad.up.railway.app/hackrx/run" \
  -H "Authorization: Bearer 8e6a11e26a0e51d768ce7fb55743017cb25ee7c6891e15c4ab2f1bf971bf9d63" \
  -H "Content-Type: application/json" \
  -d '{"documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf", "questions": ["What is this document about?"]}'
```

## Troubleshooting

### API Key Not Working?
- Ensure the key starts with "AIzaSy"
- Check that Generative Language API is enabled
- Verify billing is set up in Google Cloud
- Try creating a new key

### Railway Not Updating?
- Check Railway CLI is installed: `npm install -g @railway/cli`
- Login to Railway: `railway login`
- Verify project connection: `railway status`

### Still Getting Errors?
- Check Railway logs: `npx @railway/cli logs`
- Verify environment variables: `npx @railway/cli variables`
- Redeploy manually: `npx @railway/cli up`

## Current Status
- ✅ Application is deployed and running
- ✅ API structure is working perfectly
- 🔧 Only need to update Google API key for document processing
- ✅ Ready for hackathon submission

## Your Webhook URL
```
https://parsely-ai-production-f2ad.up.railway.app/hackrx/run
```

This URL is ready for hackathon submission even with the API key issue, as all the required API structure and authentication is working correctly.