# 🚀 Hackathon Deployment Guide

## Pre-Deployment Checklist

### ✅ Environment Setup
- [ ] Copy `.env.example` to `.env`
- [ ] Set `GOOGLE_API_KEY` with your Gemini API key
- [ ] Verify `HACKATHON_API_TOKEN` is set correctly
- [ ] Test locally: `python start_hackathon_api.py`

### ✅ API Validation
- [ ] Run validation: `python validate_hackathon_requirements.py`
- [ ] Test security: `python test_api_security.py`
- [ ] Verify all tests pass in: `pytest tests/test_hackathon_api.py -v`

### ✅ Required API Structure
- [x] POST endpoint: `/hackrx/run`
- [x] Bearer token authentication
- [x] JSON request/response format
- [x] PDF blob URL processing
- [x] 6-stage pipeline implementation

## Deployment Options

### Option 1: Railway (Recommended)
```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login and deploy
railway login
railway init
railway up

# 3. Set environment variables in Railway dashboard
GOOGLE_API_KEY=your_key_here
HACKATHON_API_TOKEN=8e6a11e26a0e51d768ce7fb55743017cb25ee7c6891e15c4ab2f1bf971bf9d63
```

### Option 2: Heroku
```bash
# 1. Install Heroku CLI and login
heroku login

# 2. Create app
heroku create your-app-name

# 3. Set environment variables
heroku config:set GOOGLE_API_KEY=your_key_here
heroku config:set HACKATHON_API_TOKEN=8e6a11e26a0e51d768ce7fb55743017cb25ee7c6891e15c4ab2f1bf971bf9d63

# 4. Deploy
git push heroku main
```

### Option 3: Docker + Cloud Provider
```bash
# 1. Build Docker image
docker build -f Dockerfile.hackathon -t hackathon-api .

# 2. Test locally
docker run -p 8000:8000 \
  -e GOOGLE_API_KEY=your_key \
  -e HACKATHON_API_TOKEN=8e6a11e26a0e51d768ce7fb55743017cb25ee7c6891e15c4ab2f1bf971bf9d63 \
  hackathon-api

# 3. Deploy to your cloud provider
```

## API Specification Compliance

### ✅ Required Endpoint
```
POST /hackrx/run
Authorization: Bearer 8e6a11e26a0e51d768ce7fb55743017cb25ee7c6891e15c4ab2f1bf971bf9d63
Content-Type: application/json
```

### ✅ Request Format
```json
{
  "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=...",
  "questions": [
    "What is the grace period for premium payment?",
    "What is the waiting period for pre-existing diseases?"
  ]
}
```

### ✅ Response Format
```json
{
  "answers": [
    "A grace period of thirty days is provided...",
    "There is a waiting period of thirty-six months..."
  ]
}
```

## Testing Your Deployment

### 1. Health Check
```bash
curl https://your-domain.com/health
```

### 2. Authentication Test
```bash
curl -X POST "https://your-domain.com/hackrx/run" \
  -H "Authorization: Bearer 8e6a11e26a0e51d768ce7fb55743017cb25ee7c6891e15c4ab2f1bf971bf9d63" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
    "questions": ["What is this document about?"]
  }'
```

### 3. Full Integration Test
```bash
curl -X POST "https://your-domain.com/hackrx/run" \
  -H "Authorization: Bearer 8e6a11e26a0e51d768ce7fb55743017cb25ee7c6891e15c4ab2f1bf971bf9d63" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
      "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
      "What is the waiting period for pre-existing diseases (PED) to be covered?"
    ]
  }'
```

## Submission Requirements

### ✅ What to Submit
1. **Webhook URL**: `https://your-domain.com/hackrx/run`
2. **Description**: "FastAPI + Gemini AI + SentenceTransformers - 6-stage LLM pipeline for intelligent document processing"

### ✅ Technical Stack
- **Backend**: FastAPI
- **LLM**: Google Gemini (gemini-2.0-flash-exp)
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **Vector Search**: Cosine similarity with scikit-learn
- **Document Processing**: PyPDF2 + OCR fallback
- **Authentication**: Bearer token
- **Deployment**: Docker-ready

### ✅ Performance Characteristics
- **Response Time**: < 30 seconds
- **Concurrent Requests**: Supported with rate limiting
- **Document Size**: Handles large PDF files
- **Question Limit**: 1-10 questions per request
- **Accuracy**: Context-aware answers with semantic search

## Troubleshooting

### Common Issues
1. **API Key Error**: Verify `GOOGLE_API_KEY` is set correctly
2. **Authentication Failed**: Check `HACKATHON_API_TOKEN` matches exactly
3. **PDF Download Failed**: Ensure blob URL is accessible and valid
4. **Timeout**: Large documents may take longer, ensure 30s+ timeout
5. **Memory Issues**: Consider increasing container memory for large PDFs

### Debug Commands
```bash
# Check environment
python -c "import os; print('GOOGLE_API_KEY:', bool(os.getenv('GOOGLE_API_KEY')))"

# Test Gemini connection
python -c "import google.generativeai as genai; genai.configure(api_key='your_key'); print('Gemini OK')"

# Validate deployment
python validate_hackathon_requirements.py
```

## Success Criteria

Your API is ready for submission when:
- ✅ All validation tests pass
- ✅ API responds within 30 seconds
- ✅ HTTPS endpoint is accessible
- ✅ Bearer token authentication works
- ✅ PDF blob URLs are processed correctly
- ✅ Answers are contextually relevant
- ✅ Response format matches specification exactly

## Final Submission

1. **Deploy** your API to a public HTTPS endpoint
2. **Test** the deployed endpoint thoroughly
3. **Submit** your webhook URL: `https://your-domain.com/hackrx/run`
4. **Monitor** your deployment during evaluation

Good luck with your hackathon submission! 🎉