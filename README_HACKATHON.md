# LLM-Powered Intelligent Query-Retrieval System - Hackathon API

## Overview

This is a comprehensive LLM-powered document processing system designed for the hackathon challenge. It processes large documents (PDFs) from blob URLs and answers natural language questions using a 6-stage intelligent pipeline.

## 🏗️ System Architecture

The system implements the required 6-stage pipeline with full document integration:

1. **Input Documents** - Download PDF from Azure blob URLs and extract text content
2. **LLM Parser** - Parse natural language questions using Gemini AI
3. **Embedding Search** - Create semantic embeddings for document chunks using SentenceTransformers
4. **Clause Matching** - Find relevant document sections using cosine similarity matching
5. **Logic Evaluation** - Generate contextual answers using Gemini with relevant document content
6. **JSON Output** - Return structured answers array matching hackathon specification

### Key Technical Features:
- **Document Chunking**: Splits large documents into searchable chunks with overlap
- **Semantic Search**: Uses `all-MiniLM-L6-v2` model for high-quality embeddings
- **Context-Aware Answers**: Provides answers based on actual document content
- **Error Handling**: Graceful fallbacks for PDF processing issues
- **Performance Optimized**: Efficient chunk-based processing for large documents

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Google Gemini API key
- Required Python packages (see `requirements_hackathon.txt`)

### Installation

1. **Clone and setup environment:**
```bash
git clone <repository>
cd <repository>
pip install -r requirements_hackathon.txt
```

2. **Configure environment variables:**
```bash
# Create .env file
GOOGLE_API_KEY=your_gemini_api_key_here
HACKATHON_API_TOKEN=8e6a11e26a0e51d768ce7fb55743017cb25ee7c6891e15c4ab2f1bf971bf9d63
LLM_MODEL=gemini-2.0-flash-exp
```

3. **Start the API server:**
```bash
python start_hackathon_api.py
```

The API will be available at `http://localhost:8000`

## 📡 API Endpoints

### Main Endpoint: `/hackrx/run`

**Method:** POST  
**Authentication:** Bearer token required  
**Content-Type:** application/json

#### Request Format

```json
{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=...",
    "questions": [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?",
        "Does this policy cover maternity expenses?"
    ]
}
```

#### Response Format

```json
{
    "answers": [
        "A grace period of thirty days is provided for premium payment...",
        "There is a waiting period of thirty-six (36) months...",
        "Yes, the policy covers maternity expenses..."
    ]
}
```

#### Example Usage

```bash
curl -X POST "http://localhost:8000/hackrx/run" \
     -H "Authorization: Bearer 8e6a11e26a0e51d768ce7fb55743017cb25ee7c6891e15c4ab2f1bf971bf9d63" \
     -H "Content-Type: application/json" \
     -d '{
       "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=...",
       "questions": [
         "What is the grace period for premium payment?",
         "What is the waiting period for pre-existing diseases?"
       ]
     }'
```

### Health Check: `/health`

**Method:** GET  
**Authentication:** None required

```bash
curl http://localhost:8000/health
```

### API Documentation

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

## 🧪 Testing

### Run Unit Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
python -m pytest tests/test_hackathon_api.py -v
```

### Test the API

```bash
# Run the comprehensive test script
python test_hackathon_api.py
```

### Validate Hackathon Requirements

```bash
# Run comprehensive validation against all hackathon requirements
python validate_hackathon_requirements.py
```

This comprehensive validation tests:
- ✅ API structure and endpoints
- ✅ Bearer token authentication
- ✅ Request/response format compliance
- ✅ PDF blob URL processing
- ✅ 6-stage pipeline implementation
- ✅ Document content integration
- ✅ Performance requirements
- ✅ Technical configuration

## 🐳 Docker Deployment

### Build and Run with Docker

```bash
# Build the Docker image
docker build -f Dockerfile.hackathon -t hackathon-api .

# Run the container
docker run -p 8000:8000 \
  -e GOOGLE_API_KEY=your_api_key \
  -e HACKATHON_API_TOKEN=your_token \
  hackathon-api
```

### Docker Compose

```bash
# Start with docker-compose
docker-compose -f docker-compose.hackathon.yml up -d

# View logs
docker-compose -f docker-compose.hackathon.yml logs -f

# Stop services
docker-compose -f docker-compose.hackathon.yml down
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google Gemini API key | Required |
| `HACKATHON_API_TOKEN` | Bearer token for API authentication | Required |
| `LLM_MODEL` | Gemini model to use | `gemini-2.0-flash-exp` |

### API Configuration

The API is configured to:
- Accept PDF blob URLs from Azure storage
- Process multiple questions per request
- Return structured JSON responses
- Handle authentication via Bearer tokens
- Provide comprehensive error handling

## 📊 Evaluation Criteria Optimization

The system is optimized for all hackathon evaluation parameters:

### 1. **Accuracy**
- Precise query understanding using Gemini AI
- Semantic search for relevant clause matching
- Context-aware answer generation

### 2. **Token Efficiency**
- Optimized prompts to minimize token usage
- Selective content retrieval
- Efficient document chunking strategies

### 3. **Latency**
- Efficient PDF processing pipeline
- Optimized embedding search
- Parallel processing where possible
- Response time typically < 30 seconds

### 4. **Reusability**
- Modular architecture supporting multiple domains
- Configurable business rules
- Extensible component design

### 5. **Explainability**
- Clear decision reasoning
- Source clause traceability
- Detailed justification in responses

## 🏢 Supported Domains

The system handles queries across multiple domains:
- **Insurance:** Policy coverage, claims, premiums
- **Legal:** Contract terms, compliance requirements
- **HR:** Employee policies, benefits
- **Compliance:** Regulatory requirements, audit trails

## 🔒 Security Features

- Bearer token authentication
- Input validation and sanitization
- Secure file handling for temporary downloads
- Error handling without information leakage

## 📈 Performance Monitoring

The API includes built-in monitoring:
- Request/response logging
- Processing time tracking
- Error rate monitoring
- Health check endpoints

## 🚨 Error Handling

The API provides comprehensive error handling:

- **400 Bad Request:** Invalid blob URL or request format
- **401 Unauthorized:** Invalid or missing bearer token
- **422 Unprocessable Entity:** Request validation errors
- **500 Internal Server Error:** Processing failures

All errors include descriptive messages and appropriate HTTP status codes.

## 📝 Logging

The system logs:
- Request processing stages
- Document download and processing
- Question processing results
- Error conditions and recovery

## 🔄 Development Workflow

1. **Local Development:**
   ```bash
   python start_hackathon_api.py
   ```

2. **Testing:**
   ```bash
   python test_hackathon_api.py
   pytest tests/test_hackathon_api.py -v
   ```

3. **Production Deployment:**
   ```bash
   docker-compose -f docker-compose.hackathon.yml up -d
   ```

## 📞 Support

For issues or questions:
1. Check the health endpoint: `/health`
2. Review API documentation: `/docs`
3. Check logs for error details
4. Verify environment configuration

## 🎯 Hackathon Submission Checklist

- ✅ `/hackrx/run` endpoint implemented
- ✅ Bearer token authentication
- ✅ PDF blob URL support
- ✅ Structured JSON responses
- ✅ Error handling and validation
- ✅ Health check endpoint
- ✅ Docker deployment ready
- ✅ Comprehensive testing
- ✅ Documentation complete
- ✅ Performance optimized

The API is ready for hackathon submission and evaluation!