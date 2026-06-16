# 🌿 Parsely AI

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)
[![Gemini](https://img.shields.io/badge/Gemini-1.5%20Flash-orange.svg)](https://ai.google.dev)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Fresh take on document processing** 🌿

## Overview
**Parsely AI** is a powerful, Gemini-powered intelligent document processing platform that uses Large Language Models to process natural language queries and retrieve relevant information from large unstructured documents such as policy documents, contracts, and emails. The system provides intelligent claim evaluation, semantic search, and structured decision-making with complete audit trails.

## 🌿 Key Features
- **🧠 Gemini-Powered Intelligence**: Uses Google's Gemini 1.5 Pro for advanced natural language understanding
- **📄 Multi-Format Document Support**: PDF, DOCX, EML (email), TXT, and HTML documents
- **🔍 Semantic Search**: Advanced semantic understanding beyond simple keyword matching
- **⚖️ Intelligent Decision Making**: AI-powered claim evaluation with detailed justifications
- **📋 Complete Audit Trails**: Full traceability from query to decision with clause mapping
- **💰 Cost-Effective**: Uses Gemini API (~90% cheaper than OpenAI GPT-4)
- **🌐 RESTful API**: FastAPI-based API with interactive documentation
- **🖥️ Web Interface**: Streamlit-based user interface
- **🔧 Extensible Architecture**: Modular design supporting multiple domains

## 🔧 Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Gemini API Key
**Option A: Interactive Setup**
```bash
python setup_api_key.py
```

**Option B: Manual Setup**
1. Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Copy `.env.example` to `.env`
3. Replace `your_gemini_api_key_here` with your actual API key

### 3. Test Your Setup
```bash
python test_gemini_setup.py
```

### 4. Add Policy Documents
Place your policy documents in the `data/policies/` directory (PDF, DOCX, or EML files).

## 🎯 Usage

### Command Line
```python
from src.insurance_claim_processor import InsuranceClaimProcessor

processor = InsuranceClaimProcessor()
result = processor.process_claim("46-year-old male, knee surgery in Pune, 3-month-old policy")
print(result)
```

### FastAPI Server
```bash
uvicorn src.api.main:app --reload
```
Visit [http://localhost:8000/docs](http://localhost:8000/docs) for interactive API documentation.

### Streamlit Web Interface
```bash
streamlit run src/ui/streamlit_app.py
```

## 📊 API Endpoints

### Core Endpoints
- `POST /process_claim` - Complete claim evaluation
- `GET /analyze_query` - Extract entities from queries
- `GET /search_clauses` - Semantic search for relevant clauses
- `GET /policy_summary` - Get policy statistics

### Gemini-Enhanced Endpoints
- `GET /classify_intent` - Classify query intent using AI
- `GET /expand_query` - Generate search variations
- `GET /health` - System health and configuration

## 🏗️ Architecture

### Enhanced Components
- **GeminiQueryParser**: AI-powered entity extraction and intent classification
- **GeminiClaimEvaluator**: Intelligent decision making with detailed reasoning
- **SemanticRetriever**: Hybrid search combining semantic and keyword matching
- **UniversalDocumentProcessor**: Multi-format document processing

### Data Flow
```
Query → Gemini Parser → Semantic Search → Gemini Evaluator → Structured Decision
```

## 💰 Cost Benefits

**Gemini vs OpenAI Pricing:**
- Gemini 1.5 Pro: ~$3.50 per 1M tokens
- Gemini 1.5 Flash: ~$0.35 per 1M tokens  
- OpenAI GPT-4: ~$30 per 1M tokens

**90% cost savings** while maintaining high-quality results!

## 🔍 Example Usage

### Sample Query
```
"46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
```

### Sample Response
```json
{
  "decision": "approved",
  "amount": "₹200,000.00",
  "justification": "Claim approved based on policy coverage for orthopedic procedures...",
  "mapped_clauses": [
    {
      "clause_id": "Section_2.1",
      "text": "Orthopedic surgeries including knee procedures are covered...",
      "relevance": "High relevance: Defines coverage eligibility"
    }
  ],
  "confidence_score": 0.95,
  "query_analysis": {
    "extracted_entities": {
      "age": 46,
      "gender": "male",
      "procedure": "knee surgery",
      "location": "Pune",
      "policy_age_months": 3
    }
  }
}
```

## 📁 Project Structure
```
├── src/
│   ├── api/                    # FastAPI endpoints
│   ├── decision_engine/        # AI-powered claim evaluation
│   │   ├── gemini_claim_evaluator.py
│   │   └── claim_evaluator.py
│   ├── document_processing/    # Multi-format document processors
│   ├── query_parsing/          # AI-powered query parsing
│   │   ├── gemini_query_parser.py
│   │   └── query_parser.py
│   ├── semantic_search/        # Hybrid semantic search
│   └── ui/                     # Streamlit interface
├── config/
│   ├── gemini_config.py        # Gemini API configuration
│   └── settings.py             # Application settings
├── .kiro/specs/                # Feature specifications
├── test_gemini_setup.py        # Comprehensive setup testing
├── setup_api_key.py           # Interactive API key setup
└── requirements.txt            # Dependencies
```

## 🧪 Testing

### Automated Testing (CI/CD)
The project includes comprehensive automated testing via GitHub Actions:

- **Continuous Integration**: Runs on every push and pull request
- **Multi-Python Testing**: Tests on Python 3.8, 3.9, and 3.10
- **Code Quality Checks**: Black, Flake8, isort, Bandit, and mypy
- **Security Scanning**: Dependency and code vulnerability checks
- **Docker Validation**: Container build and health testing
- **Coverage Reporting**: Minimum 80% test coverage required

### Local Testing

#### Run All Tests
```bash
# Comprehensive test suite
pytest tests/ -v --cov=src --cov-report=html

# Quick local CI simulation
python scripts/local_ci_test.sh  # Linux/Mac
python scripts/test_workflows.py  # Cross-platform
```

#### Test Individual Components
```bash
# Test query parsing
python -c "from src.query_parsing.gemini_query_parser import GeminiQueryParser; parser = GeminiQueryParser(); print(parser.parse_query('test query'))"

# Test API endpoints
curl -X POST "http://localhost:8000/process_claim" -H "Content-Type: application/json" -d '{"query": "test claim"}'

# Test Docker build
docker build -t parsely-ai-test .
docker run --rm -p 8000:8000 parsely-ai-test
```

#### Code Quality Checks
```bash
# Format code
black src/ tests/ --line-length 88

# Sort imports
isort src/ tests/ --profile black

# Lint code
flake8 src/ tests/ --max-line-length=88

# Security scan
bandit -r src/
safety check
```

## 🔧 Configuration

### Environment Variables (.env)
```bash
# Required
GOOGLE_API_KEY=your_gemini_api_key_here

# Model Configuration
LLM_PROVIDER=google
LLM_MODEL=gemini-1.5-pro
LLM_TEMPERATURE=0.1

# Performance Settings
MAX_CONCURRENT_REQUESTS=10
CACHE_TTL_SECONDS=3600
ENABLE_QUERY_CACHING=true
```

### Model Options
- `gemini-1.5-pro` - Best performance (recommended)
- `gemini-1.5-flash` - Faster, more cost-effective
- `gemini-pro` - Standard model

## 🚨 Troubleshooting

### Common Issues
1. **API Key Error**: Run `python setup_api_key.py` to configure your key
2. **Import Errors**: Ensure all dependencies are installed: `pip install -r requirements.txt`
3. **No Policy Documents**: Add documents to `data/policies/` directory
4. **spaCy Model Missing**: Run `python -m spacy download en_core_web_sm`
5. **CI/CD Failures**: Check [CI/CD Setup Guide](docs/CI_CD_SETUP.md) for troubleshooting

### Getting Help
1. Run the comprehensive test: `python test_gemini_setup.py`
2. Check environment configuration: `python scripts/check_environment.py`
3. Review CI/CD pipeline status in GitHub Actions
4. Check the logs for detailed error messages
5. Verify your API key at [Google AI Studio](https://makersuite.google.com/app/apikey)

### CI/CD Pipeline Status
[![CI Pipeline](https://github.com/your-org/parsely-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/parsely-ai/actions/workflows/ci.yml)
[![Release Pipeline](https://github.com/your-org/parsely-ai/actions/workflows/release.yml/badge.svg)](https://github.com/your-org/parsely-ai/actions/workflows/release.yml)

## 🎯 Next Steps

After setup, you can:
1. **Start Implementation**: Open `.kiro/specs/llm-document-processing/tasks.md` and begin Task 1
2. **Add More Documents**: Expand beyond insurance to contracts, legal docs, etc.
3. **Customize Business Rules**: Modify decision logic for your specific domain
4. **Scale the System**: Deploy using Docker and Kubernetes configurations

## 🤝 Contributing
This system is designed to be extensible. Contributions are welcome for:
- New document format support
- Additional LLM providers
- Enhanced business rules
- UI improvements
- Performance optimizations

## 📄 License
This project is part of the LLM Document Processing System specification and implementation.
