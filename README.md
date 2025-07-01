# 📚 eBook Editor Pro

> **AI-Powered Professional eBook Creation Platform**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🌟 Overview

eBook Editor Pro is a comprehensive, AI-powered platform for creating professional eBooks. It combines the power of modern web technologies with advanced natural language processing and AI models to provide an intuitive, feature-rich writing and publishing experience.

### ✨ Key Features

#### 🤖 **AI-Powered Writing Assistance**
- **Real-time AI suggestions** for grammar, style, and clarity
- **Intelligent text enhancement** with customizable AI models
- **Advanced grammar checking** using LanguageTool
- **Style analysis and improvement** recommendations
- **Readability optimization** with multiple metrics

#### 📖 **Professional eBook Generation**
- **Multiple format support**: EPUB, PDF, DOCX, HTML, TXT
- **Professional typography** and layout options
- **Automatic chapter detection** and table of contents
- **Custom metadata management** (title, author, ISBN, etc.)
- **Cover design integration** and image handling

#### 💡 **Smart Text Processing**
- **Advanced NLP analysis** using spaCy and NLTK
- **Sentiment analysis** and keyword extraction
- **Writing analytics** and progress tracking
- **Chapter structure analysis** and organization
- **Multi-language support** with localization

#### 🎨 **Rich Editor Experience**
- **WYSIWYG editor** with full formatting support
- **Real-time collaboration** features (optional)
- **Auto-save functionality** with version control
- **Dark/light theme** with customizable interface
- **Drag-and-drop file imports** (DOCX, PDF, EPUB, TXT)

#### 📊 **Analytics and Insights**
- **Writing progress tracking** with daily goals
- **Readability score monitoring** and improvement suggestions
- **Word count and time tracking** with session analytics
- **Chapter-by-chapter analysis** and statistics
- **Export analytics** and performance metrics

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (recommended: Python 3.11+)
- **Git** for cloning the repository
- **4GB+ RAM** (8GB+ recommended for AI features)
- **2GB+ disk space** for models and dependencies

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ebook-editor-pro.git
cd ebook-editor-pro

# Run the automated setup
python start_server.py --setup

# Or manual setup:
pip install -r requirements.txt
cp .env.example .env
# Edit .env file with your configuration
```

### 2. Start the Server

```bash
# Quick start (development mode)
python start_server.py

# Production mode
python start_server.py --host 0.0.0.0 --port 8000 --workers 4

# Development mode with auto-reload
python start_server.py --reload --dev
```

### 3. Access the Application

- **Main Application**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## 🏗️ Architecture

### Backend (Python/FastAPI)

```
ebook_editor/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py              # Configuration management
│   ├── models/                # Pydantic data models
│   ├── services/              # Business logic services
│   │   ├── ai_integration.py  # AI model integration
│   │   ├── text_processing.py # NLP and text analysis
│   │   ├── ebook_generator.py # eBook creation engine
│   │   └── file_handler.py    # File upload/processing
│   ├── api/                   # API route handlers
│   └── utils/                 # Utility functions
├── frontend/                  # Enhanced React frontend
├── models/                    # Local AI model storage
├── temp/                      # Temporary file processing
├── exports/                   # Generated eBook files
└── requirements.txt           # Python dependencies
```

### Frontend (React/JavaScript)

The frontend is built with modern React and includes:
- **Rich text editor** with advanced formatting
- **Real-time AI suggestions** with highlighting
- **Project management** interface
- **Analytics dashboard** with visualizations
- **Export wizard** with format options
- **Settings panel** for AI configuration

### AI Integration

The platform supports multiple AI backends:

#### Local Models (Hugging Face Transformers)
- **Text Generation**: DialoGPT, BART, T5
- **Grammar Checking**: RoBERTa-based models
- **Sentence Embeddings**: all-MiniLM-L6-v2
- **Custom model loading** and hot-swapping

#### External APIs
- **OpenAI GPT models** (GPT-3.5, GPT-4)
- **Anthropic Claude** integration
- **Local LM Studio** or similar endpoints
- **Custom API endpoints** with OpenAI-compatible format

## 📚 Usage Guide

### Basic Workflow

1. **Create New Project**
   - Set book metadata (title, author, genre)
   - Configure AI preferences
   - Set writing goals and preferences

2. **Write and Edit**
   - Use the rich text editor with formatting
   - Get real-time AI suggestions
   - Track progress with analytics

3. **Import Existing Content**
   - Drag and drop DOCX, PDF, or EPUB files
   - Automatic text extraction and formatting
   - AI-enhanced content processing

4. **Analyze and Improve**
   - Review grammar and style suggestions
   - Check readability scores
   - Optimize chapter structure

5. **Export and Publish**
   - Choose output format (EPUB, PDF, DOCX)
   - Apply professional formatting
   - Generate with AI enhancements

### Advanced Features

#### AI Configuration

```python
# Local model setup
USE_LOCAL_MODEL=true
LOCAL_MODEL_NAME=microsoft/DialoGPT-medium

# External API setup
EXTERNAL_AI_ENABLED=true
EXTERNAL_AI_BASE_URL=https://api.openai.com/v1
EXTERNAL_AI_API_KEY=your-api-key
```

#### Custom AI Models

```python
# Add custom models in config.py
AI_MODEL_CONFIGS = {
    "custom-model": {
        "model": "your-org/your-model",
        "max_length": 2048,
        "temperature": 0.7
    }
}
```

#### API Integration

```python
# Use the REST API
import requests

# Analyze text
response = requests.post('http://localhost:8000/api/analyze', json={
    'text': 'Your text here',
    'include_ai_suggestions': True
})

# Generate eBook
response = requests.post('http://localhost:8000/api/generate-ebook', json={
    'content': 'Book content',
    'format': 'epub',
    'metadata': {'title': 'My Book', 'author': 'Author Name'}
})
```

## ⚙️ Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# AI Models
USE_LOCAL_MODEL=true
LOCAL_MODEL_NAME=microsoft/DialoGPT-medium
EXTERNAL_AI_BASE_URL=http://localhost:1234/v1

# Performance
MAX_WORKERS=4
ENABLE_GPU=false
MODEL_DEVICE=auto

# Features
ENABLE_GRAMMAR_CHECK=true
ENABLE_STYLE_CHECK=true
REAL_TIME_ANALYSIS=true

# File Processing
MAX_FILE_SIZE=52428800
SUPPORTED_UPLOAD_TYPES=text/plain,application/pdf,application/epub+zip
```

### Advanced Configuration

#### Model Performance Tuning

```python
# Adjust model parameters for your hardware
ENABLE_GPU=true  # Requires CUDA
MODEL_DEVICE=cuda
MAX_SEQUENCE_LENGTH=2048
BATCH_SIZE=4
```

#### Custom NLP Pipeline

```python
# Configure text processing components
SPACY_MODEL=en_core_web_lg  # For better accuracy
ENABLE_SENTIMENT_ANALYSIS=true
ENABLE_KEYWORD_EXTRACTION=true
```

## 🔧 API Reference

### Core Endpoints

#### Text Analysis
```http
POST /api/analyze
Content-Type: application/json

{
  "text": "Text to analyze",
  "include_ai_suggestions": true,
  "suggestion_categories": ["grammar", "style", "clarity"]
}
```

#### File Upload
```http
POST /api/upload
Content-Type: multipart/form-data

file: [file]
enhance_with_ai: true
target_format: "epub"
```

#### eBook Generation
```http
POST /api/generate-ebook
Content-Type: application/json

{
  "content": "Book content",
  "format": "epub",
  "metadata": {
    "title": "Book Title",
    "author": "Author Name",
    "description": "Book description"
  },
  "format_options": {
    "font_family": "Georgia",
    "font_size": 12,
    "include_toc": true
  }
}
```

#### AI Configuration
```http
POST /api/configure-ai
Content-Type: application/json

{
  "ai_type": "local",
  "model_name": "microsoft/DialoGPT-medium",
  "api_endpoint": "http://localhost:1234/v1",
  "api_key": "optional-api-key"
}
```

### Response Formats

All API responses follow this structure:
```json
{
  "status": "success",
  "timestamp": "2025-01-01T00:00:00Z",
  "data": { /* response data */ },
  "message": "Optional message"
}
```

Error responses:
```json
{
  "status": "error",
  "error": "Error description",
  "status_code": 400,
  "timestamp": "2025-01-01T00:00:00Z"
}
```

## 🛠️ Development

### Setup Development Environment

```bash
# Install development dependencies
python start_server.py --setup --dev

# Install pre-commit hooks
pre-commit install

# Run tests
python start_server.py --test

# Start development server
python start_server.py --reload --dev
```

### Project Structure

```
ebook-editor-pro/
├── app/                    # Python backend
│   ├── main.py            # FastAPI app
│   ├── config.py          # Configuration
│   ├── services/          # Business logic
│   ├── api/               # API routes
│   └── models/            # Data models
├── frontend/              # React frontend
│   ├── components/        # React components
│   ├── hooks/             # Custom hooks
│   ├── utils/             # Utilities
│   └── styles/            # CSS/styling
├── tests/                 # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── fixtures/          # Test data
├── docs/                  # Documentation
├── scripts/               # Utility scripts
└── requirements/          # Dependency files
    ├── base.txt           # Core dependencies
    ├── dev.txt            # Development dependencies
    └── prod.txt           # Production dependencies
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/api/           # API tests
```

### Code Quality

The project uses several tools for code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pre-commit**: Git hooks

```bash
# Format code
black app/ tests/
isort app/ tests/

# Lint code
flake8 app/ tests/

# Type check
mypy app/
```

## 📈 Performance Optimization

### AI Model Optimization

1. **Model Selection**
   - Use smaller models for development
   - Larger models for production quality
   - Quantized models for memory efficiency

2. **Caching Strategies**
   - Model result caching
   - Preprocessing caching
   - Response caching with Redis

3. **Batch Processing**
   - Batch API requests
   - Background task processing
   - Queue management with Celery

### Database Optimization

```python
# Enable database optimization
DATABASE_URL=postgresql://user:pass@localhost/ebook_editor
DB_POOL_SIZE=20
DB_ECHO=false
ENABLE_DB_CACHE=true
```

### Memory Management

```python
# Configure memory limits
MAX_MEMORY_USAGE=4096  # MB
ENABLE_MEMORY_PROFILING=true
AUTO_CLEANUP_TEMP_FILES=true
```

## 🐳 Docker Deployment

### Docker Compose Setup

```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/ebook_editor
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    depends_on:
      - db
      - redis

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: ebook_editor
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

```bash
# Deploy with Docker
docker-compose up -d

# Scale workers
docker-compose up -d --scale app=3

# View logs
docker-compose logs -f app
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ebook-editor-pro
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ebook-editor-pro
  template:
    metadata:
      labels:
        app: ebook-editor-pro
    spec:
      containers:
      - name: app
        image: ebook-editor-pro:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

## 🔐 Security

### Authentication & Authorization

```python
# Enable authentication
ENABLE_AUTH=true
JWT_SECRET_KEY=your-secret-key
ACCESS_TOKEN_EXPIRE_MINUTES=30

# OAuth providers
GOOGLE_CLIENT_ID=your-google-client-id
GITHUB_CLIENT_ID=your-github-client-id
```

### Data Protection

```python
# Encryption settings
ENCRYPT_UPLOADED_FILES=true
ENCRYPTION_KEY=your-encryption-key
SECURE_COOKIES=true
HTTPS_ONLY=true
```

### Rate Limiting

```python
# API rate limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=60
RATE_LIMIT_BURST=10
```

## 📊 Monitoring

### Health Checks

```bash
# Application health
curl http://localhost:8000/health

# Detailed status
python start_server.py --status
```

### Metrics Collection

```python
# Enable metrics
ENABLE_METRICS=true
METRICS_PORT=9090
PROMETHEUS_ENDPOINT=/metrics
```

### Logging Configuration

```python
# Structured logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=logs/ebook_editor.log
ENABLE_REQUEST_LOGGING=true
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Process

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Make changes** and add tests
4. **Run quality checks**: `pre-commit run --all-files`
5. **Commit changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Create Pull Request**

### Code Standards

- Follow PEP 8 for Python code
- Use type hints where possible
- Write comprehensive tests
- Document all public APIs
- Update README for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Getting Help

- **Documentation**: Check this README and inline documentation
- **Issues**: Create a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact us at support@ebook-editor-pro.com

### Common Issues

#### Model Download Failures
```bash
# Manually download models
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')"
```

#### Memory Issues
```bash
# Reduce model size
export LOCAL_MODEL_NAME=microsoft/DialoGPT-small
export ENABLE_GPU=false
```

#### Port Conflicts
```bash
# Use different port
python start_server.py --port 8080
```

### Performance Troubleshooting

1. **Slow AI responses**: Use smaller models or external APIs
2. **High memory usage**: Enable model caching and cleanup
3. **File upload issues**: Check MAX_FILE_SIZE setting
4. **Database performance**: Enable connection pooling

## 🗺️ Roadmap

### Version 2.0 (Coming Soon)
- [ ] **Collaborative editing** with real-time sync
- [ ] **Advanced AI models** (GPT-4, Claude-3)
- [ ] **Plugin system** for extensibility
- [ ] **Mobile app** for iOS and Android
- [ ] **Cloud storage** integration
- [ ] **Advanced analytics** dashboard

### Version 2.1
- [ ] **Multi-language support** for content
- [ ] **Voice dictation** integration
- [ ] **AI-powered cover design**
- [ ] **Publishing marketplace** integration
- [ ] **Version control** system
- [ ] **Team collaboration** features

### Long-term Vision
- [ ] **AI writing assistant** with creative modes
- [ ] **Automated fact-checking** and research
- [ ] **Publishing workflow** automation
- [ ] **Reader analytics** and feedback
- [ ] **Monetization tools** for authors
- [ ] **Educational features** for writing improvement

## 👥 Team

- **Development**: AI/ML Engineers, Full-stack Developers
- **Design**: UX/UI Designers, Content Designers  
- **Research**: NLP Researchers, Linguists
- **QA**: Quality Assurance Engineers, Beta Testers

## 🙏 Acknowledgments

- **Hugging Face** for transformer models and libraries
- **FastAPI** team for the excellent web framework
- **spaCy** and **NLTK** for NLP capabilities
- **React** community for frontend components
- **Open source contributors** who make this possible

---

<div align="center">

**Made with ❤️ by the eBook Editor Pro Team**

[Website](https://ebook-editor-pro.com) • [Documentation](https://docs.ebook-editor-pro.com) • [API Reference](https://api.ebook-editor-pro.com) • [Community](https://community.ebook-editor-pro.com)

</div>
