# 🚀 eBook Editor Pro - Deployment Guide

This guide provides step-by-step instructions to deploy the eBook Editor Pro application successfully.

## 📋 Prerequisites

### System Requirements
- **Python 3.8+** (recommended: Python 3.11+)
- **Node.js 16+** (for frontend build)
- **npm or yarn** (package manager)
- **Git** (for cloning)
- **4GB+ RAM** (8GB+ recommended)
- **2GB+ disk space**

### Operating System
- Linux (Ubuntu 20.04+ recommended)
- macOS 10.15+
- Windows 10+ (with WSL2 recommended)

## 🏗️ Project Structure Setup

First, ensure your project structure looks like this:

```
ebook-editor-pro/
├── frontend/
│   ├── public/
│   │   └── index.html          # Create this
│   ├── src/
│   │   ├── components/
│   │   │   └── EBookEditor.jsx  # Already exists
│   │   ├── App.js              # Create this
│   │   ├── App.css             # Create this
│   │   ├── index.js            # Create this
│   │   └── index.css           # Create this
│   └── package.json            # Create this
├── services/                   # Already exists
├── api/                        # Already exists
├── main.py                     # Update this
├── config.py                   # Already exists
├── requirements.txt            # Already exists
└── .env                        # Already exists
```

## 🔧 Installation Steps

### Step 1: Clone and Setup Python Environment

```bash
# Navigate to your project directory
cd ebook-editor-pro

# Create Python virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install additional system dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install libxcb-cursor0
wget -nv -O- https://download.calibre-ebook.com/linux-installer.sh | sudo sh /dev/stdin
sudo apt install libmagic1 libmagic-dev
```

### Step 2: Setup Frontend

```bash
# Create frontend directory structure
mkdir -p frontend/public frontend/src/components

# Copy the provided files:
# - Copy package.json to frontend/
# - Copy public/index.html to frontend/public/
# - Copy src files to frontend/src/
# - Move EBookEditor.jsx to frontend/src/components/

# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Build the React application
npm run build

# Go back to root directory
cd ..
```

### Step 3: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit the .env file with your settings
nano .env  # or use your preferred editor
```

Key settings to configure in `.env`:
```bash
# Server Configuration
HOST=localhost
PORT=8000
DEBUG=true

# AI Configuration
USE_LOCAL_MODEL=true
LOCAL_MODEL_NAME=microsoft/DialoGPT-medium

# File Processing
MAX_FILE_SIZE=52428800
TEMP_DIR=./temp
UPLOAD_DIR=./uploads
EXPORT_DIR=./exports
```

### Step 4: Initialize Services

```bash
# Download required models and data
python -c "
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
"

# Create necessary directories
mkdir -p temp uploads exports models model_cache logs backups
```

## 🚀 Running the Application

### Development Mode

```bash
# Make sure you're in the project root and virtual environment is activated
source venv/bin/activate  # Linux/macOS
# or venv\Scripts\activate  # Windows

# Run the development server
python start_server.py --reload --dev

# Or directly with main.py
python main.py
```

The application will be available at:
- **Frontend**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Production Mode

```bash
# Build frontend for production
cd frontend
npm run build
cd ..

# Run production server
python start_server.py --host 0.0.0.0 --port 8000 --workers 4
```

## 🐳 Docker Deployment (Recommended for Production)

### Create Dockerfile

Create `Dockerfile` in the project root:

```dockerfile
# Multi-stage build for frontend
FROM node:18-alpine as frontend-builder

WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci --only=production

COPY frontend/ ./
RUN npm run build

# Python backend
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libxcb-cursor0 \
    libmagic1 \
    libmagic-dev \
    wget \
    && wget -nv -O- https://download.calibre-ebook.com/linux-installer.sh | sh /dev/stdin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy built frontend
COPY --from=frontend-builder /frontend/build ./frontend/build

# Create necessary directories
RUN mkdir -p temp uploads exports models model_cache logs

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('vader_lexicon')"

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "main.py"]
```

### Create docker-compose.yml

```yaml
version: '3.8'

services:
  ebook-editor:
    build: .
    ports:
      - "8000:8000"
    environment:
      - HOST=0.0.0.0
      - PORT=8000
      - DEBUG=false
    volumes:
      - ./uploads:/app/uploads
      - ./exports:/app/exports
      - ./models:/app/models
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Deploy with Docker

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 1. Frontend Not Loading
**Problem**: Accessing http://localhost:8000 shows API response instead of React app.

**Solution**:
```bash
# Check if frontend is built
ls frontend/build/

# If no build directory exists:
cd frontend
npm install
npm run build
cd ..

# Restart the server
python main.py
```

#### 2. AI Models Not Loading
**Problem**: AI features not working, model download errors.

**Solution**:
```bash
# Manually download models
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-medium')
"
```

#### 3. Port Already in Use
**Problem**: `Port 8000 is already in use`

**Solution**:
```bash
# Use different port
python start_server.py --port 8080

# Or kill existing process
sudo lsof -t -i tcp:8000 | xargs kill -9
```

#### 4. Memory Issues
**Problem**: Out of memory errors during AI processing.

**Solution**:
```bash
# Use smaller model
export LOCAL_MODEL_NAME=microsoft/DialoGPT-small

# Disable GPU if causing issues
export ENABLE_GPU=false
export MODEL_DEVICE=cpu
```

#### 5. File Upload Issues
**Problem**: Cannot upload files, processing fails.

**Solution**:
```bash
# Check permissions
chmod 755 temp uploads exports

# Check file size limits in .env
MAX_FILE_SIZE=52428800  # 50MB
```

## 📊 Monitoring and Maintenance

### Health Checks

```bash
# Check application health
curl http://localhost:8000/health

# Check detailed status
python start_server.py --status
```

### Log Management

```bash
# View application logs
tail -f logs/ebook_editor.log

# View Docker logs
docker-compose logs -f ebook-editor
```

### Performance Optimization

1. **Enable caching** in production:
   ```bash
   CACHE_ENABLED=true
   REDIS_URL=redis://localhost:6379
   ```

2. **Use multiple workers**:
   ```bash
   python start_server.py --workers 4
   ```

3. **Enable GPU acceleration** (if available):
   ```bash
   ENABLE_GPU=true
   MODEL_DEVICE=cuda
   ```

## 🔒 Security Configuration

### Production Security Settings

In your `.env` file for production:

```bash
# Security
SECRET_KEY=your-very-long-random-secret-key-change-this-immediately
DEBUG=false
ENVIRONMENT=production

# HTTPS (if using reverse proxy)
HTTPS_ONLY=true
SECURE_COOKIES=true

# Rate limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=60
```

### Reverse Proxy (Nginx)

Create `/etc/nginx/sites-available/ebook-editor`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # File upload size limit
    client_max_body_size 50M;
}
```

## 📈 Scaling and Production Deployment

### Load Balancing

For high-traffic deployments:

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  ebook-editor:
    build: .
    deploy:
      replicas: 3
    environment:
      - HOST=0.0.0.0
      - PORT=8000
      - WORKERS=2
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - ebook-editor
```

### Database Integration

For persistent storage:

```yaml
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: ebook_editor
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
```

## 📞 Support and Maintenance

### Regular Maintenance Tasks

1. **Clean up temporary files**:
   ```bash
   python -c "from services.file_handler import FileHandler; fh = FileHandler(settings); fh.cleanup_old_files(24)"
   ```

2. **Update models** (monthly):
   ```bash
   rm -rf model_cache/
   python start_server.py --download-models
   ```

3. **Database backup** (if using database):
   ```bash
   pg_dump ebook_editor > backup_$(date +%Y%m%d).sql
   ```

### Getting Help

- **Check logs**: Always check application logs first
- **Health endpoint**: Use `/health` for service status
- **Documentation**: API docs at `/docs`
- **Community**: Create GitHub issues for bugs/features

## ✅ Verification Checklist

After deployment, verify:

- [ ] Frontend loads at http://localhost:8000
- [ ] API documentation accessible at /docs
- [ ] Health check returns "healthy" status
- [ ] File upload works (test with small text file)
- [ ] Text analysis features work
- [ ] AI models load successfully (check logs)
- [ ] eBook generation works
- [ ] All required directories exist and have proper permissions

## 🎉 Success!

If all checks pass, your eBook Editor Pro is successfully deployed and ready for use!

For additional configuration and advanced features, refer to the main README.md file.