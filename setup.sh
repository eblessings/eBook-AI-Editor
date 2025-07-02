#!/bin/bash

# eBook Editor Pro - Quick Setup Script
# This script automates the entire setup process

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo -e "${PURPLE}${WHITE}"
    echo "============================================================"
    echo "📚 eBook Editor Pro - Quick Setup Script"
    echo "============================================================"
    echo -e "${NC}"
}

print_step() {
    echo -e "${BLUE}[Step $1]${NC} ${CYAN}$2${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${WHITE}ℹ️  $1${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
check_prerequisites() {
    print_step 1 "Checking prerequisites..."
    
    # Check Python
    if ! command_exists python3; then
        print_error "Python 3 is required but not installed."
        echo "Please install Python 3.8+ from https://python.org"
        exit 1
    fi
    
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_success "Python $python_version ✓"
    
    # Check Node.js
    if ! command_exists node; then
        print_error "Node.js is required but not installed."
        echo "Please install Node.js 16+ from https://nodejs.org"
        exit 1
    fi
    
    node_version=$(node --version)
    print_success "Node.js $node_version ✓"
    
    # Check npm
    if ! command_exists npm; then
        print_error "npm is required but not installed."
        exit 1
    fi
    
    npm_version=$(npm --version)
    print_success "npm $npm_version ✓"
    
    # Check pip
    if ! command_exists pip3; then
        print_warning "pip3 not found, trying pip..."
        if ! command_exists pip; then
            print_error "pip is required but not installed."
            exit 1
        fi
        PIP_CMD="pip"
    else
        PIP_CMD="pip3"
    fi
    print_success "pip ✓"
}

# Create virtual environment
setup_python_env() {
    print_step 2 "Setting up Python environment..."
    
    if [ ! -d "venv" ]; then
        print_info "Creating virtual environment..."
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_success "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    print_success "Virtual environment activated"
    
    # Upgrade pip
    pip install --upgrade pip > /dev/null 2>&1
    print_success "pip upgraded"
}

# Install Python dependencies
install_python_deps() {
    print_step 3 "Installing Python dependencies..."
    
    if [ -f "requirements.txt" ]; then
        print_info "This may take a few minutes..."
        pip install -r requirements.txt > /dev/null 2>&1
        print_success "Python dependencies installed"
    else
        print_warning "requirements.txt not found, installing essential packages..."
        pip install fastapi uvicorn > /dev/null 2>&1
        print_success "Essential packages installed"
    fi
}

# Setup frontend
setup_frontend() {
    print_step 4 "Setting up frontend..."
    
    # Run the Python build script to create frontend files
    print_info "Creating frontend structure..."
    python3 build.py > /dev/null 2>&1 || {
        print_error "Failed to run build script. Trying manual setup..."
        
        # Create directories
        mkdir -p frontend/{public,src/components}
        
        # Create basic package.json
        cat > frontend/package.json << 'EOF'
{
  "name": "ebook-editor-pro-frontend",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "lucide-react": "^0.263.1",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build"
  },
  "proxy": "http://localhost:8000"
}
EOF
        
        # Create basic index.html
        cat > frontend/public/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>eBook Editor Pro</title>
    <script src="https://cdn.tailwindcss.com"></script>
  </head>
  <body>
    <div id="root"></div>
  </body>
</html>
EOF
    }
    
    print_success "Frontend structure created"
}

# Install frontend dependencies
install_frontend_deps() {
    print_step 5 "Installing frontend dependencies..."
    
    cd frontend
    print_info "This may take a few minutes..."
    npm install > /dev/null 2>&1
    print_success "Frontend dependencies installed"
    cd ..
}

# Build frontend
build_frontend() {
    print_step 6 "Building frontend..."
    
    cd frontend
    print_info "Building React application..."
    npm run build > /dev/null 2>&1
    print_success "Frontend built successfully"
    cd ..
}

# Create directories
create_directories() {
    print_step 7 "Creating necessary directories..."
    
    directories=("temp" "uploads" "exports" "models" "model_cache" "logs")
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        print_success "Created directory: $dir"
    done
}

# Setup environment
setup_environment() {
    print_step 8 "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_success "Created .env from .env.example"
        else
            cat > .env << 'EOF'
# eBook Editor Pro Configuration
DEBUG=true
HOST=localhost
PORT=8000
USE_LOCAL_MODEL=true
LOCAL_MODEL_NAME=microsoft/DialoGPT-medium
TEMP_DIR=./temp
UPLOAD_DIR=./uploads
EXPORT_DIR=./exports
EOF
            print_success "Created basic .env file"
        fi
    else
        print_success ".env file already exists"
    fi
}

# Download NLTK data
download_nltk_data() {
    print_step 9 "Downloading NLTK data..."
    
    python3 -c "
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

datasets = ['punkt', 'stopwords', 'wordnet', 'vader_lexicon']
for dataset in datasets:
    try:
        nltk.download(dataset, quiet=True)
        print(f'Downloaded {dataset}')
    except:
        pass
" > /dev/null 2>&1
    
    print_success "NLTK data downloaded"
}

# Verify installation
verify_installation() {
    print_step 10 "Verifying installation..."
    
    checks=0
    total=4
    
    # Check frontend build
    if [ -f "frontend/build/index.html" ]; then
        print_success "Frontend build ✓"
        ((checks++))
    else
        print_error "Frontend build ✗"
    fi
    
    # Check directories
    if [ -d "temp" ] && [ -d "uploads" ] && [ -d "exports" ]; then
        print_success "Required directories ✓"
        ((checks++))
    else
        print_error "Required directories ✗"
    fi
    
    # Check .env file
    if [ -f ".env" ]; then
        print_success "Environment configuration ✓"
        ((checks++))
    else
        print_error "Environment configuration ✗"
    fi
    
    # Check Python packages
    if python3 -c "import fastapi, uvicorn" > /dev/null 2>&1; then
        print_success "Python dependencies ✓"
        ((checks++))
    else
        print_error "Python dependencies ✗"
    fi
    
    echo ""
    if [ $checks -eq $total ]; then
        echo -e "${GREEN}${WHITE}🎉 SETUP COMPLETED SUCCESSFULLY! 🎉${NC}"
        echo ""
        echo -e "${CYAN}Next steps:${NC}"
        echo -e "1. ${WHITE}source venv/bin/activate${NC} - Activate virtual environment"
        echo -e "2. ${WHITE}python main.py${NC} - Start the server"
        echo -e "3. Open ${BLUE}http://localhost:8000${NC} in your browser"
        echo ""
        echo -e "${YELLOW}For development mode:${NC}"
        echo -e "${WHITE}python start_server.py --reload --dev${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️ Setup completed with $((total - checks)) issues.${NC}"
        echo "Please check the verification results above."
        return 1
    fi
}

# Create start script
create_start_script() {
    cat > start.sh << 'EOF'
#!/bin/bash
# Quick start script for eBook Editor Pro

echo "🚀 Starting eBook Editor Pro..."

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️ Virtual environment not found. Run setup.sh first."
    exit 1
fi

# Start the server
python main.py
EOF
    
    chmod +x start.sh
    print_success "Created start.sh script"
}

# Main function
main() {
    print_header
    
    # Check if running from correct directory
    if [ ! -f "main.py" ] && [ ! -f "start_server.py" ]; then
        print_error "Please run this script from the eBook Editor Pro root directory."
        exit 1
    fi
    
    # Run setup steps
    check_prerequisites
    setup_python_env
    install_python_deps
    setup_frontend
    install_frontend_deps
    build_frontend
    create_directories
    setup_environment
    download_nltk_data
    create_start_script
    verify_installation
    
    echo ""
    echo -e "${GREEN}Setup script completed!${NC}"
}

# Handle interruption
trap 'echo -e "\n${YELLOW}Setup interrupted by user.${NC}"; exit 1' INT

# Run main function
main "$@"