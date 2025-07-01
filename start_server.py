#!/usr/bin/env python3
"""
eBook Editor Pro Startup Script
Handles environment setup, dependency installation, and server startup.
"""

import os
import sys
import subprocess
import shutil
import platform
from pathlib import Path
import argparse
import time


class Colors:
    """ANSI color codes for terminal output."""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_banner():
    """Print application banner."""
    banner = f"""
{Colors.PURPLE}{Colors.BOLD}
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║               📚 eBook Editor Pro 📚                        ║
    ║                                                              ║
    ║           AI-Powered Professional eBook Creation             ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
{Colors.END}
    """
    print(banner)


def check_python_version():
    """Check if Python version is compatible."""
    print(f"{Colors.BLUE}🐍 Checking Python version...{Colors.END}")
    
    if sys.version_info < (3, 8):
        print(f"{Colors.RED}❌ Python 3.8+ is required. Current version: {sys.version}{Colors.END}")
        sys.exit(1)
    
    print(f"{Colors.GREEN}✅ Python {sys.version.split()[0]} is compatible{Colors.END}")


def check_dependencies():
    """Check if required system dependencies are installed."""
    print(f"{Colors.BLUE}🔧 Checking system dependencies...{Colors.END}")
    
    required_commands = ['git', 'pip']
    optional_commands = ['node', 'npm', 'docker']
    
    missing_required = []
    missing_optional = []
    
    for cmd in required_commands:
        if not shutil.which(cmd):
            missing_required.append(cmd)
    
    for cmd in optional_commands:
        if not shutil.which(cmd):
            missing_optional.append(cmd)
    
    if missing_required:
        print(f"{Colors.RED}❌ Missing required dependencies: {', '.join(missing_required)}{Colors.END}")
        print(f"{Colors.YELLOW}Please install the missing dependencies and try again.{Colors.END}")
        sys.exit(1)
    
    if missing_optional:
        print(f"{Colors.YELLOW}⚠️  Optional dependencies not found: {', '.join(missing_optional)}{Colors.END}")
        print(f"{Colors.YELLOW}   Some features may not be available.{Colors.END}")
    
    print(f"{Colors.GREEN}✅ System dependencies checked{Colors.END}")


def setup_environment():
    """Set up environment and directories."""
    print(f"{Colors.BLUE}📁 Setting up environment...{Colors.END}")
    
    # Create necessary directories
    directories = [
        'temp', 'uploads', 'exports', 'models', 'model_cache',
        'logs', 'backups', 'frontend/static'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"{Colors.GREEN}  ✓ Created directory: {directory}{Colors.END}")
    
    # Copy .env.example to .env if it doesn't exist
    if not Path('.env').exists():
        if Path('.env.example').exists():
            shutil.copy('.env.example', '.env')
            print(f"{Colors.GREEN}  ✓ Created .env from .env.example{Colors.END}")
            print(f"{Colors.YELLOW}  ⚠️  Please edit .env file to configure your settings{Colors.END}")
        else:
            print(f"{Colors.YELLOW}  ⚠️  .env.example not found, creating basic .env{Colors.END}")
            create_basic_env()


def create_basic_env():
    """Create a basic .env file with essential settings."""
    basic_env = """# Basic eBook Editor Pro Configuration
DEBUG=true
HOST=localhost
PORT=8000
USE_LOCAL_MODEL=true
LOCAL_MODEL_NAME=microsoft/DialoGPT-medium
"""
    
    with open('.env', 'w') as f:
        f.write(basic_env)
    
    print(f"{Colors.GREEN}  ✓ Created basic .env file{Colors.END}")


def install_python_dependencies(dev=False):
    """Install Python dependencies."""
    print(f"{Colors.BLUE}📦 Installing Python dependencies...{Colors.END}")
    
    try:
        # Upgrade pip first
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                      check=True, capture_output=True)
        
        # Install main dependencies
        if Path('requirements.txt').exists():
            cmd = [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt']
            subprocess.run(cmd, check=True)
            print(f"{Colors.GREEN}  ✓ Installed main dependencies{Colors.END}")
        
        # Install development dependencies if requested
        if dev and Path('requirements-dev.txt').exists():
            cmd = [sys.executable, '-m', 'pip', 'install', '-r', 'requirements-dev.txt']
            subprocess.run(cmd, check=True)
            print(f"{Colors.GREEN}  ✓ Installed development dependencies{Colors.END}")
        
        # Download spaCy model
        try:
            subprocess.run([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'], 
                          check=True, capture_output=True)
            print(f"{Colors.GREEN}  ✓ Downloaded spaCy English model{Colors.END}")
        except subprocess.CalledProcessError:
            print(f"{Colors.YELLOW}  ⚠️  Could not download spaCy model (optional){Colors.END}")
        
        # Download NLTK data
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            print(f"{Colors.GREEN}  ✓ Downloaded NLTK data{Colors.END}")
        except Exception:
            print(f"{Colors.YELLOW}  ⚠️  Could not download NLTK data (optional){Colors.END}")
            
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}❌ Failed to install dependencies: {e}{Colors.END}")
        sys.exit(1)


def download_models():
    """Download AI models if needed."""
    print(f"{Colors.BLUE}🤖 Checking AI models...{Colors.END}")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "microsoft/DialoGPT-medium"
        model_path = Path("models") / model_name.replace("/", "_")
        
        if not model_path.exists():
            print(f"{Colors.YELLOW}  📥 Downloading {model_name}...{Colors.END}")
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./model_cache")
            model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="./model_cache")
            print(f"{Colors.GREEN}  ✓ Downloaded {model_name}{Colors.END}")
        else:
            print(f"{Colors.GREEN}  ✓ Model {model_name} already available{Colors.END}")
            
    except Exception as e:
        print(f"{Colors.YELLOW}  ⚠️  Could not download models: {e}{Colors.END}")
        print(f"{Colors.YELLOW}     Models will be downloaded when first used{Colors.END}")


def check_ports(port=8000):
    """Check if port is available."""
    import socket
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True
        except OSError:
            return False


def start_server(port=8000, host='localhost', reload=False, workers=1):
    """Start the FastAPI server."""
    print(f"{Colors.BLUE}🚀 Starting eBook Editor Pro server...{Colors.END}")
    
    if not check_ports(port):
        print(f"{Colors.RED}❌ Port {port} is already in use{Colors.END}")
        print(f"{Colors.YELLOW}   Try using a different port with --port option{Colors.END}")
        sys.exit(1)
    
    # Import here to avoid import errors during setup
    try:
        import uvicorn
    except ImportError:
        print(f"{Colors.RED}❌ uvicorn not found. Installing...{Colors.END}")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'uvicorn[standard]'], check=True)
        import uvicorn
    
    print(f"{Colors.GREEN}✅ Server starting on http://{host}:{port}{Colors.END}")
    print(f"{Colors.CYAN}📚 Open your browser and navigate to the URL above{Colors.END}")
    print(f"{Colors.CYAN}📖 API Documentation available at http://{host}:{port}/docs{Colors.END}")
    print(f"{Colors.YELLOW}🛑 Press Ctrl+C to stop the server{Colors.END}")
    
    try:
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers if not reload else 1,
            access_log=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}🛑 Server stopped by user{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}❌ Server error: {e}{Colors.END}")
        sys.exit(1)


def run_tests():
    """Run the test suite."""
    print(f"{Colors.BLUE}🧪 Running tests...{Colors.END}")
    
    try:
        result = subprocess.run([sys.executable, '-m', 'pytest', '-v'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"{Colors.GREEN}✅ All tests passed{Colors.END}")
        else:
            print(f"{Colors.RED}❌ Some tests failed{Colors.END}")
            print(result.stdout)
            print(result.stderr)
            
    except FileNotFoundError:
        print(f"{Colors.YELLOW}⚠️  pytest not found, skipping tests{Colors.END}")


def show_status():
    """Show system status."""
    print(f"{Colors.BLUE}📊 System Status{Colors.END}")
    print(f"{Colors.WHITE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.END}")
    
    # Python version
    print(f"🐍 Python: {Colors.GREEN}{sys.version.split()[0]}{Colors.END}")
    
    # Platform
    print(f"💻 Platform: {Colors.GREEN}{platform.system()} {platform.release()}{Colors.END}")
    
    # Check directories
    directories = ['temp', 'uploads', 'exports', 'models']
    for directory in directories:
        status = "✅" if Path(directory).exists() else "❌"
        print(f"📁 {directory}: {status}")
    
    # Check configuration
    env_status = "✅" if Path('.env').exists() else "❌"
    print(f"⚙️  Configuration: {env_status}")
    
    # Check dependencies
    try:
        import fastapi, uvicorn, transformers
        deps_status = "✅"
    except ImportError:
        deps_status = "❌"
    print(f"📦 Dependencies: {deps_status}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="eBook Editor Pro Startup Script")
    parser.add_argument('--setup', action='store_true', help='Run full setup')
    parser.add_argument('--install-deps', action='store_true', help='Install dependencies only')
    parser.add_argument('--dev', action='store_true', help='Install development dependencies')
    parser.add_argument('--port', type=int, default=8000, help='Server port (default: 8000)')
    parser.add_argument('--host', default='localhost', help='Server host (default: localhost)')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
    parser.add_argument('--test', action='store_true', help='Run tests')
    parser.add_argument('--status', action='store_true', help='Show system status')
    parser.add_argument('--download-models', action='store_true', help='Download AI models')
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.status:
        show_status()
        return
    
    if args.test:
        run_tests()
        return
    
    if args.setup:
        check_python_version()
        check_dependencies()
        setup_environment()
        install_python_dependencies(dev=args.dev)
        download_models()
        print(f"\n{Colors.GREEN}🎉 Setup completed successfully!{Colors.END}")
        print(f"{Colors.CYAN}Run 'python start_server.py' to start the server{Colors.END}")
        return
    
    if args.install_deps:
        install_python_dependencies(dev=args.dev)
        return
    
    if args.download_models:
        download_models()
        return
    
    # Default: start server
    check_python_version()
    
    # Quick setup check
    if not Path('.env').exists():
        print(f"{Colors.YELLOW}⚠️  No .env file found. Running quick setup...{Colors.END}")
        setup_environment()
    
    start_server(
        port=args.port,
        host=args.host,
        reload=args.reload,
        workers=args.workers
    )


if __name__ == "__main__":
    main()
