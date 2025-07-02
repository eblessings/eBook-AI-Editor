#!/usr/bin/env python3
"""
Enhanced eBook Editor Pro Startup Script with comprehensive CLI options.
Handles environment setup, dependency installation, and server startup with advanced configuration.
"""

import os
import sys
import subprocess
import shutil
import platform
from pathlib import Path
import argparse
import time
import json
import signal
import threading
import psutil


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
    ║                    Enhanced CLI Version                      ║
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
    optional_commands = ['node', 'npm', 'docker', 'nvidia-smi']
    
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


def detect_device():
    """Auto-detect best available device (CPU/CUDA)."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            print(f"{Colors.GREEN}🎮 CUDA available: {gpu_count} GPU(s) - {gpu_name}{Colors.END}")
            return "cuda"
        else:
            print(f"{Colors.YELLOW}💻 CUDA not available, using CPU{Colors.END}")
            return "cpu"
    except ImportError:
        print(f"{Colors.YELLOW}💻 PyTorch not installed, defaulting to CPU{Colors.END}")
        return "cpu"


def setup_environment():
    """Set up environment and directories."""
    print(f"{Colors.BLUE}📁 Setting up environment...{Colors.END}")
    
    # Create necessary directories
    directories = [
        'temp', 'uploads', 'exports', 'models', 'model_cache',
        'logs', 'backups', 'frontend/static', 'data/projects',
        'data/cache', 'data/sessions'
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
MODEL_DEVICE=auto
ENABLE_GPU=false
MAX_WORKERS=1
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:3000"]
"""
    
    with open('.env', 'w') as f:
        f.write(basic_env)
    
    print(f"{Colors.GREEN}  ✓ Created basic .env file{Colors.END}")


def install_python_dependencies(dev=False, force=False):
    """Install Python dependencies."""
    print(f"{Colors.BLUE}📦 Installing Python dependencies...{Colors.END}")
    
    try:
        # Upgrade pip first
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                      check=True, capture_output=True)
        
        # Install main dependencies
        if Path('requirements.txt').exists():
            cmd = [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt']
            if force:
                cmd.append('--force-reinstall')
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
            nltk.download('omw-1.4', quiet=True)
            print(f"{Colors.GREEN}  ✓ Downloaded NLTK data{Colors.END}")
        except Exception:
            print(f"{Colors.YELLOW}  ⚠️  Could not download NLTK data (optional){Colors.END}")
            
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}❌ Failed to install dependencies: {e}{Colors.END}")
        sys.exit(1)


def download_models(model_name=None):
    """Download AI models if needed."""
    print(f"{Colors.BLUE}🤖 Checking AI models...{Colors.END}")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        models_to_download = [model_name] if model_name else [
            "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-small"
        ]
        
        for model in models_to_download:
            model_path = Path("models") / model.replace("/", "_")
            
            if not model_path.exists():
                print(f"{Colors.YELLOW}  📥 Downloading {model}...{Colors.END}")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir="./model_cache")
                    model_obj = AutoModelForCausalLM.from_pretrained(model, cache_dir="./model_cache")
                    print(f"{Colors.GREEN}  ✓ Downloaded {model}{Colors.END}")
                except Exception as e:
                    print(f"{Colors.YELLOW}  ⚠️  Failed to download {model}: {e}{Colors.END}")
            else:
                print(f"{Colors.GREEN}  ✓ Model {model} already available{Colors.END}")
                
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


def kill_process_on_port(port):
    """Kill process running on specified port."""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                for conn in proc.info['connections'] or []:
                    if conn.laddr.port == port:
                        print(f"{Colors.YELLOW}🔄 Killing process {proc.info['pid']} on port {port}{Colors.END}")
                        proc.kill()
                        time.sleep(1)
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception as e:
        print(f"{Colors.YELLOW}⚠️  Could not kill process on port {port}: {e}{Colors.END}")
    return False


def update_env_file(updates):
    """Update .env file with new values."""
    env_file = Path('.env')
    if not env_file.exists():
        create_basic_env()
    
    # Read current env
    env_content = env_file.read_text()
    
    # Update values
    for key, value in updates.items():
        pattern = f"{key}=.*"
        replacement = f"{key}={value}"
        
        if f"{key}=" in env_content:
            import re
            env_content = re.sub(pattern, replacement, env_content)
        else:
            env_content += f"\n{replacement}\n"
    
    # Write back
    env_file.write_text(env_content)
    print(f"{Colors.GREEN}✅ Updated .env file{Colors.END}")


def start_server(args):
    """Start the FastAPI server with comprehensive configuration."""
    print(f"{Colors.BLUE}🚀 Starting eBook Editor Pro server...{Colors.END}")
    
    # Update .env file with CLI arguments
    env_updates = {}
    
    if args.device:
        env_updates['MODEL_DEVICE'] = args.device
        env_updates['ENABLE_GPU'] = 'true' if args.device == 'cuda' else 'false'
    
    if args.model:
        env_updates['LOCAL_MODEL_NAME'] = args.model
    
    if args.api_endpoint:
        env_updates['EXTERNAL_AI_BASE_URL'] = args.api_endpoint
        env_updates['USE_LOCAL_MODEL'] = 'false'
        env_updates['EXTERNAL_AI_ENABLED'] = 'true'
    
    if args.api_key:
        env_updates['EXTERNAL_AI_API_KEY'] = args.api_key
    
    env_updates['HOST'] = args.host
    env_updates['PORT'] = str(args.port)
    env_updates['DEBUG'] = 'true' if args.debug else 'false'
    env_updates['MAX_WORKERS'] = str(args.workers)
    
    if env_updates:
        update_env_file(env_updates)
    
    # Check and handle port conflicts
    if not check_ports(args.port):
        if args.kill_port:
            kill_process_on_port(args.port)
            time.sleep(2)
            if not check_ports(args.port):
                print(f"{Colors.RED}❌ Port {args.port} is still in use{Colors.END}")
                sys.exit(1)
        else:
            print(f"{Colors.RED}❌ Port {args.port} is already in use{Colors.END}")
            print(f"{Colors.YELLOW}   Use --kill-port to automatically kill the process{Colors.END}")
            sys.exit(1)
    
    # Import here to avoid import errors during setup
    try:
        import uvicorn
    except ImportError:
        print(f"{Colors.RED}❌ uvicorn not found. Installing...{Colors.END}")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'uvicorn[standard]'], check=True)
        import uvicorn
    
    print(f"{Colors.GREEN}✅ Server starting on http://{args.host}:{args.port}{Colors.END}")
    print(f"{Colors.CYAN}📚 Open your browser and navigate to the URL above{Colors.END}")
    print(f"{Colors.CYAN}📖 API Documentation available at http://{args.host}:{args.port}/docs{Colors.END}")
    print(f"{Colors.YELLOW}🛑 Press Ctrl+C to stop the server{Colors.END}")
    
    # Configure uvicorn settings
    uvicorn_config = {
        "app": "main:app",
        "host": args.host,
        "port": args.port,
        "reload": args.reload,
        "workers": 1 if args.reload else args.workers,  # Force single worker with reload
        "access_log": True,
        "log_level": "debug" if args.debug else "info"
    }
    
    if args.ssl_cert and args.ssl_key:
        uvicorn_config.update({
            "ssl_certfile": args.ssl_cert,
            "ssl_keyfile": args.ssl_key
        })
    
    try:
        uvicorn.run(**uvicorn_config)
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
    """Show comprehensive system status."""
    print(f"{Colors.BLUE}📊 System Status{Colors.END}")
    print(f"{Colors.WHITE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.END}")
    
    # Python version
    print(f"🐍 Python: {Colors.GREEN}{sys.version.split()[0]}{Colors.END}")
    
    # Platform
    print(f"💻 Platform: {Colors.GREEN}{platform.system()} {platform.release()}{Colors.END}")
    
    # Memory usage
    memory = psutil.virtual_memory()
    print(f"🧠 Memory: {Colors.GREEN}{memory.percent}% used ({memory.used//1024//1024}MB/{memory.total//1024//1024}MB){Colors.END}")
    
    # Disk space
    disk = psutil.disk_usage('.')
    print(f"💾 Disk: {Colors.GREEN}{disk.percent}% used ({disk.used//1024//1024//1024}GB/{disk.total//1024//1024//1024}GB){Colors.END}")
    
    # Device detection
    device = detect_device()
    print(f"🎮 Compute Device: {Colors.GREEN}{device.upper()}{Colors.END}")
    
    # Check directories
    directories = ['temp', 'uploads', 'exports', 'models', 'frontend/build']
    for directory in directories:
        status = "✅" if Path(directory).exists() else "❌"
        print(f"📁 {directory}: {status}")
    
    # Check configuration
    env_status = "✅" if Path('.env').exists() else "❌"
    print(f"⚙️  Configuration: {env_status}")
    
    # Check dependencies
    deps_status = "❌"
    try:
        import fastapi, uvicorn, transformers
        deps_status = "✅"
    except ImportError:
        pass
    print(f"📦 Dependencies: {deps_status}")
    
    # Check models
    model_dir = Path('models')
    model_count = len(list(model_dir.glob('*'))) if model_dir.exists() else 0
    print(f"🤖 AI Models: {Colors.GREEN}{model_count} available{Colors.END}")


def benchmark_performance():
    """Run performance benchmarks."""
    print(f"{Colors.BLUE}⚡ Running performance benchmarks...{Colors.END}")
    
    try:
        import time
        import numpy as np
        
        # CPU benchmark
        start_time = time.time()
        np.random.rand(1000, 1000).dot(np.random.rand(1000, 1000))
        cpu_time = time.time() - start_time
        print(f"💻 CPU Matrix Multiplication (1000x1000): {Colors.GREEN}{cpu_time:.3f}s{Colors.END}")
        
        # Memory benchmark
        start_time = time.time()
        large_array = np.random.rand(10000, 1000)
        del large_array
        memory_time = time.time() - start_time
        print(f"🧠 Memory Allocation/Deallocation: {Colors.GREEN}{memory_time:.3f}s{Colors.END}")
        
        # GPU benchmark (if available)
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.device('cuda')
                start_time = time.time()
                a = torch.rand(1000, 1000, device=device)
                b = torch.rand(1000, 1000, device=device)
                c = torch.mm(a, b)
                torch.cuda.synchronize()
                gpu_time = time.time() - start_time
                print(f"🎮 GPU Matrix Multiplication (1000x1000): {Colors.GREEN}{gpu_time:.3f}s{Colors.END}")
        except:
            pass
            
    except ImportError:
        print(f"{Colors.YELLOW}⚠️  NumPy not available for benchmarks{Colors.END}")


def create_parser():
    """Create comprehensive argument parser."""
    parser = argparse.ArgumentParser(
        description="eBook Editor Pro - AI-Powered Professional eBook Creation Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_server.py                                    # Start with default settings
  python start_server.py --setup --dev                     # Full setup with dev dependencies
  python start_server.py --device cuda --workers 4        # Use GPU with 4 workers
  python start_server.py --model microsoft/DialoGPT-large  # Use specific model
  python start_server.py --api-endpoint http://localhost:1234/v1 --api-key sk-xxx  # External API
  python start_server.py --port 8080 --kill-port          # Use port 8080, kill if occupied
  python start_server.py --ssl-cert cert.pem --ssl-key key.pem  # HTTPS mode
        """
    )
    
    # Setup and maintenance commands
    setup_group = parser.add_argument_group('Setup and Maintenance')
    setup_group.add_argument('--setup', action='store_true', help='Run full setup')
    setup_group.add_argument('--install-deps', action='store_true', help='Install dependencies only')
    setup_group.add_argument('--dev', action='store_true', help='Install development dependencies')
    setup_group.add_argument('--force-install', action='store_true', help='Force reinstall dependencies')
    setup_group.add_argument('--download-models', action='store_true', help='Download AI models')
    setup_group.add_argument('--clean', action='store_true', help='Clean temporary files and cache')
    
    # Server configuration
    server_group = parser.add_argument_group('Server Configuration')
    server_group.add_argument('--host', default='localhost', help='Server host (default: localhost)')
    server_group.add_argument('--port', type=int, default=8000, help='Server port (default: 8000)')
    server_group.add_argument('--workers', type=int, default=1, help='Number of worker processes (default: 1)')
    server_group.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    server_group.add_argument('--debug', action='store_true', help='Enable debug mode')
    server_group.add_argument('--kill-port', action='store_true', help='Kill process on port if occupied')
    
    # AI and model configuration
    ai_group = parser.add_argument_group('AI Configuration')
    ai_group.add_argument('--device', choices=['cpu', 'cuda', 'auto'], help='Compute device (cpu/cuda/auto)')
    ai_group.add_argument('--model', help='AI model name (e.g., microsoft/DialoGPT-medium)')
    ai_group.add_argument('--api-endpoint', help='External AI API endpoint')
    ai_group.add_argument('--api-key', help='External AI API key')
    ai_group.add_argument('--disable-ai', action='store_true', help='Disable AI features')
    ai_group.add_argument('--model-cache-dir', default='./model_cache', help='Model cache directory')
    
    # SSL/Security
    security_group = parser.add_argument_group('Security')
    security_group.add_argument('--ssl-cert', help='SSL certificate file path')
    security_group.add_argument('--ssl-key', help='SSL private key file path')
    security_group.add_argument('--cors-origins', help='CORS allowed origins (comma-separated)')
    
    # Utilities
    util_group = parser.add_argument_group('Utilities')
    util_group.add_argument('--test', action='store_true', help='Run tests')
    util_group.add_argument('--status', action='store_true', help='Show system status')
    util_group.add_argument('--benchmark', action='store_true', help='Run performance benchmarks')
    util_group.add_argument('--version', action='version', version='eBook Editor Pro 1.0.0')
    
    return parser


def clean_temp_files():
    """Clean temporary files and cache."""
    print(f"{Colors.BLUE}🧹 Cleaning temporary files...{Colors.END}")
    
    dirs_to_clean = ['temp', 'logs', 'model_cache/__pycache__', '.pytest_cache']
    files_cleaned = 0
    
    for dir_path in dirs_to_clean:
        path = Path(dir_path)
        if path.exists():
            if path.is_dir():
                for file in path.rglob('*'):
                    if file.is_file() and file.suffix in ['.tmp', '.log', '.pyc', '.cache']:
                        try:
                            file.unlink()
                            files_cleaned += 1
                        except Exception:
                            pass
    
    print(f"{Colors.GREEN}✅ Cleaned {files_cleaned} temporary files{Colors.END}")


def main():
    """Main entry point with enhanced CLI handling."""
    parser = create_parser()
    args = parser.parse_args()
    
    print_banner()
    
    # Handle utility commands first
    if args.status:
        show_status()
        return
    
    if args.test:
        run_tests()
        return
    
    if args.benchmark:
        benchmark_performance()
        return
    
    if args.clean:
        clean_temp_files()
        return
    
    # Auto-detect device if not specified
    if args.device == 'auto' or args.device is None:
        args.device = detect_device()
    
    # Handle setup commands
    if args.setup:
        check_python_version()
        check_dependencies()
        setup_environment()
        install_python_dependencies(dev=args.dev, force=args.force_install)
        download_models()
        print(f"\n{Colors.GREEN}🎉 Setup completed successfully!{Colors.END}")
        print(f"{Colors.CYAN}Run 'python start_server.py' to start the server{Colors.END}")
        return
    
    if args.install_deps:
        install_python_dependencies(dev=args.dev, force=args.force_install)
        return
    
    if args.download_models:
        download_models(args.model)
        return
    
    # Default: start server
    check_python_version()
    
    # Quick setup check
    if not Path('.env').exists():
        print(f"{Colors.YELLOW}⚠️  No .env file found. Running quick setup...{Colors.END}")
        setup_environment()
    
    # Ensure single worker if using AI features to avoid crashes
    if args.workers > 1 and not args.disable_ai:
        print(f"{Colors.YELLOW}⚠️  Forcing single worker when AI is enabled to prevent crashes{Colors.END}")
        args.workers = 1
    
    start_server(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}🛑 Operation cancelled by user{Colors.END}")
        sys.exit(0)
    except Exception as e:
        print(f"{Colors.RED}❌ Unexpected error: {e}{Colors.END}")
        sys.exit(1)