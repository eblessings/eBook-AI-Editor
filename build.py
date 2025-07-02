#!/usr/bin/env python3
"""
Automated build script for eBook Editor Pro.
This script sets up the frontend and ensures everything is ready for deployment.
"""

import os
import sys
import subprocess
import shutil
import json
from pathlib import Path


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


def print_step(step, message):
    """Print a colored step message."""
    print(f"{Colors.BLUE}{Colors.BOLD}[Step {step}]{Colors.END} {Colors.CYAN}{message}{Colors.END}")


def print_success(message):
    """Print a success message."""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")


def print_error(message):
    """Print an error message."""
    print(f"{Colors.RED}❌ {message}{Colors.END}")


def print_warning(message):
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")


def print_info(message):
    """Print an info message."""
    print(f"{Colors.WHITE}ℹ️  {message}{Colors.END}")


def run_command(command, cwd=None, check=True):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            cwd=cwd, 
            check=check,
            capture_output=True,
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {command}")
        print_error(f"Error: {e.stderr}")
        return None


def check_prerequisites():
    """Check if all prerequisites are installed."""
    print_step(1, "Checking prerequisites...")
    
    # Check Python
    python_version = sys.version_info
    if python_version < (3, 8):
        print_error(f"Python 3.8+ required. Current version: {python_version.major}.{python_version.minor}")
        return False
    print_success(f"Python {python_version.major}.{python_version.minor} ✓")
    
    # Check Node.js
    node_result = run_command("node --version", check=False)
    if not node_result or node_result.returncode != 0:
        print_error("Node.js not found. Please install Node.js 16+ from https://nodejs.org/")
        return False
    
    node_version = node_result.stdout.strip()
    print_success(f"Node.js {node_version} ✓")
    
    # Check npm
    npm_result = run_command("npm --version", check=False)
    if not npm_result or npm_result.returncode != 0:
        print_error("npm not found. Please install npm.")
        return False
    
    npm_version = npm_result.stdout.strip()
    print_success(f"npm {npm_version} ✓")
    
    return True


def create_directories():
    """Create necessary directories."""
    print_step(2, "Creating directories...")
    
    directories = [
        "frontend/public",
        "frontend/src/components",
        "temp",
        "uploads", 
        "exports",
        "models",
        "model_cache",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print_success(f"Created directory: {directory}")


def create_frontend_files():
    """Create frontend configuration and source files."""
    print_step(3, "Creating frontend files...")
    
    # Create package.json
    package_json = {
        "name": "ebook-editor-pro-frontend",
        "version": "1.0.0",
        "private": True,
        "dependencies": {
            "@testing-library/jest-dom": "^5.16.4",
            "@testing-library/react": "^13.3.0",
            "@testing-library/user-event": "^13.5.0",
            "lucide-react": "^0.263.1",
            "react": "^18.2.0",
            "react-dom": "^18.2.0",
            "react-scripts": "5.0.1",
            "web-vitals": "^2.1.4"
        },
        "scripts": {
            "start": "react-scripts start",
            "build": "react-scripts build",
            "test": "react-scripts test",
            "eject": "react-scripts eject"
        },
        "eslintConfig": {
            "extends": ["react-app", "react-app/jest"]
        },
        "browserslist": {
            "production": [">0.2%", "not dead", "not op_mini all"],
            "development": ["last 1 chrome version", "last 1 firefox version", "last 1 safari version"]
        },
        "proxy": "http://localhost:8000"
    }
    
    with open("frontend/package.json", "w") as f:
        json.dump(package_json, f, indent=2)
    print_success("Created package.json")
    
    # Create public/index.html
    index_html = '''<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <link rel="icon" href="%PUBLIC_URL%/favicon.ico" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta name="description" content="AI-Powered Professional eBook Creation Platform" />
    <title>eBook Editor Pro</title>
    <script src="https://cdn.tailwindcss.com"></script>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>'''
    
    with open("frontend/public/index.html", "w") as f:
        f.write(index_html)
    print_success("Created public/index.html")
    
    # Create src/index.js
    index_js = '''import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);'''
    
    with open("frontend/src/index.js", "w") as f:
        f.write(index_js)
    print_success("Created src/index.js")
    
    # Create src/App.js
    app_js = '''import React from 'react';
import EBookEditor from './components/EBookEditor';
import './App.css';

function App() {
  return (
    <div className="App">
      <EBookEditor />
    </div>
  );
}

export default App;'''
    
    with open("frontend/src/App.js", "w") as f:
        f.write(app_js)
    print_success("Created src/App.js")
    
    # Create src/App.css
    app_css = '''.App {
  min-height: 100vh;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}'''
    
    with open("frontend/src/App.css", "w") as f:
        f.write(app_css)
    print_success("Created src/App.css")
    
    # Create src/index.css
    index_css = '''@import 'tailwindcss/base';
@import 'tailwindcss/components';
@import 'tailwindcss/utilities';

body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  background-color: #f9fafb;
}'''
    
    with open("frontend/src/index.css", "w") as f:
        f.write(index_css)
    print_success("Created src/index.css")


def move_existing_component():
    """Move existing EBookEditor component to correct location."""
    print_step(4, "Moving existing components...")
    
    # Check if EBookEditor.jsx exists in the current location
    current_location = "frontend/src/components/EBookEditor.jsx"
    
    if os.path.exists(current_location):
        print_success("EBookEditor.jsx already in correct location")
    else:
        # Try to find it in other locations
        possible_locations = [
            "EBookEditor.jsx",
            "frontend/EBookEditor.jsx",
            "components/EBookEditor.jsx"
        ]
        
        found = False
        for location in possible_locations:
            if os.path.exists(location):
                shutil.copy2(location, current_location)
                print_success(f"Moved EBookEditor.jsx from {location} to {current_location}")
                found = True
                break
        
        if not found:
            print_warning("EBookEditor.jsx not found. You may need to copy it manually.")


def install_frontend_dependencies():
    """Install frontend dependencies."""
    print_step(5, "Installing frontend dependencies...")
    
    print_info("This may take a few minutes...")
    result = run_command("npm install", cwd="frontend")
    
    if result and result.returncode == 0:
        print_success("Frontend dependencies installed successfully")
        return True
    else:
        print_error("Failed to install frontend dependencies")
        return False


def build_frontend():
    """Build the frontend for production."""
    print_step(6, "Building frontend...")
    
    print_info("Building React application for production...")
    result = run_command("npm run build", cwd="frontend")
    
    if result and result.returncode == 0:
        print_success("Frontend built successfully")
        
        # Check if build directory was created
        if os.path.exists("frontend/build"):
            print_success("Build directory created: frontend/build/")
            return True
        else:
            print_error("Build directory not found")
            return False
    else:
        print_error("Frontend build failed")
        return False


def install_python_dependencies():
    """Install Python dependencies."""
    print_step(7, "Installing Python dependencies...")
    
    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if not in_venv:
        print_warning("Not in a virtual environment. Consider creating one:")
        print_info("python -m venv venv")
        print_info("source venv/bin/activate  # Linux/macOS")
        print_info("venv\\Scripts\\activate  # Windows")
    
    if os.path.exists("requirements.txt"):
        result = run_command(f"{sys.executable} -m pip install -r requirements.txt")
        if result and result.returncode == 0:
            print_success("Python dependencies installed successfully")
            return True
        else:
            print_error("Failed to install Python dependencies")
            return False
    else:
        print_warning("requirements.txt not found")
        return False


def setup_environment():
    """Setup environment configuration."""
    print_step(8, "Setting up environment...")
    
    if not os.path.exists(".env"):
        if os.path.exists(".env.example"):
            shutil.copy2(".env.example", ".env")
            print_success("Created .env from .env.example")
        else:
            # Create basic .env
            basic_env = """# eBook Editor Pro Configuration
DEBUG=true
HOST=localhost
PORT=8000
USE_LOCAL_MODEL=true
LOCAL_MODEL_NAME=microsoft/DialoGPT-medium
TEMP_DIR=./temp
UPLOAD_DIR=./uploads
EXPORT_DIR=./exports
"""
            with open(".env", "w") as f:
                f.write(basic_env)
            print_success("Created basic .env file")
    else:
        print_success(".env file already exists")


def download_nltk_data():
    """Download required NLTK data."""
    print_step(9, "Downloading NLTK data...")
    
    nltk_script = '''
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

datasets = ['punkt', 'stopwords', 'wordnet', 'vader_lexicon', 'omw-1.4']
for dataset in datasets:
    try:
        nltk.download(dataset, quiet=True)
        print(f"✅ Downloaded {dataset}")
    except Exception as e:
        print(f"⚠️ Failed to download {dataset}: {e}")
'''
    
    result = run_command(f'{sys.executable} -c "{nltk_script}"')
    if result:
        print_success("NLTK data download completed")


def verify_installation():
    """Verify that everything is set up correctly."""
    print_step(10, "Verifying installation...")
    
    checks = []
    
    # Check frontend build
    if os.path.exists("frontend/build/index.html"):
        checks.append(("Frontend build", True))
    else:
        checks.append(("Frontend build", False))
    
    # Check Python dependencies
    try:
        import fastapi, uvicorn, transformers
        checks.append(("Python dependencies", True))
    except ImportError:
        checks.append(("Python dependencies", False))
    
    # Check directories
    required_dirs = ["temp", "uploads", "exports", "models"]
    dirs_exist = all(os.path.exists(d) for d in required_dirs)
    checks.append(("Required directories", dirs_exist))
    
    # Check configuration
    checks.append((".env file", os.path.exists(".env")))
    
    # Print results
    print(f"\n{Colors.BOLD}Installation Verification:{Colors.END}")
    all_good = True
    for check_name, status in checks:
        if status:
            print(f"{Colors.GREEN}✅ {check_name}{Colors.END}")
        else:
            print(f"{Colors.RED}❌ {check_name}{Colors.END}")
            all_good = False
    
    return all_good


def main():
    """Main build function."""
    print(f"{Colors.PURPLE}{Colors.BOLD}")
    print("=" * 60)
    print("📚 eBook Editor Pro - Automated Build Script")
    print("=" * 60)
    print(f"{Colors.END}")
    
    try:
        # Run all build steps
        if not check_prerequisites():
            print_error("Prerequisites check failed. Please install missing requirements.")
            sys.exit(1)
        
        create_directories()
        create_frontend_files()
        move_existing_component()
        
        if not install_frontend_dependencies():
            print_error("Frontend dependency installation failed.")
            sys.exit(1)
        
        if not build_frontend():
            print_error("Frontend build failed.")
            sys.exit(1)
        
        if not install_python_dependencies():
            print_error("Python dependency installation failed.")
            sys.exit(1)
        
        setup_environment()
        download_nltk_data()
        
        if verify_installation():
            print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 BUILD SUCCESSFUL! 🎉{Colors.END}")
            print(f"\n{Colors.CYAN}Next steps:{Colors.END}")
            print(f"1. {Colors.WHITE}python main.py{Colors.END} - Start the server")
            print(f"2. Open {Colors.BLUE}http://localhost:8000{Colors.END} in your browser")
            print(f"3. API docs at {Colors.BLUE}http://localhost:8000/docs{Colors.END}")
            print(f"\n{Colors.YELLOW}For development with auto-reload:{Colors.END}")
            print(f"{Colors.WHITE}python start_server.py --reload --dev{Colors.END}")
        else:
            print(f"\n{Colors.YELLOW}⚠️ Build completed with issues. Check the verification results above.{Colors.END}")
    
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Build interrupted by user.{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Build failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()