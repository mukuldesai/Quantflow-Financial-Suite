#!/usr/bin/env python3
"""
QuantFlow Financial Suite - Complete Setup Script
Run this script to set up your entire QuantFlow project structure
"""

import os
import sys
from pathlib import Path
import subprocess

def create_directory_structure(root_path):
    """Create the complete QuantFlow directory structure"""
    
    print("🏗️  Creating QuantFlow Financial Suite directory structure...")
    
    # Define all directories
    directories = [
        "data/raw",
        "data/processed", 
        "data/manual",
        "quantflow/fetchers",
        "quantflow/models",
        "quantflow/analyzers", 
        "quantflow/utils",
        "notebooks",
        "templates",
        "dashboard",
        "config",
        "tests",
        "logs",
        "outputs"
    ]
    
    # Create directories
    for directory in directories:
        full_path = Path(root_path) / directory
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    return True

def create_init_files(root_path):
    """Create __init__.py files for Python package structure"""
    
    print("\n📄 Creating Python package files...")
    
    init_files = [
        "quantflow/__init__.py",
        "quantflow/fetchers/__init__.py", 
        "quantflow/models/__init__.py",
        "quantflow/analyzers/__init__.py",
        "quantflow/utils/__init__.py",
        "tests/__init__.py"
    ]
    
    for init_file in init_files:
        init_path = Path(root_path) / init_file
        if not init_path.exists():
            init_path.touch()
            print(f"📄 Created: {init_file}")

def create_gitignore(root_path):
    """Create .gitignore file for the project"""
    
    gitignore_content = """# QuantFlow Financial Suite - .gitignore

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter Notebooks
.ipynb_checkpoints/
*.ipynb

# Data files (keep structure, ignore data)
data/raw/*.csv
data/raw/*.xlsx
data/processed/*.csv
data/processed/*.xlsx
data/manual/*.csv
data/manual/*.xlsx

# Logs
logs/*.log
*.log

# API Keys and Environment
.env
.env.local
.env.production
secrets.yaml

# OS
.DS_Store
Thumbs.db

# Excel temporary files
~$*.xlsx
~$*.xls

# Outputs
outputs/*.pdf
outputs/*.xlsx
outputs/*.png

# Cache
.cache/
.pytest_cache/

# Coverage
htmlcov/
.coverage
coverage.xml
"""
    
    gitignore_path = Path(root_path) / ".gitignore"
    with open(gitignore_path, 'w') as f:
        f.write(gitignore_content)
    print("📄 Created: .gitignore")

def create_env_template(root_path):
    """Create .env.template file"""
    
    env_template = """# QuantFlow Financial Suite - Environment Variables Template
# Copy this file to .env and fill in your API keys

# Financial Data API Keys
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
IEX_API_KEY=your_iex_cloud_key_here
QUANDL_API_KEY=your_quandl_key_here
FRED_API_KEY=your_fred_key_here

# Model Parameters (optional - will use config/assumptions.yaml if not set)
RISK_FREE_RATE=0.043
DEFAULT_TAX_RATE=0.21
MARKET_RISK_PREMIUM=0.065

# Dashboard Settings
STREAMLIT_SERVER_PORT=8501
DASHBOARD_REFRESH_INTERVAL=3600

# Database (optional)
DATABASE_URL=sqlite:///quantflow.db

# Development Settings
DEBUG=True
LOG_LEVEL=INFO
"""
    
    env_path = Path(root_path) / ".env.template"
    with open(env_path, 'w') as f:
        f.write(env_template)
    print("📄 Created: .env.template")

def install_dependencies(root_path):
    """Install Python dependencies"""
    
    print("\n📦 Installing Python dependencies...")
    
    requirements_path = Path(root_path) / "requirements.txt"
    
    if requirements_path.exists():
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_path)
            ])
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Warning: Could not install dependencies: {e}")
            print("You can install them manually with: pip install -r requirements.txt")
    else:
        print("⚠️  requirements.txt not found. Please create it first.")

def create_sample_notebook(root_path):
    """Create a sample Jupyter notebook"""
    
    notebook_content = """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# QuantFlow Financial Suite - Getting Started\\n",
    "\\n",
    "Welcome to QuantFlow! This notebook will help you get started with the financial modeling suite."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import QuantFlow\\n",
    "import sys\\n",
    "sys.path.append('..')\\n",
    "\\n",
    "import quantflow\\n",
    "from quantflow.config import get_config\\n",
    "\\n",
    "# Display welcome message\\n",
    "quantflow.welcome()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load configuration\\n",
    "config = get_config()\\n",
    "quantflow.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Next Steps\\n",
    "\\n",
    "1. Set up your API keys in `.env` file\\n",
    "2. Configure assumptions in `config/assumptions.yaml`\\n",
    "3. Start fetching financial data\\n",
    "4. Build your first DCF model"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}"""
    
    notebook_path = Path(root_path) / "notebooks" / "00_getting_started.ipynb"
    with open(notebook_path, 'w') as f:
        f.write(notebook_content)
    print("📄 Created: notebooks/00_getting_started.ipynb")

def main():
    """Main setup function"""
    
    print("🚀 QuantFlow Financial Suite Setup")
    print("=" * 50)
    
    # Get root path
    if len(sys.argv) > 1:
        root_path = Path(sys.argv[1])
    else:
        root_path = Path("E:/Study/Personal Projects/QuantFlowFinancialSuite")
    
    print(f"📁 Setting up project at: {root_path}")
    
    # Create root directory if it doesn't exist
    root_path.mkdir(parents=True, exist_ok=True)
    
    # Change to project directory
    os.chdir(root_path)
    
    try:
        # Step 1: Create directory structure
        create_directory_structure(root_path)
        
        # Step 2: Create __init__.py files
        create_init_files(root_path)
        
        # Step 3: Create .gitignore
        create_gitignore(root_path)
        
        # Step 4: Create .env.template
        create_env_template(root_path)
        
        # Step 5: Create sample notebook
        create_sample_notebook(root_path)
        
        # Step 6: Install dependencies (optional)
        install_deps = input("\n📦 Install Python dependencies now? (y/N): ").lower().strip()
        if install_deps in ['y', 'yes']:
            install_dependencies(root_path)
        
        print("\n🎉 QuantFlow Financial Suite setup complete!")
        print("\n📋 Next Steps:")
        print("1. Copy .env.template to .env and add your API keys")
        print("2. Review config/assumptions.yaml for model parameters") 
        print("3. Run: jupyter notebook notebooks/00_getting_started.ipynb")
        print("4. Start building your financial models!")
        print(f"\n📁 Project location: {root_path}")
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        print("Please check the error and try again.")
        return False
    
    return True

if __name__ == "__main__":
    main()