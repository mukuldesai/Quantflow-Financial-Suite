"""
QuantFlow Financial Suite

AI-Augmented Valuation & Dashboarding for Public Companies
A comprehensive financial modeling platform for DCF analysis and company valuation.
"""

__version__ = "1.0.0"
__author__ = "QuantFlow Team"
__description__ = "AI-Augmented Valuation & Dashboarding for Public Companies"

# Core imports for easy access
from .config import get_config, setup_project

# Version info
VERSION_INFO = {
    "version": __version__,
    "description": __description__,
    "python_requires": ">=3.9",
    "author": __author__
}

def get_version():
    """Return the current version of QuantFlow Financial Suite"""
    return __version__

def welcome():
    """Display welcome message and version info"""
    print("🚀 Welcome to QuantFlow Financial Suite!")
    print(f"📊 Version: {__version__}")
    print(f"💼 {__description__}")
    print("=" * 50)

def info():
    """Display system information and configuration"""
    config = get_config()
    print(f"Root Path: {config.root_path}")
    print(f"Risk-Free Rate: {config.risk_free_rate:.2%}")
    print(f"Terminal Growth: {config.terminal_growth_rate:.1%}")
    print(f"Projection Years: {config.projection_years}")

# Make key classes available at package level
__all__ = [
    "get_config",
    "setup_project", 
    "get_version",
    "welcome",
    "info",
    "VERSION_INFO"
]