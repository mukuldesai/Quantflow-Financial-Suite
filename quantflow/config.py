# quantflow/config.py
"""
QuantFlow Financial Suite - Configuration Management
Handles YAML configs, environment variables, and global settings
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

class QuantFlowConfig:
    """Central configuration management for QuantFlow Financial Suite"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.root_path = Path(__file__).parent.parent
        self.config_path = config_path or self.root_path / "config" / "assumptions.yaml"
        self._alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self._config = self._load_config()
        self._setup_logging()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            print(f"⚠️  Config file not found at {self.config_path}")
            return self._get_default_config()
        except yaml.YAMLError as e:
            print(f"❌ Error parsing YAML config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if YAML file is missing"""
        return {
            "market": {
                "risk_free_rate": 0.043,
                "market_risk_premium": 0.065,
                "default_beta": 1.15
            },
            "dcf": {
                "projection_years": 5,
                "terminal_growth_rate": 0.04,
                "tax_rate": 0.21
            },
            "paths": {
                "raw_data": "data/raw",
                "processed_data": "data/processed",
                "manual_data": "data/manual"
            }
        }
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_config = self._config.get("logging", {})
        log_level = log_config.get("level", "INFO")
        log_format = log_config.get("format", "%(asctime)s - %(levelname)s - %(message)s")
        
        # Create logs directory if it doesn't exist
        log_dir = self.root_path / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format,
            handlers=[
                logging.FileHandler(log_dir / "quantflow.log"),
                logging.StreamHandler()
            ]
        )
    
    # Market Data Properties
    @property
    def risk_free_rate(self) -> float:
        """Get risk-free rate from config or environment"""
        return float(os.getenv("RISK_FREE_RATE", 
                              self._config["market"]["risk_free_rate"]))
    
    @property
    def market_risk_premium(self) -> float:
        """Get market risk premium"""
        return self._config["market"]["market_risk_premium"]
    
    @property
    def default_beta(self) -> float:
        """Get default beta for companies"""
        return self._config["market"]["default_beta"]
    
    # DCF Model Properties
    @property
    def projection_years(self) -> int:
        """Number of years to project in DCF"""
        return self._config["dcf"]["projection_years"]
    
    @property
    def terminal_growth_rate(self) -> float:
        """Terminal growth rate for DCF"""
        return self._config["dcf"]["terminal_growth_rate"]
    
    @property
    def tax_rate(self) -> float:
        """Default corporate tax rate"""
        return self._config["dcf"]["tax_rate"]
    
    # Path Properties
    @property
    def raw_data_path(self) -> Path:
        """Path to raw data directory"""
        return self.root_path / self._config["paths"]["raw_data"]
    
    @property
    def processed_data_path(self) -> Path:
        """Path to processed data directory"""
        return self.root_path / self._config["paths"]["processed_data"]
    
    @property
    def manual_data_path(self) -> Path:
        """Path to manual data directory"""
        return self.root_path / self._config["paths"]["manual_data"]
    
    @property
    def templates_path(self) -> Path:
        """Path to Excel templates directory"""
        return self.root_path / "templates"
    
    # Company-Specific Methods
    def get_company_config(self, ticker: str) -> Dict[str, Any]:
        """Get company-specific configuration"""
        companies = self._config.get("companies", {})
        return companies.get(ticker.upper(), {})
    
    def get_company_growth_rates(self, ticker: str) -> Dict[str, float]:
        """Get revenue growth rates for specific company"""
        company_config = self.get_company_config(ticker)
        return company_config.get("revenue_growth_rates", 
                                 self._config["dcf"]["revenue_growth_rates"])
    
    def get_company_margins(self, ticker: str) -> Dict[str, float]:
        """Get margin assumptions for specific company"""
        company_config = self.get_company_config(ticker)
        return company_config.get("margins", 
                                 self._config["dcf"]["margins"])
    
    def get_company_beta(self, ticker: str) -> float:
        """Get beta for specific company"""
        company_config = self.get_company_config(ticker)
        return company_config.get("beta", self.default_beta)
    
    # API Configuration
    def get_api_config(self, provider: str) -> Dict[str, Any]:
        """Get API configuration for data providers"""
        data_sources = self._config.get("data_sources", {})
        return data_sources.get(provider, {})
    
    # File Naming
    def get_filename(self, ticker: str, file_type: str) -> str:
        """Get standardized filename for data files"""
        naming = self._config.get("naming", {})
        template = naming.get(file_type, f"{ticker}_{file_type}.csv")
        return template.format(ticker=ticker.upper())
    
    @property
    def base_path(self) -> Path:
        """Get the base project path (alias for root_path)"""
        return self.root_path

# Also add this property if you don't have it:
    @property  
    def output_path(self) -> Path:
        """Get the output directory path"""
        output_dir = self.root_path / "outputs"
        output_dir.mkdir(exist_ok=True)
        return output_dir
    
    # Environment Variables for API Keys
    @property
    def alpha_vantage_api_key(self) -> Optional[str]:
        """Get Alpha Vantage API key from environment"""
        return os.getenv("ALPHA_VANTAGE_API_KEY")
    
    @property
    def iex_api_key(self) -> Optional[str]:
        """Get IEX Cloud API key from environment"""
        return os.getenv("IEX_API_KEY")
    
    @property
    def quandl_api_key(self) -> Optional[str]:
        """Get Quandl API key from environment"""
        return os.getenv("QUANDL_API_KEY")
    
    # Utility Methods
    def create_directories(self):
        """Create all necessary directories"""
        directories = [
            self.raw_data_path,
            self.processed_data_path, 
            self.manual_data_path,
            self.templates_path,
            self.root_path / "logs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def update_config(self, key_path: str, value: Any):
        """Update configuration value (key_path like 'dcf.tax_rate')"""
        keys = key_path.split('.')
        config_section = self._config
        
        # Navigate to the correct section
        for key in keys[:-1]:
            if key not in config_section:
                config_section[key] = {}
            config_section = config_section[key]
        
        # Set the value
        config_section[keys[-1]] = value
    
    def save_config(self):
        """Save current configuration back to YAML file"""
        try:
            with open(self.config_path, 'w') as file:
                yaml.dump(self._config, file, default_flow_style=False, indent=2)
            print(f"Configuration saved to {self.config_path}")
        except Exception as e:
            print(f"Error saving configuration: {e}")

# Global configuration instance
config = QuantFlowConfig()

# Convenience functions
def get_config() -> QuantFlowConfig:
    """Get the global configuration instance"""
    return config

def setup_project():
    """Setup project directories and initial configuration"""
    config.create_directories()
    print("QuantFlow Financial Suite setup complete!")
    print(f"Root directory: {config.root_path}")
    print(f"Data directory: {config.raw_data_path}")
    print(f"Config file: {config.config_path}")

if __name__ == "__main__":
    setup_project()