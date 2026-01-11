#/storage/emulated/0/Vxt/Vxt/vera_xt/core/config_manager.py
#!/usr/bin/env python3
"""
Configuration Manager - Handles environment variables and settings
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigManager:
    def __init__(self, env_file: str = ".env"):
        self.env_file = Path(env_file)
        self.config = {}
        
        # Load environment variables from .env file
        self.load_env_file()
        
        # Set default configuration
        self.set_defaults()
        
        print("⚙️  Configuration Manager initialized")
        print(f"📋 Using config from: {self.env_file}")
    
    def load_env_file(self):
        """Load environment variables from .env file"""
        if self.env_file.exists():
            with open(self.env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"\'')  # Remove quotes
                        os.environ[key] = value
                        self.config[key] = value
            print("✅ Environment variables loaded from .env file")
        else:
            print("⚠️  .env file not found, using system environment")
    
    def set_defaults(self):
        """Set default configuration values"""
        defaults = {
            'OPENROUTER_API_KEY': '',
            'DEFAULT_MODEL': 'tinyllama-1.1b-chat-v1.0-q4_k_m.gguf',
            'SERVER_HOST': '127.0.0.1',
            'SERVER_PORT': '8080',
            'MEMORY_DIR': 'Memory_Data',
            'WORKSPACE_DIR': 'Workspace',
            'MAX_TOKENS': '512',
            'TEMPERATURE': '0.7',
            'TOP_P': '0.9'
        }
        
        for key, default_value in defaults.items():
            if key not in os.environ:
                os.environ[key] = default_value
                self.config[key] = default_value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return os.environ.get(key, default)
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer configuration value"""
        try:
            return int(os.environ.get(key, default))
        except ValueError:
            return default
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float configuration value"""
        try:
            return float(os.environ.get(key, default))
        except ValueError:
            return default
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean configuration value"""
        value = os.environ.get(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
    
    def update_config(self, key: str, value: Any):
        """Update configuration value"""
        os.environ[key] = str(value)
        self.config[key] = str(value)
    
    def save_to_env_file(self):
        """Save current configuration to .env file"""
        with open(self.env_file, 'w', encoding='utf-8') as f:
            f.write("# Vera_XT Configuration File\n")
            f.write("# Generated automatically\n\n")
            
            for key, value in self.config.items():
                f.write(f"{key}={value}\n")
        
        print(f"💾 Configuration saved to {self.env_file}")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            'api_key_set': bool(self.get('OPENROUTER_API_KEY')),
            'default_model': self.get('DEFAULT_MODEL'),
            'server_config': {
                'host': self.get('SERVER_HOST'),
                'port': self.get_int('SERVER_PORT')
            },
            'memory_config': {
                'memory_dir': self.get('MEMORY_DIR'),
                'workspace_dir': self.get('WORKSPACE_DIR')
            },
            'performance_config': {
                'max_tokens': self.get_int('MAX_TOKENS'),
                'temperature': self.get_float('TEMPERATURE'),
                'top_p': self.get_float('TOP_P')
            }
        }

# Global config instance
config_manager = ConfigManager()
