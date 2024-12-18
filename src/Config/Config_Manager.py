import yaml
import os
from datetime import datetime
from typing import Dict, Any, Optional

class ConfigManager:
    def __init__(self, config_path: str = 'src/Config/configBT_1.yaml', create_output_dir: bool = True):
        """
        Initialize ConfigManager with a specified config file path
        
        Args:
            config_path: Path to the YAML configuration file
            create_output_dir: Whether to create an output directory automatically
        """
        self.config_path = config_path
        self.load_config()
        
        # Create timestamp for output directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup output directory if requested
        if create_output_dir and 'output_settings' in self.config:
            self.setup_directories()
    
    def load_config(self) -> None:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {str(e)}")
    
    def setup_directories(self) -> None:
        """Create necessary directories for output"""
        if 'output_settings' not in self.config:
            raise KeyError("Output settings not found in configuration")
            
        base_path = self.config['output_settings'].get('base_path')
        model_type = self.config['output_settings'].get('model_type', 'model_')
        
        if not base_path:
            raise KeyError("Base path not specified in output settings")
            
        self.output_dir = os.path.join(base_path, f"{model_type}{self.timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        self.save_config_copy()
    
    def save_config_copy(self) -> None:
        """Save a copy of the configuration in the output directory"""
        config_copy_path = os.path.join(self.output_dir, 'config_used.yaml')
        with open(config_copy_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)
    
    def get_config_value(self, *keys: str, default: Any = None) -> Any:
        """
        Get a value from nested configuration using a sequence of keys
        
        Args:
            *keys: Sequence of keys to access nested configuration
            default: Default value if the key path doesn't exist
        """
        result = self.config
        for key in keys:
            if isinstance(result, dict) and key in result:
                result = result[key]
            else:
                return default
        return result
    
    def get_paths(self) -> Dict[str, str]:
        """Get input and output paths"""
        paths = self.get_config_value('data_paths', default={})
        paths['output_dir'] = getattr(self, 'output_dir', None)
        return paths
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific model
        
        Args:
            model_name: Name of the model to get configuration for
        """
        return self.get_config_value('models', model_name, default={})
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values
        
        Args:
            updates: Dictionary containing configuration updates
        """
        self.config.update(updates)
        if hasattr(self, 'output_dir'):
            self.save_config_copy()

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access to configuration"""
        return self.config[key]

    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration"""
        return key in self.config
