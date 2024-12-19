# Configuration Management

## Overview
The project uses a YAML-based configuration system through the `ConfigManager` class, providing centralized parameter management and consistent settings across components.

## Configuration Files

### Data Processing Config
```yaml
data_paths:
  base_path: "data"
  leagues:
    Premier_League:
      raw_data: "data/Premier_League/premier-league-matches-2007-2023"
      processed_data: "data/Premier_League/clean_premiere_league_data"
      start_year: 2012
      end_year: 2024
    France:
      raw_data: "data/France/league-1-matches-2009-2023"
      processed_data: "data/France/clean_league_1_data"
      start_year: 2009
      end_year: 2024

league_patterns:
  Premier_League: "england-premier-league-matches-*-stats.csv"
  France: "france-ligue-1-matches-*-stats.csv"

processing_params:
  last_n_matches: 5
```

### Model Training Config
```yaml
model_parameters:
  random_forest:
    n_estimators: 100
    max_depth: 10
    min_samples_split: 5
  svm:
    kernel: 'rbf'
    C: 1.0
    probability: True

training_settings:
  test_size: 0.2
  random_state: 42
  cv_folds: 5

output_settings:
  save_predictions: True
  save_model: True
  metrics_report: True
```

## Usage Examples

### Basic Configuration
```python
from src.Config.Config_Manager import ConfigManager

def get_config():
    config_path = 'src/Config/data_processing_config.yaml'
    return ConfigManager(config_path)
```

### Accessing Settings
```python
config = get_config()

# Get data paths
paths = config.get_paths()

# Get specific parameters
n_matches = config.get_config_value('processing_params', 'last_n_matches')
model_params = config.get_model_config('random_forest')
```

### Configuration Best Practices
1. Always use relative paths in config files
2. Include default values for optional parameters
3. Validate configuration on load
4. Use environment variables for sensitive data
5. Document all configuration options
