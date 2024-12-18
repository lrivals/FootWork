# Football Match Prediction Project

Last update 18.12.2024

## Overview
This project aims to develop machine learning models for predicting football match outcomes. The approach is incremental, starting with Premier League data and gradually expanding to include other competitions. The final goal is to create a robust model that can make predictions across different leagues and competitions.

## Project Structure
```
FootWork/
├── data/                          # Data storage directory
│   └── Premier_League/           # Premier League specific data
│       ├── clean_premiere_league_data/    # Processed and cleaned data
│       ├── premier-league-matches-2007-2023/  # Raw historical data
│       └── Stat_Test/            # Statistical test results and analysis
├── results/                      # Model outputs and analysis results
│   └── Premier_League/
│       ├── Analysis/            # Data analysis results
│       ├── Binary_Target/       # Binary prediction results (Win/Loss)
│       └── Multiclass_Target/   # Multiple class prediction results
├── src/                         # Source code
│   ├── Analysis/               # Scripts for data analysis
│   ├── Config/                 # Configuration files
│   ├── Data_Processing/        # Data cleaning and preparation scripts
│   └── Models/                 # Prediction models
│       ├── Binary_Target/      # Binary outcome prediction models
│       └── Multiclass_Target/  # Multiple class outcome prediction models
```


## Model Performance Comparison

### Binary Prediction
Best performers:
- Away Win: Random Forest/AdaBoost (72.52% accuracy)
- Home Win: SVM (64.30% accuracy)

### Multiclass Prediction (HomeWin/AwayWin/Draw)
- Top accuracy: Logistic Regression & AdaBoost (52.48%)
- Major challenge: Poor Draw prediction (best recall 36% with SVM)
- Home/Away predictions significantly weaker than binary approach

### Recommendation
Use binary approach with specialized models:
- SVM for Home Win prediction
- Random Forest for Away Win prediction
- Skip Draw prediction due to low accuracy and class imbalance

Reasoning: Binary models show ~20% better accuracy than multiclass approach and provide more reliable predictions for practical use.



## Configuration Management

### Overview
The project uses a YAML-based configuration system through `ConfigManager` class, ensuring consistent settings across components and providing centralized parameter management.

### Configuration Structure
```yaml
data_paths:
  full_dataset: "path/to/dataset.csv"
model_parameters:
  random_forest:
    n_estimators: 100
cross_validation:
  n_splits: 5
```

### Implementation Examples

1. Basic Configuration Loading:
```python
from src.Config.Config_Manager import ConfigManager

def get_config():
    config_path = 'src/Config/configBT_1.yaml'
    return ConfigManager(config_path)
```

2. Accessing Configuration Values:
```python
# Get paths and parameters
paths = config.get_paths()
n_splits = config.get_config_value('cross_validation', 'n_splits', default=5)
```

## Setup Instructions

### Prerequisites
- Python 3.8+
- Git
- Virtual environment tool (venv recommended)

### Installation
1. Clone the repository:
```bash
git clone git@github.com:lrivals/FootWork.git
cd FootWork
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix
# or
venv\Scripts\activate     # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Configuration Setup
1. Copy appropriate configuration template from `src/Config/`
2. Modify parameters as needed
3. Specify the config file path when running scripts

### Data Processing
```bash
# Run data processing scripts with config
python src/Data_Processing/process_premier_league.py --config src/Config/configBT_1.yaml
```

### Training Models
```bash
# Run model training with config
python src/Models/Binary_Target/train_model.py --config src/Config/configBT_1.yaml
```

## Data Sources
- Premier League historical matches (2007-2023)
- Source: https://footystats.org/

## Model Evaluation
Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Cross-validation scores

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Contact
Rivals Leonard - leonardrivals@gmail.com
Project Link: https://github.com/lrivals/FootWork

## Acknowledgments
- Data sources: https://footystats.org/
- Contributors:
  - Leonard Rivals
- Thanks to Claude.ai