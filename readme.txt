# Football Match Prediction Project

Last update 17.12.2024

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

## Project Phases

### 1. Data Processing
- Data cleaning and standardization
- Feature engineering
- Statistical analysis
- Data validation

### 2. Model Development
#### Phase 1: Premier League
- Initial model development using Premier League data
- Binary classification (Win/Loss)
- Multiclass classification (Win/Draw/Loss)
- Model evaluation and optimization

#### Phase 2: Other Competitions
- Extend models to other leagues
- Analyze performance across different competitions
- Identify league-specific patterns

#### Phase 3: Combined Model
- Merge data from all competitions
- Develop unified prediction model
- Cross-competition validation

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

## Data Sources
- Premier League historical matches (2007-2023)
- [Add other data sources as they are incorporated]

## Model Evaluation
Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Cross-validation scores

## Usage
[To be completed as project develops]

### Data Processing
```bash
# Run data processing scripts
python src/Data_Processing/process_premier_league.py
```

### Training Models
```bash
# Run model training
python src/Models/Binary_Target/train_model.py
```

### Making Predictions
[To be added]

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License
[Choose and add appropriate license]

## Contact
Rivals Leonard - leonardrivals@gmail.com
Project Link: https://github.com/lrivals/FootWork

## Acknowledgments
- Data sources : https://footystats.org/
- Contributors :
	Leonard Rivals
- Thanks to Claude.ia
