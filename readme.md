# Football Match Prediction Project

Last update 19.12.2024

## Overview
This project aims to develop machine learning models for predicting football match outcomes across multiple European leagues. The development follows a systematic approach:

1. Initial Development (Premier League)
   - Data processing and feature engineering ( more info in `docs/Data_processing.md` )
   - Binary prediction models for Home/Away wins
   - Multiclass prediction including Draw outcomes
   - Model evaluation and optimization

2. Multi-League Integration
   - Extension to other major European leagues (Ligue 1, Bundesliga, Serie A, La Liga)
   - Standardized data processing across leagues
   - League-specific model performance analysis

3. Cross-League Development
   - Combined dataset creation
   - Universal prediction model development
   - League-specific adjustments and calibration

## Project Structure
```
FootWork/
├── data/                          # League-specific data
│   ├── Premier_League/           
│   │   ├── clean_premiere_league_data/    
│   │   └── premier-league-matches-2007-2023/
│   ├── France/
│   │   ├── clean_league_1_data/
│   │   └── league-1-matches-2009-2023/
│   ├── Germany/
│   │   └── bundesliga-matches-2006-to-2023/
│   ├── Italy/
│   │   └── serie-a-matches-2008-to-2023/
│   └── Spain/
│       └── la-liga-matches-2008-to-2023/
├── docs/                          # Logging directory & documentation
│   └── Models/
├── results/                      # Analysis outputs
│   └── [League_Name]/
├── src/                         # Source code
    ├── Analysis/               
    ├── Config/                 
    ├── Data_Processing/        
    └── Models/                 
```

## Model Evaluation
Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Cross-validation scores



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
2. Modify parameters as needed for your specific use case
3. Ensure correct data paths and processing parameters

### Data Processing
```bash
# Process league data
python src/Data_Processing/Multi-Season_Match_Data_Processor.py --config src/Config/data_processing_config.yaml
```

## Data Sources
- Premier League (2012-2024)
- Ligue 1 (2009-2024)
- Bundesliga (2006-2024)
- Serie A (2008-2024)
- La Liga (2008-2024)
Source: https://footystats.org/

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
  - ?
- Thanks to Claude.ai
Note: For detailed information about model performance and configuration management, please refer to the documentation in `docs/model_performance.md` and `docs/configuration.md` respectively.



