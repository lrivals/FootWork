# FootWork — Football Match Prediction

Last update: 2026-02-25

## Overview

FootWork is a machine learning project that predicts football match outcomes
(Home Win / Draw / Away Win) across 6 major leagues. The pipeline covers the
full data science workflow: raw data ingestion, multi-season feature
engineering, temporal train/calibration/test splitting, multi-model training,
probability calibration, and threshold optimisation.

Two prediction tasks are tackled in parallel:

| Task | Target | Models |
| --- | --- | --- |
| **Multiclass** | HomeWin / Draw / AwayWin (3 classes) | 10 classifiers |
| **Binary** | Home Win vs Not, Away Win vs Not | 9 classifiers × 2 targets |

---

## Project Structure

```
FootWork/
├── data/
│   ├── all_leagues_combined.csv          # Master dataset (24 489 matches, 130 cols)
│   ├── Premier_League/clean_premiere_league_data/
│   ├── France/clean_league_1_data/
│   ├── Germany/clean_bundesliga_data/
│   ├── Italy/clean_serie-a_data/
│   ├── Spain/clean_la_liga_data/
│   └── Brazil/clean_Brazil_data/
│
├── src/
│   ├── Config/
│   │   ├── Config_Manager.py             # YAML config loader
│   │   ├── configMC_1.yaml               # Multiclass pipeline config
│   │   ├── configBT_1.yaml               # Binary pipeline config
│   │   └── data_processing_config.yaml   # Data processor config
│   │
│   ├── Data_Processing/
│   │   └── Multi-Season_Match_Data_Processor.py   # Main feature engineering
│   │
│   ├── Features/
│   │   ├── ELO_Rating.py                 # ELO calculator (cross-season)
│   │   └── __init__.py
│   │
│   └── Models/
│       ├── threshold_optimizer.py        # F1-optimal threshold search (binary + multiclass)
│       ├── Multiclass_Target/
│       │   └── Football_Match_Prediction_Pipeline.py
│       └── Binary_Target/
│           └── Football_Match_Binary_Prediction_Pipeline.py
│
├── notebooks/
│   ├── 01_Data_Processing.ipynb
│   ├── 02_Multiclass_Models.ipynb
│   └── 03_Binary_Models.ipynb
│
├── results/
│   └── All_Leagues/
│       ├── Multiclass_Target/multiclass_prediction<timestamp>/
│       │   ├── metrics_results_Full_Dataset_Multiclass.txt
│       │   ├── roc_*.png
│       │   ├── calibration_*.png
│       │   └── confusion_matrix_*.png
│       └── Binary_Target/Multiple_Model<timestamp>/
│           ├── metrics_results_Full_Dataset_Home_Win.txt
│           ├── metrics_results_Full_Dataset_Away_Win.txt
│           └── *.png
│
├── docs/
├── requirements.txt
└── readme.md
```

---

## Data

### Leagues & Coverage

| League | Country | Seasons | Matches |
| --- | --- | --- | --- |
| Premier League | England | 2012–2024 | ~4 400 |
| Ligue 1 | France | 2013–2024 | ~3 800 |
| Bundesliga | Germany | 2013–2024 | ~3 400 |
| Serie A | Italy | 2010–2024 | ~4 800 |
| La Liga | Spain | 2012–2024 | ~4 100 |
| Série A | Brazil | 2013–2024 | ~3 800 |
| **Total** | | **2012–2024** | **24 489** |

Source: [footystats.org](https://footystats.org/)

### Temporal Split (3-way)

```
──────────────────────────────────────────────────────────────────────
  TRAIN            │  CALIBRATION  │  TEST
  year < 2020      │  2020 – 2021  │  year ≥ 2022
  14 919 matches   │  4 309 matches│  5 261 matches
──────────────────────────────────────────────────────────────────────
```

- **Train** — fit all classifiers
- **Calibration** — fit `CalibratedClassifierCV(FrozenEstimator)` and optimise decision thresholds
- **Test** — final evaluation (never touched during training or calibration)

Class distribution in the test set: HomeWin 44 % · AwayWin 30 % · Draw 25 %

---

## Feature Engineering

130 columns total (after dropping metadata). Built by
`Multi-Season_Match_Data_Processor.py`, which processes each season
sequentially while preserving cross-season state.

### Per-Team Statistics (×2 teams = 96 features)

Each of the 48 features below is computed for both the home and away team:

| Category | Features |
| --- | --- |
| **Season aggregate** | games_played, wins, draws, losses, points_per_game, avg_goals_scored/conceded, avg_goal_diff |
| **xG / shooting** | avg_xg_scored/conceded, xg_vs_goals_diff, shot_conversion_rate, shots_on_target_ratio, avg_shots_per_game/on_target |
| **Possession / set pieces** | avg_possession, possession_efficiency, avg_corners_for, corner_efficiency |
| **Discipline** | avg_fouls_committed, avg_yellows, avg_reds, cards_first_half_ratio |
| **Venue** | venue_games, venue_win_ratio, venue_goals_avg, venue_conceded_avg |
| **Patterns** | clean_sheets_ratio, scoring_ratio, comeback_ratio, lead_loss_ratio, goals_scored/conceded_first_half_ratio |
| **Short-term form** | recent_ppg / goals_scored / goals_conceded / clean_sheets / draw_ratio for last **3, 5 and 10** matches |
| **Form trend** | form_trend (slope of recent points over last 10) |

### Odds-Derived Features (4)

| Feature | Description |
| --- | --- |
| `implied_prob_home/draw/away` | Bookmaker probabilities from odds (margin-removed) |
| `odds_ratio` | Home / Away implied probability ratio |

### ELO Features (3)

ELO ratings computed by `ELO_Rating.py` with K=20 and home advantage=100.
State persists across seasons (no reset at season start).

| Feature | Description |
| --- | --- |
| `home_elo` | Home team ELO before the match |
| `away_elo` | Away team ELO before the match |
| `elo_diff` | `home_elo - away_elo` |

### Differential Features (8)

Head-to-head comparisons between the two teams:

| Feature | Description |
| --- | --- |
| `diff_ppg` | Δ points per game (season) |
| `diff_recent_ppg` | Δ recent PPG (last 5) |
| `diff_goals_scored/conceded` | Δ average goals |
| `diff_xg_scored` | Δ expected goals scored |
| `diff_shots` | Δ shots per game |
| `diff_possession` | Δ average possession |
| `diff_form_trend` | Δ form trend slope |

### Draw Propensity Features (2)

| Feature | Description |
| --- | --- |
| `combined_draw_tendency` | Mean of home and away draw ratios (last 10) |
| `match_competitiveness` | ELO-based probability that strength is close |

### Head-to-Head History Features (6)

Historical H2H stats between the two teams. State persists across seasons
and across all processed leagues.

| Feature | Description |
| --- | --- |
| `h2h_home_wins` | H2H wins for the home team |
| `h2h_away_wins` | H2H wins for the away team |
| `h2h_draws` | H2H draws |
| `h2h_home_goals_avg` | Average goals scored by the home team in H2H |
| `h2h_away_goals_avg` | Average goals scored by the away team in H2H |
| `h2h_matches_count` | Total H2H matches recorded |

---

## Models

### Multiclass Pipeline (10 classifiers)

| Model | Class Balancing |
| --- | --- |
| Random Forest | `class_weight='balanced'` |
| Extra Trees | `class_weight='balanced'` |
| Gradient Boosting | `sample_weight` (balanced) |
| XGBoost | `sample_weight` (balanced) |
| LightGBM | `class_weight='balanced'` |
| CatBoost | `class_weights=[1.0, 1.74, 1.0]` (Draw boosted) |
| SVM | `class_weight='balanced'` |
| Logistic Regression | `class_weight='balanced'` |
| AdaBoost | `sample_weight` (balanced) |
| KNN | — |

### Binary Pipeline (9 classifiers × 2 targets)

Same classifiers (excluding XGBoost), trained independently for
**Home Win** and **Away Win**. CatBoost uses `compute_class_weight`
dynamically per target.

---

## Probability Calibration

After training, each fitted model is wrapped with
`CalibratedClassifierCV(FrozenEstimator(model), method='isotonic')`
and fit on the calibration set (2020–2021). This corrects overconfident
or underconfident probability estimates without retraining.

Calibration runs **serially** (after parallel training) so any failure
is fully visible with a traceback.

---

## Threshold Optimisation

Default threshold of 0.5 is often suboptimal when classes are imbalanced.
`src/Models/threshold_optimizer.py` implements:

- **Binary**: sweep thresholds on calibration probabilities, keep the one that maximises F1 on the positive class.
- **Multiclass (OvR)**: find one optimal F1 threshold per class, then predict via `argmax(proba / threshold)` — this boosts underrepresented classes (Draw, Away Win).

---

## Results — Test Set (5 261 matches, 2022–2024)

### Multiclass (HomeWin / Draw / AwayWin)

#### Ranking by Macro AUC

| Model | Accuracy | Bal. Acc | MCC | Macro AUC | Cal. AUC | Δ Cal |
| --- | --- | --- | --- | --- | --- | --- |
| **CatBoost** | 0.473 | 0.463 | 0.214 | **0.663** | **0.665** | +0.002 |
| Logistic Reg | 0.468 | 0.468 | 0.214 | 0.660 | 0.659 | -0.000 |
| AdaBoost | 0.464 | 0.466 | 0.213 | 0.655 | 0.658 | +0.003 |
| SVM | 0.484 | 0.461 | 0.210 | 0.648 | 0.643 | -0.005 |
| Extra Trees | 0.513 | 0.442 | 0.216 | 0.644 | 0.647 | +0.003 |
| Random Forest | **0.520** | 0.443 | 0.226 | 0.637 | 0.641 | +0.003 |
| LightGBM | 0.454 | 0.445 | 0.183 | 0.635 | 0.640 | +0.004 |
| XGBoost | 0.455 | 0.449 | 0.188 | 0.635 | 0.639 | +0.004 |
| Gradient Boost | 0.447 | 0.440 | 0.173 | 0.628 | 0.630 | +0.002 |
| KNN | 0.454 | 0.411 | 0.134 | 0.587 | 0.592 | +0.005 |

#### Per-Class AUC (One-vs-Rest)

| Model | AwayWin AUC | Draw AUC | HomeWin AUC |
| --- | --- | --- | --- |
| CatBoost | **0.716** | **0.559** | **0.713** |
| Logistic Reg | 0.714 | 0.555 | 0.710 |
| AdaBoost | 0.706 | 0.550 | 0.707 |
| SVM | 0.694 | 0.552 | 0.698 |
| Extra Trees | 0.692 | 0.544 | 0.697 |

> Draw is the hardest class to predict (~0.55 AUC vs ~0.71 for win/loss). This
> is a known structural difficulty: draws have lower correlation with historical
> features than decisive outcomes.

#### Impact of Threshold Optimisation (Multiclass)

| Model | Acc (default) | Acc (opt) | Bal.Acc (default) | Bal.Acc (opt) |
| --- | --- | --- | --- | --- |
| CatBoost | 0.473 | **0.501** | 0.463 | **0.476** |
| SVM | 0.484 | **0.518** | 0.461 | 0.456 |
| Random Forest | 0.520 | 0.491 | 0.443 | **0.449** |
| Logistic Reg | 0.468 | 0.485 | **0.468** | 0.463 |

Optimal OvR thresholds (CatBoost): AwayWin ~0.28 · Draw ~0.23 · HomeWin ~0.33
— all well below 0.5, reflecting class imbalance.

---

### Binary — Home Win

#### Ranking by ROC-AUC

| Model | Accuracy | Bal. Acc | MCC | ROC-AUC | Cal. AUC | Δ Cal |
| --- | --- | --- | --- | --- | --- | --- |
| **CatBoost** | **0.660** | **0.653** | **0.308** | **0.712** | 0.711 | -0.001 |
| AdaBoost | 0.659 | 0.652 | 0.306 | 0.706 | 0.706 | +0.000 |
| Logistic Reg | 0.658 | 0.645 | 0.299 | 0.711 | 0.708 | -0.003 |
| SVM | 0.654 | 0.648 | 0.296 | 0.693 | 0.692 | -0.001 |
| Extra Trees | 0.642 | 0.623 | 0.262 | 0.697 | 0.695 | -0.002 |
| LightGBM | 0.633 | 0.613 | 0.243 | 0.672 | 0.670 | -0.002 |
| Random Forest | 0.632 | 0.611 | 0.239 | 0.680 | 0.678 | -0.002 |
| Gradient Boost | 0.631 | 0.617 | 0.241 | 0.674 | 0.673 | -0.001 |
| KNN | 0.596 | 0.591 | 0.182 | 0.623 | 0.621 | -0.002 |

Calibration is slightly negative for all Home Win models (−0.001 to −0.003):
the models were already well-calibrated thanks to `class_weight='balanced'`.

#### Threshold Optimisation — Home Win

| Model | Bal.Acc (default) | Bal.Acc (opt) | Opt. Threshold |
| --- | --- | --- | --- |
| CatBoost | 0.653 | 0.636 | 0.37 |
| Logistic Reg | 0.645 | **0.640** | 0.36 |
| SVM | 0.648 | 0.618 | 0.32 |

---

### Binary — Away Win

#### Ranking by AUC and Balanced Accuracy

| Model | Accuracy | Bal. Acc | MCC | ROC-AUC | Cal. AUC | Δ Cal |
| --- | --- | --- | --- | --- | --- | --- |
| Extra Trees | **0.715** | 0.551 | 0.192 | 0.687 | 0.685 | -0.002 |
| AdaBoost | 0.713 | 0.643 | 0.298 | 0.706 | **0.706** | +0.001 |
| Random Forest | 0.703 | 0.516 | 0.104 | 0.680 | 0.679 | -0.001 |
| KNN | 0.685 | 0.575 | 0.175 | 0.623 | 0.622 | -0.000 |
| LightGBM | 0.666 | 0.644 | 0.272 | 0.695 | 0.692 | -0.002 |
| **CatBoost** | 0.663 | **0.656** | **0.291** | **0.719** | 0.719 | -0.000 |
| SVM | 0.656 | 0.645 | 0.271 | 0.685 | 0.684 | -0.001 |
| Logistic Reg | 0.638 | 0.654 | 0.284 | 0.714 | 0.711 | -0.003 |
| Gradient Boost | 0.620 | 0.631 | 0.240 | 0.684 | 0.681 | -0.002 |

> Extra Trees and Random Forest achieve the highest raw accuracy by almost
> never predicting Away Win (~4% recall). Their balanced accuracy (0.55 / 0.52)
> reveals this bias. CatBoost is the most reliable model: best AUC (0.719)
> and best balanced accuracy (0.656) simultaneously.

#### Threshold Optimisation — Away Win (Major Gains)

Away Win is minority class (30%). Default threshold 0.5 is strongly biased
toward "Not Away Win". The optimised threshold (~0.28) dramatically corrects this:

| Model | Bal.Acc (default) | Bal.Acc (opt) | Gain | Opt. Threshold |
| --- | --- | --- | --- | --- |
| Random Forest | 0.516 | **0.617** | **+10.2 pp** | 0.28 |
| Extra Trees | 0.551 | **0.637** | **+8.6 pp** | 0.27 |
| LightGBM | 0.644 | 0.641 | ~0 | 0.28 |
| CatBoost | 0.656 | 0.656 | ~0 | 0.29 |
| AdaBoost | 0.643 | 0.638 | −0.5 pp | 0.30 |

Models already using balanced sampling (CatBoost, LightGBM) gain little from
threshold optimisation; tree ensembles without it gain the most.

---

## Development Phases

### Phase 1 — Single League (Premier League)

- Basic feature engineering (aggregated season stats + recent form)
- Binary and multiclass classifiers evaluated on PL data only

### Phase 2 — Multi-League Expansion

- Standardised processing pipeline for France, Germany, Italy, Spain, Brazil
- All leagues processed with the same feature schema
- Combined dataset (`all_leagues_combined.csv`) created
- Cross-league model training

### Phase 3 — Model Quality Improvements

Four axes implemented simultaneously:

#### Axe 1 — Class Balancing

Added `class_weight='balanced'` to Logistic Regression, LightGBM, SVM,
Extra Trees, Random Forest. Added `sample_weight` for XGBoost, Gradient
Boosting, AdaBoost. CatBoost binary uses `compute_class_weight` dynamically.
CatBoost multiclass uses `class_weights=[1.0, 1.74, 1.0]` to boost Draw.

#### Axe 2 — Threshold Optimisation

Created `src/Models/threshold_optimizer.py` with F1-maximising threshold
search for binary and OvR multiclass. Integrated in both pipelines and
notebooks.

#### Axe 3 — Probability Calibration

Introduced 3-way temporal split (train / cal / test). Each fitted model
is wrapped with `CalibratedClassifierCV(FrozenEstimator(model),
method='isotonic')` and fit on the calibration set. Calibration curves
(PNG) saved for each model.

#### Axe 4 — H2H Features + ELO

- `ELO_Rating.py`: per-league ELO with cross-season persistence (K=20, home advantage=100). Produces `home_elo`, `away_elo`, `elo_diff`.
- H2H history: 6 features tracking historical results between teams, persisting across seasons and across league processing runs.
- 8 differential features comparing the two teams directly.
- 2 draw propensity features (`combined_draw_tendency`, `match_competitiveness`).

### Bug Fixes (post-Phase 3)

| Bug | Root Cause | Fix |
| --- | --- | --- |
| `best_iter=1` for CatBoost MultiClass | Metric `"MultiClass"` not matched by `'loss' in name` check → `argmax` on decreasing curve | Added `_LOSS_METRICS` set including `'multiclass'` |
| Calibration N/A (silent failure) | `CalibratedClassifierCV` ran inside `ThreadPoolExecutor` threads; exceptions swallowed by tqdm | Moved calibration to a serial loop after thread pool closes, with `traceback.print_exc()` |
| `cv='prefit'` invalid in sklearn 1.8 | Removed in sklearn 1.2+; raised `InvalidParameterError` silently caught by the serial loop | Replaced with `FrozenEstimator(model)` — new sklearn API for calibrating pre-fitted models |

---

## Setup

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- scikit-learn >= 1.2 (handled automatically by `environment.yml`)

### Installation

```bash
git clone git@github.com:lrivals/FootWork.git
cd FootWork

# Create and activate the conda environment
conda env create -f environment.yml
conda activate footwork

# Update an existing footwork env after changes to environment.yml
# conda env update -f environment.yml --prune
```

### Running the Pipelines

```bash
# 1. Build the combined dataset (only needed when feature engineering changes)
python src/Data_Processing/Multi-Season_Match_Data_Processor.py

# 2. Multiclass pipeline (HomeWin / Draw / AwayWin)
python src/Models/Multiclass_Target/Football_Match_Prediction_Pipeline.py

# 3. Binary pipeline (Home Win + Away Win)
python src/Models/Binary_Target/Football_Match_Binary_Prediction_Pipeline.py
```

Results are saved in `results/All_Leagues/<task>/<model_type><timestamp>/`.

### Notebooks

```bash
jupyter lab
# Open notebooks/01_Data_Processing.ipynb  → feature engineering walkthrough
# Open notebooks/02_Multiclass_Models.ipynb → multiclass training + results
# Open notebooks/03_Binary_Models.ipynb     → binary training + results
```

---

## Key Design Decisions

| Decision | Rationale |
| --- | --- |
| Temporal split (not random) | Avoids data leakage — future matches never inform past model training |
| Separate calibration set | Keeps calibration and test strictly independent |
| Per-league ELO (no cross-league) | ELO ratings are only meaningful within a league |
| Serial calibration (not parallel) | Exceptions are fully visible; calibration is fast compared to training |
| FrozenEstimator for calibration | Correct sklearn 1.2+ API; ensures base model weights are not modified |

---

## Contact

Rivals Leonard — leonardrivals@gmail.com

Project: <https://github.com/lrivals/FootWork>

## Contributors

- Leonard Rivals
- Maksym Lytvynenko
- Claude Sonnet 4.6 (Anthropic)
