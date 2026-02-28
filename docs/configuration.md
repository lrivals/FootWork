# Configuration Management

*Dernière mise à jour : Février 2026*

## Vue d'ensemble

Le projet utilise un système de configuration YAML centralisé via la classe `ConfigManager` (`src/Config/Config_Manager.py`). Chaque pipeline dispose de son propre fichier de config.

---

## Fichiers de configuration

| Fichier | Pipeline | Description |
|---------|----------|-------------|
| `src/Config/data_processing_config.yaml` | Data processing | Chemins des ligues, paramètres ELO et H2H |
| `src/Config/configMC_1.yaml` | Multiclass pipeline | Modèles sklearn, split temporel, Optuna, SHAP |
| `src/Config/configBT_1.yaml` | Binary pipeline | Identique à MC mais pour les cibles binaires |
| `src/Config/configNN_1.yaml` | Neural Network | Architecture DNN, training, class weights |
| `src/Config/configFS_1.yaml` | Feature selection | RFECV par modèle |

---

## Classe `ConfigManager`

```python
from src.Config.Config_Manager import ConfigManager

config = ConfigManager('src/Config/configMC_1.yaml')
```

### Méthodes principales

| Méthode | Description |
|---------|-------------|
| `get_paths()` | Retourne les chemins d'entrée/sortie |
| `get_config_value(*keys, default=None)` | Accès imbriqué au YAML |
| `get_model_config(model_name)` | Paramètres d'un modèle spécifique |
| `update_config(updates)` | Mise à jour et sauvegarde |
| `save_config_copy()` | Copie la config dans le dossier de sortie |
| `config['key']` | Accès dict-like |

---

## `configMC_1.yaml` — Pipeline Multiclasse

```yaml
data_paths:
  full_dataset: "data/all_leagues_combined.csv"

output_settings:
  base_path: "results/All_Leagues/Multiclass_Target"
  model_type: "multiclass_prediction"

# Colonnes exclues du training (metadonnees + cotes brutes)
excluded_columns:
  - date
  - season
  - league
  - home_team
  - away_team
  - target_result
  - target_home_goals
  - target_away_goals
  - raw_odds_home     # stockees mais non utilisees en training
  - raw_odds_draw
  - raw_odds_away

model_parameters:
  random_forest:
    n_estimators: 100
    random_state: 42
    class_weight: balanced

  logistic_regression:
    max_iter: 1000
    random_state: 42
    class_weight: balanced

  svm:
    probability: true
    random_state: 42
    class_weight: balanced

  gradient_boosting:
    n_estimators: 100
    learning_rate: 0.1
    random_state: 42

  xgboost:
    n_estimators: 100
    learning_rate: 0.1
    random_state: 42
    use_label_encoder: false
    eval_metric: mlogloss

  lightgbm:
    n_estimators: 100
    learning_rate: 0.1
    random_state: 42
    class_weight: balanced

  catboost:
    iterations: 100
    learning_rate: 0.1
    random_seed: 42
    verbose: false
    class_weights: [1.0, 1.74, 1.0]   # [AwayWin, Draw, HomeWin]

  neural_network:                       # sklearn MLPClassifier
    hidden_layer_sizes: [256, 128, 64]
    activation: relu
    solver: adam
    alpha: 0.001
    learning_rate: adaptive
    max_iter: 500
    early_stopping: true
    validation_fraction: 0.1
    n_iter_no_change: 20
    random_state: 42

  knn:
    n_neighbors: 5
    weights: distance

  adaboost:
    n_estimators: 100
    random_state: 42

  extra_trees:
    n_estimators: 100
    random_state: 42
    class_weight: balanced

data_split:
  temporal_split_year: 2022   # test >= 2022
  cal_split_year: 2020        # train < 2020, cal 2020-2021
  test_size: 0.2              # fallback si split temporel indisponible
  random_state: 42

# Phase 5 - Optuna (desactive par defaut)
optuna:
  enabled: false
  n_trials: 50
  timeout_seconds: 600        # 0 = illimite

# Phase 5 - SHAP (desactive par defaut, CatBoost uniquement)
shap:
  enabled: false
  n_samples: 500
```

**Notes importantes :**

- `class_weights: [1.0, 1.74, 1.0]` pour CatBoost : ordre alphabétique LabelEncoder → `[AwayWin, Draw, HomeWin]`, Draw surpondéré à 1.74.
- `raw_odds_*` exclues du training mais présentes dans le CSV — utilisées pour le calcul EV/Kelly du backtest.
- Le dossier de sortie est horodaté automatiquement par `ConfigManager` : `results/.../multiclass_prediction<timestamp>/`.

---

## `configBT_1.yaml` — Pipeline Binaire

Structure identique à `configMC_1.yaml`, deux différences :

```yaml
output_settings:
  base_path: "results/All_Leagues/Binary_Target"
  model_type: "Multiple_Model"
```

CatBoost sans `class_weights` (les cibles binaires sont gérées différemment, avec `class_weight: balanced` dans les modèles sklearn).

---

## `configNN_1.yaml` — Pipeline Neural Network (PyTorch)

```yaml
data_paths:
  full_dataset: "data/all_leagues_combined.csv"

output_settings:
  base_path: "results/All_Leagues/Neural_Network"

excluded_columns:
  - date
  - season
  - league
  - home_team
  - away_team
  - target_result
  - target_home_goals
  - target_away_goals

architecture:
  hidden_layers: [128, 64]
  dropout_rates: [0.4, 0.3]
  batch_norm: true

training:
  epochs: 200
  batch_size: 256
  learning_rate: 0.0005
  weight_decay: 0.001             # regularisation L2
  early_stopping_patience: 10
  scheduler_patience: 5           # ReduceLROnPlateau patience
  scheduler_factor: 0.5

class_weights:
  AwayWin: 1.0
  Draw: 1.5
  HomeWin: 1.0

data_split:
  temporal_split_year: 2022
  cal_split_year: 2020
```

---

## Usage

### Accès aux chemins

```python
config = ConfigManager('src/Config/configMC_1.yaml')
paths = config.get_paths()
# paths['full_dataset'] → "data/all_leagues_combined.csv"
# paths['output_dir']   → "results/.../multiclass_prediction<timestamp>/"
```

### Accès aux paramètres

```python
# Parametres d'un modele specifique
rf_params = config.get_model_config('random_forest')
# → {'n_estimators': 100, 'random_state': 42, 'class_weight': 'balanced'}

# Valeur imbriquee avec defaut
split_year = config.get_config_value('data_split', 'temporal_split_year')
# → 2022

optuna_enabled = config.get_config_value('optuna', 'enabled', default=False)
# → False
```

### Activation d'Optuna

```yaml
optuna:
  enabled: true
  n_trials: 50
  timeout_seconds: 600
```

Voir [optuna_and_threshold_optimizer.md](optuna_and_threshold_optimizer.md) pour les détails.

### Activation SHAP

```yaml
shap:
  enabled: true
  n_samples: 500   # reduire pour accelerer
```

---

## Bonnes pratiques

1. Utiliser des chemins relatifs dans les configs
2. Les colonnes exclues du training doivent rester dans le CSV (nécessaires pour EV, filtrage par ligue, etc.)
3. Le dossier de sortie est créé automatiquement avec un timestamp par `ConfigManager`
4. Copier la config dans le dossier de sortie via `save_config_copy()` pour reproductibilité
