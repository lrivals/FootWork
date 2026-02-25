# Instructions — Prochaine session : Phase 3 (améliorations post-analyse)

Copier-coller ce bloc en début de conversation.

---

## Contexte du projet

Projet **FootWork** : prédiction de résultats de matchs de football (6 ligues, ~24k matchs).

- **Split temporel** : train < 2022 / test ≥ 2022
- **Multiclass** : HomeWin / Draw / AwayWin (`src/Models/Multiclass_Target/Football_Match_Prediction_Pipeline.py`)
- **Binary** : HomeWin vs ¬HomeWin et AwayWin vs ¬AwayWin (`src/Models/Binary_Target/Football_Match_Binary_Prediction_Pipeline.py`)
- **Configs** : `src/Config/configMC_1.yaml` (multiclass) et `src/Config/configBT_1.yaml` (binary)
- **Données combinées** : `data/all_leagues_combined.csv`
- **Feature engineering** : `src/Data_Processing/Multi-Season_Match_Data_Processor.py`
- **ELO (à créer)** : `src/Features/ELO_Rating.py`

Derniers résultats (run 2026-02-24) :
- Multiclass best accuracy : CatBoost 53 % — **Draw quasi non prédit (recall 1 %)**
- Binary HomeWin : CatBoost 66 % ROC-AUC 0.71 — correct
- Binary AwayWin : modèles biaisés (Random Forest recall AW = 13 %) — **déséquilibre sévère**
- Analyses complètes dans `results/All_Leagues/*/analyse.md`

---

## 4 axes à implémenter (dans cet ordre)

---

### Axe 1 — Rééquilibrage des classes (`class_weight`)

**Problème** : le Draw (25 % des matchs) et AwayWin (30 %) sont ignorés faute de pondération.

**Ce qu'il faut faire :**

1. Dans `configMC_1.yaml` et `configBT_1.yaml`, ajouter `class_weight: balanced` pour tous les modèles qui le supportent (LogReg, SVM, RF, ET, GB, AdaBoost, ExtraTrees).

2. Pour XGBoost (multiclass) : pas de `class_weight` natif → utiliser `sample_weight` calculé :
```python
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight('balanced', y_train)
# Passer dans model.fit(..., sample_weight=sample_weights)
# Modifier _fit_with_tracking() pour accepter sample_weight en paramètre optionnel
```

3. Pour LightGBM : `class_weight: balanced` dans les params YAML.

4. Pour CatBoost : `class_weights` dans les params (liste, pas string) :
```yaml
catboost:
  class_weights: [1.0, 1.74, 1.0]   # multiclass : [AwayWin, Draw, HomeWin]
  # Pour binary AwayWin : [1.0, 2.3]  (69.7/30.3)
```

5. Modifier `_fit_with_tracking()` dans les deux pipelines pour propager `sample_weight` si fourni.

6. Ajouter la même logique dans les notebooks (`02_Multiclass_Models.ipynb`, `03_Binary_Models.ipynb`).

**Fichiers à modifier :**
- `src/Config/configMC_1.yaml`
- `src/Config/configBT_1.yaml`
- `src/Models/Multiclass_Target/Football_Match_Prediction_Pipeline.py` (`_fit_with_tracking`, `train_and_evaluate_model`, `train_all_models`)
- `src/Models/Binary_Target/Football_Match_Binary_Prediction_Pipeline.py` (idem)
- `notebooks/02_Multiclass_Models.ipynb` (cell `_fit_with_tracking` + training loop)
- `notebooks/03_Binary_Models.ipynb` (idem)

---

### Axe 2 — Seuils de décision optimisés par classe

**Problème** : le seuil par défaut (argmax des probas / 0.5 binaire) n'est pas optimal pour les classes déséquilibrées.

**Ce qu'il faut faire :**

1. Créer une fonction `find_optimal_threshold(model, X_val, y_val, metric='f1')` dans un nouveau fichier `src/Models/threshold_optimizer.py` :
```python
from sklearn.metrics import precision_recall_curve, f1_score
import numpy as np

def find_optimal_threshold(model, X_val, y_val, pos_label=1):
    """Trouve le seuil qui maximise le F1-score pour la classe positive."""
    proba = model.predict_proba(X_val)[:, pos_label]
    precision, recall, thresholds = precision_recall_curve(y_val, proba, pos_label=pos_label)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1[:-1])
    return thresholds[best_idx], f1[best_idx]
```

2. Pour le multiclass : utiliser un seuil par classe (OvR). Après entraînement, chercher le seuil qui maximise le F1 Draw sur un fold de validation (ou split temporel interne 2020–2021).

3. Intégrer dans `train_and_evaluate_model()` des deux pipelines : calculer les métriques avec le seuil optimisé **en plus** des métriques au seuil par défaut (ne pas remplacer, comparer).

4. Sauvegarder les seuils optimaux dans les résultats texte.

5. Ajouter une cellule dédiée dans les deux notebooks après l'entraînement.

**Fichiers à créer/modifier :**
- `src/Models/threshold_optimizer.py` (nouveau)
- `src/Models/Multiclass_Target/Football_Match_Prediction_Pipeline.py`
- `src/Models/Binary_Target/Football_Match_Binary_Prediction_Pipeline.py`
- `notebooks/02_Multiclass_Models.ipynb`
- `notebooks/03_Binary_Models.ipynb`

---

### Axe 3 — Calibration des probabilités (`CalibratedClassifierCV`)

**Problème** : les AUC sont correctes (~0.67–0.71) mais les probabilités brutes sont mal calibrées — les modèles ne donnent pas des P(AwayWin) ou P(Draw) fiables en valeur absolue.

**Ce qu'il faut faire :**

1. Après chaque `model.fit(...)` dans `train_and_evaluate_model()`, wrapper le modèle dans `CalibratedClassifierCV` en mode `'prefit'` (car déjà entraîné) sur un fold de validation interne :
```python
from sklearn.calibration import CalibratedClassifierCV

# Utiliser les données 2020-2021 comme validation de calibration
calibrated = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
calibrated.fit(X_cal, y_cal)
```

2. Créer un split de calibration dans `load_and_prepare_data()` : conserver les années 2020–2021 comme ensemble de calibration (distinct du test ≥ 2022).

3. Comparer les métriques avec/sans calibration : ajouter `calibrated_macro_auc` dans les résultats.

4. Tracer une **courbe de calibration** (`sklearn.calibration.CalibrationDisplay`) par modèle et sauvegarder en PNG dans le dossier output.

5. Mettre à jour `save_results()` pour inclure une section "Calibrated vs Uncalibrated AUC".

6. Ajouter dans les notebooks une cellule dédiée avec `CalibrationDisplay.from_predictions()`.

**Fichiers à modifier :**
- `src/Models/Multiclass_Target/Football_Match_Prediction_Pipeline.py` (`load_and_prepare_data`, `train_and_evaluate_model`, `save_results`)
- `src/Models/Binary_Target/Football_Match_Binary_Prediction_Pipeline.py` (idem)
- `notebooks/02_Multiclass_Models.ipynb`
- `notebooks/03_Binary_Models.ipynb`

---

### Axe 4 — ELO ratings + Head-to-Head features (Phase 3)

**Problème** : les features actuelles sont des rolling windows par équipe, sans capture de la force relative dynamique entre deux équipes spécifiques.

**Ce qu'il faut faire :**

#### 4a — ELO ratings

Créer `src/Features/ELO_Rating.py` :

```python
"""
ELO rating calculator intégré dans le pipeline de feature engineering.

Principe :
- Chaque équipe commence avec un ELO de 1500
- Après chaque match : ELO est mis à jour selon le résultat vs l'attendu
- K-factor recommandé : 20 (stable) ou 32 (sensible)
- Les features à ajouter : elo_home, elo_away, elo_diff (home - away)
- Cross-saison : l'ELO persiste d'une saison à l'autre (comme team_history)
"""

class ELOCalculator:
    def __init__(self, k=20, initial_elo=1500):
        self.k = k
        self.initial_elo = initial_elo
        self.ratings = {}  # {team_name: elo}

    def get_elo(self, team):
        return self.ratings.get(team, self.initial_elo)

    def expected_score(self, elo_a, elo_b):
        return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

    def update(self, home_team, away_team, home_goals, away_goals):
        """Met à jour les ELOs après un match."""
        ...

    def get_pre_match_features(self, home_team, away_team):
        """Retourne (elo_home, elo_away, elo_diff) AVANT le match."""
        ...
```

Intégrer dans `Multi-Season_Match_Data_Processor.py` :
- Instancier `ELOCalculator` au début du traitement
- Pour chaque match (en ordre chronologique) : récupérer les ELOs PRE-match → stocker dans le row → puis mettre à jour les ELOs
- Ajouter les colonnes : `elo_home`, `elo_away`, `elo_diff`
- **Important** : l'ELO doit être calculé AVANT le match (pas de data leakage)

#### 4b — Head-to-Head features

Dans `Multi-Season_Match_Data_Processor.py`, ajouter une structure `h2h_history = {}` (clé : frozenset {équipe_A, équipe_B}) :

Features à calculer (5 dernières rencontres entre les deux équipes) :
- `h2h_home_wins` : victoires à domicile de l'équipe home dans le h2h
- `h2h_away_wins` : victoires à l'extérieur de l'équipe away dans le h2h
- `h2h_draws` : nuls dans le h2h
- `h2h_home_goals_avg` : moyenne de buts de l'équipe home dans le h2h
- `h2h_away_goals_avg` : idem away
- `h2h_matches_count` : nombre de matchs disputés entre les deux (pour pondérer)

**Ordre de traitement** : toujours ajouter les features H2H AVANT de mettre à jour l'historique (éviter le leakage).

#### 4c — Intégration dans la config

Ajouter dans `data_processing_config.yaml` :
```yaml
features:
  elo:
    enabled: true
    k_factor: 20
    initial_rating: 1500
  head_to_head:
    enabled: true
    window: 5  # 5 dernières rencontres
```

**Fichiers à créer/modifier :**
- `src/Features/ELO_Rating.py` (nouveau)
- `src/Data_Processing/Multi-Season_Match_Data_Processor.py` (intégration ELO + H2H)
- `src/Config/data_processing_config.yaml` (nouvelles clés)
- Régénérer `data/all_leagues_combined.csv` après les changements

---

## Ordre d'implémentation recommandé

```
1. Axe 1 (class_weight)     → impact immédiat, risque faible, 1–2h
2. Axe 2 (seuils)           → dépend d'Axe 1 pour être pertinent, 1h
3. Axe 3 (calibration)      → dépend d'Axe 1, nécessite split calibration, 2h
4. Axe 4a (ELO)             → le plus impactant long terme, 2–3h
5. Axe 4b (H2H)             → complémentaire à ELO, 1–2h
6. Réentraîner les pipelines et comparer avec les résultats actuels
```

## Validation finale

Après implémentation, comparer avec la run de référence :
- Multiclass Draw recall : doit passer de 1 % (CatBoost) à >20 %
- AwayWin balanced accuracy : doit passer de 0.60 à >0.65
- Macro AUC multiclass : doit passer de 0.668 à >0.70 (avec ELO)

Commandes de réentraînement :
```bash
# Régénérer les données (si ELO/H2H ajoutés)
python src/Data_Processing/Multi-Season_Match_Data_Processor.py

# Réentraîner
python src/Models/Multiclass_Target/Football_Match_Prediction_Pipeline.py
python src/Models/Binary_Target/Football_Match_Binary_Prediction_Pipeline.py
```
