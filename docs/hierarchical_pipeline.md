# Pipeline Hiérarchique — Documentation

**Module :** `src/Models/Hierarchical/Football_Match_Hierarchical_Pipeline.py`
**Config utilisée :** `src/Config/configMC_1.yaml`
**Dépendances :** CatBoost, sklearn, seaborn, src/Models/threshold_optimizer.py
**Sortie :** `results/All_Leagues/Hierarchical/hierarchical_<timestamp>/`

---

## Motivation

Les classifieurs multiclasses standards souffrent d'un biais systématique : ils sous-prédisent fortement les nuls (Draw), qui représentent pourtant 25,7% des matchs. L'approche hiérarchique décompose le problème en deux sous-problèmes binaires pour améliorer le recall de la classe Draw.

---

## Architecture à 2 Étages

```
Match
  │
  ▼
Stage 1: HomeWin vs Not-HomeWin  (CatBoost binaire + calibration isotonique)
  │
  ├── P(HomeWin)  ──────────────────────────────────────────┐
  │                                                          │
  └── P(Not-HomeWin) ──────────────────────────────────────┐│
                │                                           ││
                ▼                                           ││
        Stage 2: Draw vs AwayWin                           ││
        (entraîné uniquement sur les matchs Not-HomeWin)   ││
                │                                           ││
                ├── P(Draw | Not-HW)                        ││
                └── P(AwayWin | Not-HW)                     ││
                                                            ││
Probabilités finales combinées :                            ││
  P(HomeWin)  = Stage1.P(HomeWin)  ◄─────────────────────────┘│
  P(Draw)     = (1 − P(HomeWin)) × Stage2.P(Draw)             │
  P(AwayWin)  = (1 − P(HomeWin)) × Stage2.P(AwayWin) ◄────────┘
```

### Formules de combinaison

```python
p_home  = proba_stage1              # P(HomeWin)
p_not_home = 1.0 - proba_stage1     # P(Not-HomeWin)
p_draw  = p_not_home * proba_stage2 # P(Draw) = P(Not-HW) × P(Draw | Not-HW)
p_away  = p_not_home * (1 - proba_stage2)  # P(AwayWin)
```

La prédiction finale = argmax des probabilités combinées (avec ajustement de seuil optionnel).

---

## Paramètres CatBoost (chaque stage)

| Paramètre | Valeur |
|-----------|--------|
| `iterations` | 200 |
| `learning_rate` | 0.05 |
| `random_seed` | 42 |
| `verbose` | False |

---

## Split temporel

Identique au pipeline multiclass :
- **Entraînement :** matchs avec date < 2020
- **Calibration :** 2020 ≤ date < 2022
- **Test :** date ≥ 2022

---

## Fonctions clés

### `load_data(config)`
Charge le dataset, applique le split temporel, extrait les métadonnées du set de test (date, équipes, cotes brutes, résultat réel).

### `prepare_features(df, exclude_columns, scaler=None, fit_scaler=False)`
Supprime les colonnes exclues, normalise via `StandardScaler`. Le scaler est fitté sur le set d'entraînement et appliqué à cal et test.

### `_train_catboost_stage(X_train, y_train, stage_name, output_dir)`
Entraîne un CatBoost binaire pour un étage. Logs CatBoost écrits dans `output_dir/catboost_<stage_name>/`.

### `_calibrate(model, X_cal, y_cal)`
Applique une calibration isotonique via `CalibratedClassifierCV(FrozenEstimator(model), method='isotonic')` sur le set de calibration.

### `hierarchical_predict(proba_s1, proba_s2, thresh1=0.5, thresh2=0.5)`

Combine les sorties des deux stages. Supporte l'ajustement de seuils :

```python
scores[:, 0] /= thresh1   # ajustement seuil HomeWin
scores[:, 1] /= thresh2   # ajustement seuil Draw
# prédiction = argmax(scores)
```

**Arguments :**
- `proba_s1` : array (N,) — P(HomeWin) depuis Stage 1
- `proba_s2` : array (N,) — P(Draw | Not-HomeWin) depuis Stage 2
- `thresh1` : seuil Stage 1 (optimal via calibration set)
- `thresh2` : seuil Stage 2 (optimal via calibration set)

**Retourne :**
- `predictions` : array de str (`HomeWin`, `Draw`, `AwayWin`)
- `combined_proba` : array (N, 3) — `[P(HomeWin), P(Draw), P(AwayWin)]`

### `run_hierarchical_pipeline(config, output_dir)`

Orchestre l'ensemble du pipeline :

1. Chargement des données et split temporel
2. **Stage 1** : entraînement + calibration isotonique sur les not-HomeWin
3. Optimisation du seuil Stage 1 via `find_optimal_threshold()` sur le cal set
4. **Stage 2** : filtre les matchs Not-HomeWin → entraînement + calibration
5. Optimisation du seuil Stage 2 sur le cal set (Not-HomeWin uniquement)
6. Prédictions finales : `pred_default` (argmax pur) + `pred_opt` (seuils optimisés)
7. Métriques, plots (confusion matrix), sauvegarde CSV

---

## Sorties produites

| Fichier | Description |
|---------|-------------|
| `metrics_hierarchical.txt` | Accuracy, Balanced Accuracy, MCC, Kappa, rapport de classification |
| `confusion_matrix_hierarchical.png` | Matrice de confusion Stage1+Stage2 |
| `predictions_Hierarchical.csv` | Prédictions par match avec prob_*, pred_*, ev_*, kelly_* |
| `catboost_stage1/`, `catboost_stage2/` | Logs d'entraînement CatBoost |

### Format du CSV de prédictions

Compatible avec `Betting_Backtest.py` :

| Colonne | Description |
|---------|-------------|
| `prob_homewin`, `prob_draw`, `prob_awaywin` | Probabilités calibrées de la cascade |
| `pred_default` | Argmax des probabilités combinées (thresh=0.5) |
| `pred_opt` | Prédiction avec seuils optimaux (thresh1, thresh2 du cal set) |
| `ev_home`, `ev_draw`, `ev_away` | Expected Value = prob × odds − 1 |
| `kelly_home`, `kelly_draw`, `kelly_away` | Kelly quart, cappé à 10% |

---

## Métriques calculées

- Accuracy
- Balanced Accuracy
- MCC (Matthews Correlation Coefficient)
- Cohen's Kappa
- Classification report par classe (precision, recall, F1)
- ROC-AUC par stage (binaire)

---

## Utilisation

```bash
python src/Models/Hierarchical/Football_Match_Hierarchical_Pipeline.py
```

Résultats dans `results/All_Leagues/Hierarchical/hierarchical_<timestamp>/`.

Pour backtest des prédictions :

```bash
python src/Analysis/Betting_Backtest.py \
    --predictions results/All_Leagues/Hierarchical/hierarchical_<timestamp>/predictions_Hierarchical.csv \
    --bankroll 1000 \
    --output results/Backtest/
```

---

## Avantages et limites

**Avantages :**
- Améliore structurellement le recall Draw en lui dédiant un classifieur binaire optimisé
- Calibration et optimisation de seuils indépendantes pour chaque stage
- Output compatible avec le module de backtest

**Limites :**
- Stage 2 ne voit que les matchs Not-HomeWin → set d'entraînement réduit (~9 000 matchs)
- Les erreurs de Stage 1 se propagent à Stage 2 (cascade d'erreurs)
- Deux modèles à maintenir vs un seul modèle multiclasse
