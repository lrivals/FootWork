# Optimisation des hyperparamètres (Optuna) et des seuils de décision

Ce document couvre deux modules complémentaires d'optimisation :
- **`src/Models/Multiclass_Target/optuna_tuner.py`** — recherche automatique des hyperparamètres CatBoost
- **`src/Models/threshold_optimizer.py`** — optimisation des seuils de décision par classe

---

## 1. Optuna Tuner (`optuna_tuner.py`)

### Vue d'ensemble

Utilise [Optuna](https://optuna.org/) pour optimiser automatiquement les hyperparamètres de CatBoost et la méthode de calibration. L'objectif est de maximiser un F1 pondéré sur le set de calibration, avec une emphase sur les classes minoritaires (Draw, AwayWin).

**Activé via config :** `optuna.enabled: true` dans `configMC_1.yaml`

---

### Espace de recherche

| Hyperparamètre | Type | Plage | Description |
|----------------|------|-------|-------------|
| `depth` | int | [4, 10] | Profondeur maximale des arbres |
| `learning_rate` | float (log) | [0.01, 0.2] | Taux d'apprentissage |
| `l2_leaf_reg` | float (log) | [1.0, 15.0] | Régularisation L2 |
| `bagging_temperature` | float | [0.0, 2.0] | Contrôle le bruit dans le bagging |
| `min_data_in_leaf` | int | [10, 100] | Nombre minimum d'exemples par feuille |
| `class_weight_awaywin` | float | [1.0, 2.5] | Poids de la classe AwayWin (index 0) |
| `class_weight_draw` | float | [1.0, 3.5] | Poids de la classe Draw (index 1) |
| `calibration_method` | catégorique | `['isotonic', 'sigmoid']` | Méthode de calibration |

HomeWin (index 2) est fixé à `1.0`.

**Ordre LabelEncoder (alphabétique) :** `[AwayWin=0, Draw=1, HomeWin=2]`

---

### Métrique objectif

```python
score = 0.4 × F1_HomeWin + 0.3 × F1_Draw + 0.3 × F1_AwayWin
```

Cette pondération favorise Draw et AwayWin (classes minoritaires), tout en maintenant HomeWin comme classe principale.

---

### Fonctions

#### `run_optuna_study(X_train, y_train, X_cal, y_cal, n_trials=50, timeout=600, output_dir)`

Lance l'étude Optuna et retourne les meilleurs paramètres.

**Arguments :**
| Argument | Défaut | Description |
|----------|--------|-------------|
| `X_train`, `y_train` | — | Données d'entraînement (numpy, labels encodés) |
| `X_cal`, `y_cal` | — | Données de calibration |
| `n_trials` | 50 | Nombre d'essais Optuna |
| `timeout` | 600 | Durée maximale en secondes (0 = illimité) |
| `output_dir` | `/tmp/optuna_catboost` | Répertoire pour les logs CatBoost |

**Retourne :** `dict` compatible avec les kwargs de `CatBoostClassifier` :

```python
{
    'depth': int,
    'learning_rate': float,
    'l2_leaf_reg': float,
    'bagging_temperature': float,
    'min_data_in_leaf': int,
    'class_weights': [float, float, float],   # [AwayWin, Draw, HomeWin]
    'calibration_method': str,                 # 'isotonic' ou 'sigmoid'
}
```

#### `save_best_params(best_params, output_dir)`

Sauvegarde les meilleurs paramètres dans `best_params_optuna.yaml` pour reproductibilité.

---

### Activation dans le pipeline

Dans `configMC_1.yaml` :

```yaml
optuna:
  enabled: true          # activer la recherche
  n_trials: 50           # nombre d'essais
  timeout_seconds: 600   # limite de temps (10 min)
```

Quand activé, le pipeline remplace les hyperparamètres CatBoost par défaut par ceux trouvés par Optuna, et utilise la `calibration_method` optimale pour CatBoost uniquement.

---

### Prérequis

```bash
pip install optuna
```

---

## 2. Threshold Optimizer (`threshold_optimizer.py`)

### Vue d'ensemble

Optimise les seuils de décision pour classifier multi- ou bi-classe. Au lieu d'utiliser `0.5` comme seuil fixe pour toutes les classes, on trouve le seuil qui maximise le F1 pour chaque classe sur le set de calibration.

**Effet principal :** les classes minoritaires (Draw, AwayWin) obtiennent souvent un seuil optimal < 0.5, ce qui augmente leur recall sans modifier le modèle.

---

### Fonctions

#### `find_optimal_threshold(proba_pos, y_val) → (threshold, f1)`

Trouve le seuil binaire qui maximise le F1 pour la classe positive.

```python
precision, recall, thresholds = precision_recall_curve(y_val, proba_pos)
f1 = 2 × precision × recall / (precision + recall + 1e-8)
best_threshold = thresholds[argmax(f1)]
```

**Arguments :**
- `proba_pos` : array 1D — probabilités prédites pour la classe positive
- `y_val` : array 1D binaire — labels vrais (0/1)

**Retourne :** `(float, float)` — seuil optimal et son F1

---

#### `find_optimal_thresholds_multiclass(model, X_val, y_val) → dict`

Trouve un seuil optimal par classe (stratégie OvR).

```python
for i in range(n_classes):
    y_bin = (y_val == i).astype(int)   # OvR binaire
    thr, f1 = find_optimal_threshold(proba[:, i], y_bin)
    thresholds[i] = (thr, f1)
```

**Retourne :** `{class_idx: (threshold, f1)}`, ex :
```python
{0: (0.28, 0.41), 1: (0.21, 0.35), 2: (0.39, 0.52)}
# AwayWin       Draw             HomeWin
```

---

#### `predict_with_thresholds(proba, thresholds) → array`

Prédit les classes en ajustant les probabilités par leurs seuils respectifs.

```python
adjusted = proba / threshold_array   # rescaling
prediction = argmax(adjusted, axis=1)
```

**Principe :** une classe dont le seuil optimal est bas (ex. Draw à 0.21) voit ses scores relatifs augmenter, ce qui la rend plus susceptible d'être prédite. C'est équivalent à baisser la barre pour les classes sous-prédites.

**Arguments :**
- `proba` : array 2D (N_samples, N_classes)
- `thresholds` : `dict` retourné par `find_optimal_thresholds_multiclass`

**Retourne :** array 1D d'entiers (indices de classes)

---

#### `predict_binary_with_threshold(proba_pos, threshold) → array`

Applique un seuil personnalisé aux prédictions binaires.

```python
return (proba_pos >= threshold).astype(int)
```

Utilisé dans le pipeline hiérarchique pour chaque stage.

---

### Intégration dans les pipelines

Le threshold optimizer est utilisé **après** la calibration isotonique dans tous les pipelines :

```python
# 1. Calibration
calibrated = CalibratedClassifierCV(FrozenEstimator(model), method='isotonic')
calibrated.fit(X_cal, y_cal)

# 2. Optimisation des seuils sur le cal set
optimal_thresholds = find_optimal_thresholds_multiclass(calibrated, X_cal, y_cal)

# 3. Prédiction avec seuils sur le test set
proba_test = calibrated.predict_proba(X_test)
y_pred_opt = predict_with_thresholds(proba_test, optimal_thresholds)
```

Les résultats CSV exportent les deux prédictions :
- `pred_default` : argmax sans ajustement de seuil
- `pred_opt` : argmax avec seuils optimaux (via `predict_with_thresholds`)

---

### Impact attendu

L'optimisation des seuils améliore principalement :
- **Draw recall** : seuil typiquement < 0.25, forte augmentation du recall
- **AwayWin recall** : seuil souvent < 0.35
- **Balanced Accuracy** : amélioration notable vs seuil par défaut

Au prix d'une légère baisse de l'accuracy globale (HomeWin moins prédit, qui était sur-représenté).
