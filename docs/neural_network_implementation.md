# Pipeline Neural Network (PyTorch) — Documentation

**Module :** `src/Models/Neural_Network/Football_Match_NN_Pipeline.py`
**Config :** `src/Config/configNN_1.yaml`
**Framework :** PyTorch + sklearn
**Sortie :** `results/All_Leagues/Neural_Network/`

> Note : Ce document décrit l'implémentation technique. Pour l'analyse de faisabilité initiale, voir [neural_network_analysis.md](neural_network_analysis.md).

---

## Architecture du réseau

### `FootballMatchNet` (nn.Module)

MLP (Multi-Layer Perceptron) avec BatchNorm et Dropout pour la prédiction à 3 classes :

```
Input (n_features)
    → Linear(n_features, 256) + BatchNorm1d(256) + ReLU + Dropout(0.4)
    → Linear(256, 128)        + BatchNorm1d(128) + ReLU + Dropout(0.3)
    → Linear(128, 64)         + BatchNorm1d(64)  + ReLU + Dropout(?)
    → Linear(64, 3)
    → Softmax (via CrossEntropyLoss pendant training, explicite en inference)
```

La structure des couches est entièrement configurable via `configNN_1.yaml` :
- `hidden_layers` : tailles des couches cachées (ex : `[128, 64]`)
- `dropout_rates` : taux de dropout par couche (ex : `[0.4, 0.3]`)

---

## Classe `FootballNNWrapper` (sklearn-compatible)

Enveloppe `FootballMatchNet` dans l'API sklearn (`BaseEstimator`, `ClassifierMixin`) pour permettre l'usage du threshold optimizer sans modification.

### Méthodes principales

#### `fit(X, y, X_val=None, y_val=None)`

Entraîne le réseau PyTorch.

- Si `X_val` est `None`, 10% des données d'entraînement sont réservées pour la validation interne.
- **Optimizer :** Adam avec `weight_decay` pour la régularisation L2
- **Loss :** CrossEntropyLoss avec `class_weights` (pour rééquilibrer Draw et AwayWin)
- **Scheduler :** `ReduceLROnPlateau` — réduit le LR si la val_loss stagne
- **Early stopping :** arrêt si la val_loss ne s'améliore pas pendant `patience` epochs ; restauration des meilleurs poids
- Stocke `loss_history_` : `{'train': [...], 'val': [...]}`

#### `predict_proba(X)`

Retourne les probabilités softmax (array numpy N×3). Mode `eval()` + `torch.no_grad()`.

#### `predict(X)`

Retourne `argmax(predict_proba(X))`.

---

## Classe `_IsotonicCalibratedNN`

Calibration isotonique par classe pour `FootballNNWrapper`.

**Pourquoi cette classe custom ?** `CalibratedClassifierCV(FrozenEstimator(...))` échoue car `FrozenEstimator` ne transmet pas `_estimator_type` à `is_classifier()` de sklearn — bug connu.

### Fonctionnement

```python
calibrated = _IsotonicCalibratedNN(wrapper)
calibrated.fit(X_cal, y_cal)  # un IsotonicRegression par classe (OvR)
probs = calibrated.predict_proba(X_test)  # normalisation ligne après calibration
```

Pour chaque classe `i` :
1. `y_bin = (y_cal == i).astype(float)` — cible binaire OvR
2. `IsotonicRegression(out_of_bounds='clip').fit(raw_probs[:, i], y_bin)` — calibrage monotone
3. Les probabilités calibrées sont normalisées pour sommer à 1 par ligne.

---

## Paramètres de configuration (`configNN_1.yaml`)

```yaml
architecture:
  hidden_layers: [128, 64]       # tailles des couches cachées
  dropout_rates: [0.4, 0.3]      # dropout par couche
  batch_norm: true               # BatchNorm activé

training:
  epochs: 200                    # max epochs (early stopping peut arrêter avant)
  batch_size: 256
  learning_rate: 0.0005
  weight_decay: 0.001            # régularisation L2 (Adam weight_decay)
  early_stopping_patience: 10    # patience early stopping
  scheduler_patience: 5          # patience ReduceLROnPlateau
  scheduler_factor: 0.5          # facteur de réduction du LR

class_weights:
  AwayWin: 1.0
  Draw: 1.5                      # surpondération des nuls
  HomeWin: 1.0
```

**Ordre LabelEncoder (alphabétique) :** `['AwayWin', 'Draw', 'HomeWin']` → indices `[0, 1, 2]`

---

## Split temporel

Identique aux autres pipelines :
- **Entraînement :** date < 2020 (~14 900 matchs)
- **Calibration :** 2020 ≤ date < 2022 (~4 300 matchs) — utilisé pour `_IsotonicCalibratedNN` et threshold optimizer
- **Test :** date ≥ 2022 (~5 200 matchs)

---

## Pipeline complet (`train_nn_pipeline`)

1. Chargement des données + split temporel 3 voies
2. StandardScaler fitté sur train
3. Initialisation `FootballNNWrapper` avec paramètres de la config
4. Entraînement : `wrapper.fit(X_train, y_train, X_val=X_cal, y_val=y_cal)`
5. Évaluation brute (avant calibration) + plots ROC + loss curve + confusion matrix
6. Calibration isotonique : `_IsotonicCalibratedNN(wrapper).fit(X_cal, y_cal)`
7. Optimisation des seuils : `find_optimal_thresholds_multiclass()` sur le cal set
8. Prédictions finales avec seuils optimaux
9. Sauvegarde résultats

---

## Sorties produites

| Fichier | Description |
|---------|-------------|
| `metrics_results_DNN_PyTorch.txt` | Métriques complètes : brutes, calibrées, seuils optimaux |
| `loss_curve_DNN_PyTorch.png` | Courbes train/validation loss + meilleure epoch |
| `roc_DNN_PyTorch.png` | ROC curves OvR avant calibration |
| `roc_DNN_PyTorch_calibrated.png` | ROC curves OvR après calibration |
| `calibration_DNN_PyTorch.png` | Reliability diagrams (brut vs calibré) par classe |
| `confusion_matrix_DNN_PyTorch_raw.png` | Matrice de confusion avant calibration |
| `confusion_matrix_DNN_PyTorch_opt.png` | Matrice de confusion avec seuils optimaux |

### Contenu de `metrics_results_DNN_PyTorch.txt`

- Métriques baselines (accuracy, balanced accuracy, MCC, Kappa, Macro AUC)
- AUC par classe (OvR)
- Comparaison calibré vs non-calibré (Macro AUC delta)
- Seuils optimaux par classe avec leur F1 sur le cal set
- Classification reports (seuils par défaut et optimaux)
- Matrices de confusion
- Résumé de la courbe d'entraînement (total epochs, best epoch, best val_loss)

---

## Utilisation

```bash
python src/Models/Neural_Network/Football_Match_NN_Pipeline.py
```

Le script utilise `src/Config/configNN_1.yaml` par défaut.

**Prérequis :**
```bash
pip install torch torchvision
# ou avec conda :
conda install pytorch -c pytorch
```

Le réseau détecte automatiquement CUDA et l'utilise si disponible (`torch.device('cuda' if torch.cuda.is_available() else 'cpu')`).

---

## Différences avec le pipeline sklearn multiclass

| Aspect | Pipeline sklearn (`configMC_1.yaml`) | Pipeline NN (`configNN_1.yaml`) |
|--------|--------------------------------------|----------------------------------|
| Calibration | `CalibratedClassifierCV` (isotonic/sigmoid) | `_IsotonicCalibratedNN` custom |
| Class weights | `class_weight` param sklearn | `CrossEntropyLoss(weight=...)` |
| Threshold optimizer | `find_optimal_thresholds_multiclass()` | Identique |
| Format sortie | `predictions_<Model>.csv` | Pas de CSV de prédictions (métriques uniquement) |
| GPU support | Non | Oui (CUDA auto-détecté) |

---

## Notes de performance

- Dataset d'entraînement ~14 900 matchs : modeste pour du deep learning
- Le GBDT (CatBoost, LightGBM) tend à surpasser les MLP sur des données tabulaires structurées
- L'avantage potentiel du NN réside dans la capture d'interactions complexes non-linéaires entre features
- BatchNorm améliore la stabilité d'entraînement sur des features hétérogènes (différentes échelles)
