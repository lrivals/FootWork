# Analyse : Pertinence d'un Réseau de Neurones pour FootWork

**Date :** 2026-02-25
**Contexte :** Évaluation post-Phase 3 — 10 modèles ML classiques entraînés, meilleures performances actuelles connues

---

## 1. État des Lieux

### Données disponibles
| Dimension | Valeur |
|-----------|--------|
| Matchs total | 24 489 |
| Features numériques | 121 |
| Features catégorielles utilisables | équipes (204 uniques) |
| Valeurs manquantes | 0% |
| Split train / cal / test | <2020 / 2020-2021 / ≥2022 |
| Matchs en train | ~14 000 |
| Matchs en test | ~5 261 |

### Target (multiclasse)
| Classe | Fréquence |
|--------|-----------|
| HomeWin | 45.3% |
| AwayWin | 29.4% |
| Draw | 25.4% |

### Meilleures performances actuelles
| Métrique | Valeur | Modèle |
|----------|--------|--------|
| Accuracy multiclasse | 52.0% | Random Forest |
| Macro AUC multiclasse | 0.663 | CatBoost |
| AUC Draw | 0.559 | CatBoost |
| Accuracy binaire (Home Win) | 66.0% | CatBoost |
| AUC binaire (Home Win) | 0.712 | CatBoost |

---

## 2. Analyse de Pertinence

### 2.1 Arguments POUR un réseau de neurones

**A. Interactions non-linéaires complexes**
Les 121 features contiennent des groupes naturellement interdépendants :
- ELO × forme récente × odds implicites → interactions que les GBDT capturent mal globalement
- `diff_possession` × `diff_xg_scored` × `shot_conversion_rate` → patron tactique non-linéaire
- Les features H2H interagissent avec les features de forme dans un contexte historique long

Un MLP profond apprend automatiquement ces combinaisons sans avoir à les spécifier manuellement.

**B. Embeddings d'équipes appris**
Avec 204 équipes, les noms ne sont actuellement pas utilisés comme features directes.
Un NN peut apprendre des **vecteurs d'équipe de 16-32 dimensions** capturant le "style de jeu", le niveau, la culture tactique — au-delà de ce que les stats rolling capturent.

**C. Draw sous-prédit (AUC 0.56)**
Le nul est structurellement difficile pour les GBDT car les frontières de décision sont floues. Une architecture multi-tâche (prédire résultat + écart de buts) peut donner un signal de supervision plus riche, particulièrement pour les matchs serrés (home PPG ≈ away PPG → Draw probable).

**D. Flexibilité architecturale**
- Calibration intégrée via une couche Softmax (probabilités somment à 1, pas de post-calibration obligatoire)
- Threshold optimization directement applicable (module `threshold_optimizer.py` existant)
- Class weights gérables nativement (`pos_weight` PyTorch)

**E. MLP déjà prévu dans le code**
`MLPClassifier` est **commenté** dans les deux pipelines — preuve que l'idée était planifiée. Le réactiver avec de bons hyperparamètres est une quick win.

---

### 2.2 Arguments CONTRE (risques et limites)

**A. Taille dataset modeste pour du deep learning**
14k matchs en train : en dessous du seuil habituel où les NNs deviennent clairement supérieurs aux GBDT sur données tabulaires (règle empirique : ~50k-100k samples).
→ Risque de surapprentissage nécessitant une régularisation agressive (Dropout ≥ 0.3, L2, early stopping).

**B. GBDT dominent sur données tabulaires**
Consensus de la littérature (TabNet, revisité en 2023-2024) : les GBDT (CatBoost, XGBoost, LightGBM) restent supérieurs aux NNs sur la plupart des jeux de données tabulaires < 500k rows.
→ Le gain attendu est **marginal** : +1-3% accuracy, +0.01-0.02 AUC, non garanti.

**C. Temps d'entraînement et complexité**
Un DNN PyTorch correctement tuné (hyperparams, scheduler, early stopping) est significativement plus lent et complexe à maintenir que CatBoost.

**D. Interprétabilité réduite**
Les SHAP values sont disponibles pour CatBoost/XGBoost. Un MLP profond est plus opaque, même si SHAP Gradient est applicable.

---

### 2.3 Verdict

> **OUI, pertinent — avec une stratégie progressive en 3 niveaux.**

Le ratio bénéfice/effort est favorable si l'on commence par le niveau 1 (MLP sklearn rapide) avant de s'engager dans du PyTorch. La vraie valeur ajoutée viendra des **embeddings d'équipes** et du **multi-task learning** (Niveau 3), qui explorent une direction que les modèles actuels ne peuvent pas prendre.

L'objectif réaliste :
- Court terme : **+1-2% accuracy multiclasse** (MLP sklearn)
- Moyen terme : **+2-4% accuracy + AUC Draw > 0.58** (DNN PyTorch avec class weights)
- Long terme : **architecture différente** (LSTM sur séquences brutes = potentiel transformateur)

---

## 3. Plan d'Implémentation

### Niveau 1 — MLP Sklearn (Quick Win, ~1h de travail)

**Objectif :** Baseline NN intégré dans le pipeline existant, comparable directement aux 10 autres modèles.

**Fichiers à modifier :**
- [src/Config/configMC_1.yaml](../src/Config/configMC_1.yaml) — décommenter/ajouter MLPClassifier
- [src/Config/configBT_1.yaml](../src/Config/configBT_1.yaml) — idem pour binary
- [src/Models/Multiclass_Target/Football_Match_Prediction_Pipeline.py](../src/Models/Multiclass_Target/Football_Match_Prediction_Pipeline.py) — décommenter MLPClassifier

**Configuration YAML à ajouter :**
```yaml
MLPClassifier:
  hidden_layer_sizes: [256, 128, 64]
  activation: relu
  solver: adam
  alpha: 0.001          # L2 regularization
  learning_rate: adaptive
  max_iter: 500
  early_stopping: true
  validation_fraction: 0.1
  n_iter_no_change: 20
  random_state: 42
```

**Résultat attendu :** Directement comparé aux 10 modèles dans les fichiers metrics_results_*.txt existants.

---

### Niveau 2 — DNN PyTorch avec Calibration (Recommandé, ~2-3 jours)

**Objectif :** Architecture profonde optimisée pour la prédiction de matchs de football, avec calibration et threshold optimization intégrées.

**Nouveau fichier :** `src/Models/Neural_Network/Football_Match_NN_Pipeline.py`

**Architecture recommandée :**
```
Input(121)
    │
    ▼
Dense(256) + BatchNorm + ReLU + Dropout(0.3)
    │
    ▼
Dense(128) + BatchNorm + ReLU + Dropout(0.3)
    │
    ▼
Dense(64) + BatchNorm + ReLU + Dropout(0.2)
    │
    ▼
Dense(3) + Softmax → [P(HomeWin), P(Draw), P(AwayWin)]
```

**Stratégie d'entraînement :**
```python
# Loss avec class weights pour compenser l'imbalance Draw (25%)
class_weights = torch.tensor([1.0, 1.74, 1.0])  # inverse freq Draw
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Scheduler : réduire LR si val_loss plateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=10, factor=0.5, min_lr=1e-5
)

# Early stopping : 30 epochs sans amélioration
```

**Split temporel :** Identique au pipeline existant
- Train : matchs avant 2020 (~14k)
- Calibration : 2020-2021 (~5k) → threshold optimization
- Test : ≥2022 (~5.2k)

**Intégration avec l'existant :**
- Réutiliser `src/Models/threshold_optimizer.py` pour les seuils optimaux par classe
- Réutiliser `CalibratedClassifierCV` sklearn (wrapper PyTorch → sklearn interface)
- Génération des mêmes métriques (accuracy, balanced accuracy, MCC, macro AUC, per-class AUC)

**Fichier de config :** `src/Config/configNN_1.yaml`
```yaml
architecture:
  hidden_layers: [256, 128, 64]
  dropout_rates: [0.3, 0.3, 0.2]
  activation: relu
  batch_norm: true

training:
  epochs: 200
  batch_size: 256
  learning_rate: 0.001
  weight_decay: 0.0001
  early_stopping_patience: 30

class_weights:
  HomeWin: 1.0
  Draw: 1.74
  AwayWin: 1.0

data:
  train_year_max: 2020
  calibration_year_max: 2022
  test_year_min: 2022
```

---

### Niveau 3 — Architecture Avancée (Optionnel, futur)

**A. Team Embeddings**
```python
# Ajouter une couche Embedding pour chaque équipe
team_embedding = nn.Embedding(num_teams=204, embedding_dim=32)

# Forward pass :
home_emb = team_embedding(home_team_idx)  # [batch, 32]
away_emb = team_embedding(away_team_idx)  # [batch, 32]
x = torch.cat([stats_features, home_emb, away_emb], dim=1)  # [batch, 121+64]
```

**B. Multi-Task Learning**
Prédire simultanément résultat + nombre de buts :
```
Shared layers (256 → 128)
    ├── Tête 1 : Dense(64) → Softmax(3)  [résultat]
    └── Tête 2 : Dense(64) → Output(2)   [home_goals, away_goals]

Loss = α × CrossEntropy(result) + (1-α) × MSE(goals), α=0.7
```

**C. LSTM sur séquences de matchs**
Nécessite une restructuration majeure des données :
```
Pour chaque match à prédire :
  - Séquence home_team : derniers 10 matchs (chacun = vecteur de features)
  - Séquence away_team : derniers 10 matchs

Architecture :
  LSTM(input=n_features, hidden=64, layers=2) × 2 (home + away)
  → Concat → Dense → Softmax
```
Cette approche est la plus prometteuse à long terme mais nécessite de refactoriser la génération de données.

---

## 4. Métriques Cibles

| Niveau | Accuracy Multiclasse | Macro AUC | AUC Draw |
|--------|---------------------|-----------|----------|
| Baseline actuel | 52.0% | 0.663 | 0.559 |
| Niveau 1 (MLP sklearn) | 50-54% | 0.65-0.67 | 0.56-0.60 |
| Niveau 2 (DNN PyTorch) | 52-56% | 0.66-0.68 | 0.57-0.62 |
| Niveau 3 (embeddings + multitask) | 54-58% | 0.67-0.70 | 0.60-0.65 |

> Note : La prédiction de matchs de football est fondamentalement bornée autour de 52-60% (bruit inhérent au sport). Tout modèle dépassant 60% de manière consistante serait suspect.

---

## 5. Ordre de Priorité Recommandé

1. **[PRIORITÉ 1]** Décommenter `MLPClassifier` dans les pipelines existants + tuner les hyperparamètres → effort minimal, résultat immédiat
2. **[PRIORITÉ 2]** Implémenter le DNN PyTorch Niveau 2 avec class weights et early stopping → principale valeur ajoutée
3. **[PRIORITÉ 3]** Ajouter team embeddings au DNN existant → faible coût marginal, potentiel élevé
4. **[FUTUR]** LSTM sur séquences brutes → refonte significative, réserver pour une Phase 4 dédiée

---

## 6. Dépendances à Installer

```bash
pip install torch torchvision torchaudio  # PyTorch CPU
# ou pour GPU :
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install torch-tb-profiler  # optionnel : profiling TensorBoard
```

Pour le Niveau 1 sklearn, aucune dépendance supplémentaire (sklearn déjà installé).
