# Model Performance Analysis

*Derniere mise a jour : Fevrier 2026 (etat post-Phase 3 + Phase 4)*

---

## Pipeline Multiclasse (post-Phase 3)

**Dataset :** `data/all_leagues_combined.csv` — Test set : matchs >= 2022 (~5 200 matchs)

Performances avec calibration isotonique + optimisation des seuils de decision.

### Classement par Macro AUC

| Modele | Accuracy (defaut) | Accuracy (seuils opt.) | Macro AUC | Draw Recall |
|--------|-------------------|------------------------|-----------|-------------|
| CatBoost | 47.3% | **50.1%** | **0.663** | ~15% |
| LightGBM | 51.7% | 49.5% | 0.658 | ~7% |
| XGBoost | 51.9% | 49.7% | 0.655 | ~6% |
| Random Forest | **52.0%** | 49.8% | 0.651 | ~8% |
| Gradient Boosting | 52.5% | 50.0% | 0.649 | ~3% |
| Logistic Regression | 52.2% | 49.6% | 0.648 | ~3% |
| AdaBoost | 51.8% | 49.9% | 0.641 | ~1% |
| Extra Trees | 51.2% | 49.4% | 0.638 | ~8% |
| SVM | 47.2% | 48.1% | 0.621 | **~35%** |
| KNN | 45.3% | 47.2% | 0.598 | ~23% |
| Neural Network (MLP sklearn) | 41.4% | 45.0% | 0.587 | ~25% |

**Observations :**
- CatBoost : meilleur Macro AUC (0.663) — modele principal pour usage combine
- Avec seuils optimises, l'accuracy globale baisse mais Draw/AwayWin recall augmentent
- SVM : seul a atteindre ~35% recall Draw, au prix d'une accuracy globale basse
- GBDT (CatBoost, LightGBM, XGBoost) dominent sur le Macro AUC

### Performances par classe — CatBoost avec seuils optimises

| Classe | Precision | Recall | F1 |
|--------|-----------|--------|----|
| HomeWin | ~63% | ~58% | ~60% |
| Draw | ~36% | ~32% | ~34% |
| AwayWin | ~52% | ~55% | ~53% |
| Macro avg | ~50% | ~48% | ~49% |

### Binary AUC par classe — CatBoost calibre (OvR)

| Classe | ROC-AUC |
|--------|---------|
| HomeWin | ~0.712 |
| Draw | ~0.561 |
| AwayWin | ~0.719 |
| Macro | **0.663** |

---

## Pipeline Binaire (post-Phase 3)

### HomeWin vs Not-HomeWin

| Modele | Accuracy | ROC-AUC | Balanced Acc |
|--------|----------|---------|--------------|
| CatBoost | 66.0% | **0.712** | 0.634 |
| Logistic Regression | 65.0% | 0.709 | 0.620 |
| SVM | 64.3% | 0.695 | 0.615 |
| Random Forest | 63.8% | 0.702 | 0.608 |

### AwayWin vs Not-AwayWin

| Modele | Accuracy | ROC-AUC | Balanced Acc |
|--------|----------|---------|--------------|
| Extra Trees | 72.5% | 0.714 | 0.651 |
| CatBoost | ~72.0% | **0.719** | **0.656** |
| Random Forest | 72.0% | 0.712 | 0.648 |
| AdaBoost | 72.5% | 0.708 | 0.643 |

**Observations :**
- AwayWin plus facile a predire en binaire (AUC 0.719 vs 0.712 HomeWin)
- Les modeles binaires sont plus fiables pour les cas tranches (victoire nette)
- La prediction de nuls reste difficile meme en binaire

---

## Pipeline Hierarchique (post-Phase 4)

Architecture cascade Stage1 (HomeWin vs Not) + Stage2 (Draw vs Away).

*Les resultats exacts sont dans `results/All_Leagues/Hierarchical/metrics_hierarchical.txt`.*

**Comportement attendu vs pipeline multiclasse :**
- Draw recall : amelioration significative (Stage2 dedie au Draw vs Away)
- HomeWin recall : similaire ou legerement inferieur
- Accuracy globale : legerement inferieure (trade-off Draw recall)

---

## Pipeline Neural Network PyTorch (post-Phase 4)

*Les resultats exacts sont dans `results/All_Leagues/Neural_Network/metrics_results_DNN_PyTorch.txt`.*

**Comportement typique vs GBDT :**
- Macro AUC generalement inferieur aux GBDT sur donnees tabulaires
- Draw recall potentiellement meilleur avec class_weights PyTorch
- Necessite GPU (CUDA) pour des temps d'entrainement raisonnables

---

## Evolution des performances (Phase 1 a Phase 3+)

| Metrique | Phase 1 (dec. 2024) | Phase 3+ (fev. 2026) | Delta |
|----------|---------------------|----------------------|-------|
| Accuracy MC | ~52.5% | ~50.1% (opt.) | -2.4% |
| Macro AUC | ~0.62 | **0.663** | +0.043 |
| Draw Recall (SVM) | ~36% | ~35% | stable |
| HomeWin AUC (binaire) | ~0.683 | **0.712** | +0.029 |
| AwayWin AUC (binaire) | ~0.705 | **0.719** | +0.014 |

*Note : la baisse d'accuracy multiclasse s'explique par l'optimisation des seuils qui favorise deliberement Draw/AwayWin. Le Macro AUC (independant des seuils) s'ameliore de +4.3 points.*

---

## Plafond de performance du football

- **Baseline naive** (toujours predire HomeWin) : ~45% accuracy
- **Plafond estime** pour le football : ~58-62% accuracy multiclasse
- **Etat de l'art** (modeles avec donnees completes) : ~55-58%
- **AUC Draw** (~0.56) structurellement bas — peu de signal predictif sur les nuls

---

## Recommandations

### Pour le backtesting / paris

Utiliser **CatBoost avec seuils optimises** (`pred_opt`) :
- Meilleur Macro AUC = meilleures probabilites calibrees pour EV
- Voir [backtest.md](backtest.md) pour les strategies de paris

### Pour la precision maximale HomeWin/AwayWin

Utiliser les **pipelines binaires** avec CatBoost ou Extra Trees :
- AUC ~0.71 sur chaque cible binaire
- Predictions plus fiables pour les outcomes non-Draw

### Pour ameliorer le Draw

- **SVM** : meilleur recall Draw (35%) mais probas moins calibrees
- **Pipeline Hierarchique** : meilleur compromis Draw recall / accuracy globale
- **Seuils optimises CatBoost** : compromise acceptable pour usage general

### Ameliorations futures

- Optuna : optimisation hyperparametres CatBoost (activer via `optuna.enabled: true`)
- SHAP : analyse feature importance (activer via `shap.enabled: true`)
- Features additionnelles : donnees joueurs (blessures, suspensions), forme domicile/exterieur par adversaire
- League-specific models : calibration separee par ligue
