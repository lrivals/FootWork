# Analyse des résultats — Prédiction Binaire (HomeWin / AwayWin)
> Run : `Multiple_Model2026-02-24_12-43-37` · Split temporel : train < 2022 / test ≥ 2022 · Test : **5 261 matchs**

---

## 1. Contexte et distribution des cibles

| Cible binaire      | Positifs | Négatifs | Taux positif | Baseline naïve |
|--------------------|----------|----------|--------------|----------------|
| HomeWin            | 2 330    | 2 931    | 44,3 %       | 55,7 % (tout ¬HW) |
| AwayWin            | 1 593    | 3 668    | 30,3 %       | 69,7 % (tout ¬AW) |

L'approche binaire découpe le problème en deux tâches indépendantes. Le **AwayWin** souffre d'un déséquilibre plus prononcé (30/70) — un modèle naïf qui ne prédit jamais "Away Win" obtient déjà **69,7 % d'accuracy**.

---

## 2. HomeWin — Résultats et classement

### 2.1 Tableau de classement

| Rang | Modèle              | Accuracy | Bal. Acc | MCC    | Kappa  | ROC-AUC |
|------|---------------------|----------|----------|--------|--------|---------|
| 1    | **CatBoost**        | **0.6649**| 0.6533  | **0.3136**| **0.3111**| **0.7141**|
| 2    | AdaBoost            | 0.6582   | **0.6504**| 0.3033| 0.3028 | 0.7002  |
| 3    | Logistic Regression | 0.6567   | 0.6478   | 0.2991 | 0.2982 | 0.7130  |
| 4    | Random Forest       | 0.6525   | 0.6355   | 0.2851 | 0.2779 | 0.6995  |
| 5    | Gradient Boosting   | 0.6520   | 0.6347   | 0.2838 | 0.2764 | 0.7053  |
| 6    | SVM                 | 0.6514   | 0.6439   | 0.2899 | 0.2895 | 0.6942  |
| 7    | Extra Trees         | 0.6476   | 0.6298   | 0.2744 | 0.2666 | 0.6986  |
| 8    | LightGBM            | 0.6444   | 0.6319   | 0.2705 | 0.2679 | 0.6965  |
| 9    | KNN                 | 0.6016   | 0.5906   | 0.1843 | 0.1833 | 0.6222  |

*XGBoost absent du résultat HomeWin — probablement une erreur d'exécution silencieuse.*

### 2.2 Observations HomeWin

**La tâche HomeWin est la mieux résolue des trois.** L'écart entre les modèles est faible (64–66 %), mais l'ensemble des métriques est cohérent.

- **CatBoost domine** : meilleur sur tous les critères (accuracy, MCC, Kappa, ROC-AUC). ROC-AUC = 0.7141.
- **LogReg très proche** : 0.7130 AUC, 0.2991 MCC. Meilleure référence linéaire.
- **Le spread accuracy/bal.accuracy est faible** (~1,5 pts pour CatBoost) : les modèles ne sont pas fortement biaisés vers la classe majoritaire (¬HomeWin = 55,7 %).

### 2.3 Matrices de confusion — cas extrêmes

**CatBoost (meilleur global)** :
```
              Prédit ¬HW  Prédit HW
Réel ¬HW       2 212       719    → recall ¬HW = 75 %
Réel HW        1 044     1 286    → recall HW = 55 %
```
- 719 faux positifs (HomeWin prédit à tort) — erreur type I raisonnable
- 1 044 faux négatifs (HomeWin manqué) — recall 55 % est correct pour le football

**Random Forest (recall HW le plus bas)** :
```
              Prédit ¬HW  Prédit HW
Réel ¬HW       2 300       631    → recall ¬HW = 78 % (meilleur ¬HW)
Réel HW        1 197     1 133    → recall HW = 49 % (le plus bas)
```
Random Forest penche trop vers ¬HomeWin. Moins utile pour identifier les victoires à domicile.

### 2.4 Courbe d'entraînement (boosters)

| Modèle            | Métrique        | Best iter | Train   | Val     | Overfit |
|-------------------|-----------------|-----------|---------|---------|---------|
| CatBoost          | Logloss         | 37        | 0.5744  | 0.6165  | ✅ Faible |
| Gradient Boosting | accuracy        | 52        | 0.6813  | 0.6520  | ✅ Modéré |
| LightGBM          | binary_logloss  | 26        | 0.4875  | 0.6278  | ⚠️ Fort  |

LightGBM sur-entraîne fortement : écart train/val = 0.14 points de logloss. CatBoost est le plus régulier.

---

## 3. AwayWin — Résultats et classement

### 3.1 Tableau de classement (deux lectures nécessaires)

| Rang | Modèle              | Accuracy | Bal. Acc | MCC    | Kappa  | ROC-AUC |
|------|---------------------|----------|----------|--------|--------|---------|
| 1★   | **CatBoost**        | **0.7276**| 0.6051  | 0.2720 | 0.2457 | **0.7186**|
| —    | **Logistic Regression** | 0.6565 | **0.6599**| **0.2957**| 0.2822 | 0.7157|
| —    | **AdaBoost**        | 0.7143   | 0.6380   | **0.2920**| **0.2897**| 0.7100|
| —    | Extra Trees         | 0.7191   | 0.5688   | 0.2188 | 0.1713 | 0.6917  |
| —    | Random Forest       | 0.7172   | 0.5502   | 0.1992 | 0.1303 | 0.6835  |
| —    | LightGBM            | 0.7168   | 0.5948   | 0.2418 | 0.2204 | 0.6990  |
| —    | Gradient Boosting   | 0.7141   | 0.5839   | 0.2244 | 0.1985 | 0.6992  |
| —    | KNN                 | 0.6803   | 0.5766   | 0.1738 | 0.1684 | 0.6140  |
| —    | SVM                 | 0.6487   | 0.6451   | 0.2691 | 0.2586 | 0.6907  |

★ = meilleur en accuracy, mais ce classement est **trompeur** (voir §3.2).

### 3.2 ⚠️ Alerte : l'accuracy masque un biais sévère

Le déséquilibre AwayWin (30,3 %) crée une **illusion d'optique** dans le classement par accuracy.

| Modèle              | Accuracy | AwayWin Recall | AwayWin F1 | Commentaire             |
|---------------------|----------|----------------|------------|-------------------------|
| Random Forest       | 72 %     | **13 %**       | 0.21       | Prédit presque jamais AW|
| Extra Trees         | 72 %     | 19 %           | 0.29       | Idem                    |
| CatBoost            | **73 %** | 29 %           | 0.40       | Légèrement mieux        |
| Gradient Boosting   | 71 %     | 25 %           | 0.35       | Pareil                  |
| LightGBM            | 72 %     | 29 %           | 0.38       | Proche CatBoost         |
| KNN                 | 68 %     | 31 %           | 0.37       | Plus équilibré          |
| **AdaBoost**        | 71 %     | **44 %**       | **0.49**   | ✅ Meilleur recall AW    |
| **SVM**             | 65 %     | 64 %           | 0.52       | ✅ Recall AW élevé       |
| **Logistic Regression**| 66 % | **67 %**      | **0.54**   | ✅ Meilleur équilibre    |

**Random Forest et Extra Trees se cachent derrière 72 % d'accuracy en prédisant "Not Away Win" dans 87–81 % des cas.** Leur balanced accuracy (0.55 et 0.57) trahit la réalité.

### 3.3 Deux philosophies de modèles

**Modèles "haute accuracy / faible recall"** (CatBoost, RF, ET, LGB, GB) :
- Accuracy 71–73 % (proche de la baseline naïve 69,7 %)
- AwayWin recall 13–29 %
- Stratégie implicite : éviter les faux positifs AwayWin → prudents mais inutiles pour détecter des victoires à l'extérieur

**Modèles "basse accuracy / haut recall"** (LogReg, SVM, AdaBoost) :
- Accuracy 65–71 %
- AwayWin recall 44–67 %
- Stratégie : acceptent plus de faux positifs pour capturer les vrais AwayWin
- **Ces modèles sont les plus utiles en pratique pour une application paris/prédiction**

### 3.4 Comparaison directe des deux meilleurs modèles équilibrés

**Logistic Regression** :
```
              Prédit ¬AW  Prédit AW
Réel ¬AW       2 389     1 279    → recall ¬AW = 65 %
Réel AW          528     1 065    → recall AW = 67 %
```
Modèle le plus équilibré. Balanced accuracy = 0.6599 (meilleure de tous). Accepte beaucoup de faux positifs (1 279) mais capture 67 % des vraies victoires à l'extérieur.

**AdaBoost** :
```
              Prédit ¬AW  Prédit AW
Réel ¬AW       3 050       618    → recall ¬AW = 83 %
Réel AW          885       708    → recall AW = 44 %
```
Compromis intermédiaire : balanced accuracy 0.6380, moins de faux positifs que LogReg.

### 3.5 Courbe d'entraînement AwayWin

| Modèle            | Best iter | Train logloss | Val logloss | Overfit |
|-------------------|-----------|---------------|-------------|---------|
| CatBoost          | 55        | 0.4965        | 0.5508      | ✅ Faible |
| Gradient Boosting | 23        | (accuracy) 0.7596 | 0.7141   | ✅ Modéré |
| LightGBM          | 28        | **0.4087**    | 0.5611      | ⚠️ Fort  |

LightGBM sur-entraîne encore plus sur AwayWin que HomeWin (écart train/val = 0.15 pts). CatBoost reste le plus stable.

---

## 4. Comparaison HomeWin vs AwayWin

| Dimension                    | HomeWin              | AwayWin              |
|------------------------------|----------------------|----------------------|
| Meilleur modèle (accuracy)   | CatBoost 66.5 %      | CatBoost 72.8 %      |
| Meilleur ROC-AUC             | CatBoost 0.714       | CatBoost 0.719       |
| Meilleur équilibre           | CatBoost ~pareil     | Logistic Regression  |
| Rappel de la classe positive | 55 % (CatBoost)      | 29 % (CatBoost)      |
| Baseline naïve               | 55.7 %               | 69.7 %               |
| Gain vs baseline             | +9 pts               | +3 pts (apparent)    |
| Difficulté relative          | Modérée              | Forte (déséquilibre) |

**HomeWin est nettement mieux résolu que AwayWin.** La victoire à domicile a des signaux plus stables (avantage terrain, meilleures features historiques pour les équipes dominantes). La victoire à l'extérieur reste difficile à anticiper, en partie parce que les équipes extérieures sont souvent moins représentées dans les features cumulatives.

---

## 5. ROC-AUC : vue d'ensemble

| Modèle              | AUC HomeWin | AUC AwayWin | Moyenne |
|---------------------|-------------|-------------|---------|
| CatBoost            | **0.7141**  | **0.7186**  | **0.716**|
| Logistic Regression | 0.7130      | 0.7157      | 0.714   |
| AdaBoost            | 0.7002      | 0.7100      | 0.705   |
| Gradient Boosting   | 0.7053      | 0.6992      | 0.702   |
| LightGBM            | 0.6965      | 0.6990      | 0.698   |
| SVM                 | 0.6942      | 0.6907      | 0.692   |
| Extra Trees         | 0.6986      | 0.6917      | 0.695   |
| Random Forest       | 0.6995      | 0.6835      | 0.691   |
| KNN                 | 0.6222      | 0.6140      | 0.618   |

**CatBoost et Logistic Regression** forment un duo de tête cohérent sur les deux tâches. Leurs AUC ~0.71 sont proches — différence non statistiquement significative sur 5 261 matchs.

---

## 6. Axes d'amélioration prioritaires

### 6.1 Traitement du déséquilibre (impact critique sur AwayWin)
```python
# Option 1 : pondération dans le modèle
CatBoostClassifier(class_weights=[1.0, 2.3])  # 69.7/30.3
XGBClassifier(scale_pos_weight=2.3)
LGBMClassifier(class_weight='balanced')

# Option 2 : rééchantillonnage
from imblearn.over_sampling import SMOTE
X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(X_train, y_away_train)
```

### 6.2 Seuil de décision optimisé
Le seuil par défaut (0.5) n'est pas optimal pour les classes déséquilibrées. Optimiser sur le F1-score ou la balanced accuracy :
```python
from sklearn.metrics import precision_recall_curve
# Trouver le seuil qui maximise F1 pour AwayWin
precision, recall, thresholds = precision_recall_curve(y_test, proba_away)
f1 = 2 * precision * recall / (precision + recall + 1e-8)
best_threshold = thresholds[np.argmax(f1)]  # typiquement ~0.35 pour AwayWin
```

### 6.3 Calibration des probabilités
CatBoost et LogReg ont de bonnes AUC mais leurs probabilités brutes peuvent être mal calibrées. Utiliser `CalibratedClassifierCV` en post-processing :
```python
from sklearn.calibration import CalibratedClassifierCV
calibrated = CalibratedClassifierCV(catboost_model, method='isotonic', cv='prefit')
calibrated.fit(X_val, y_val)
```

### 6.4 Correction de l'overfitting LightGBM
LightGBM sur-entraîne sur les deux tâches (best_iter ~26–28, fort écart train/val) :
```yaml
# Dans configBT_1.yaml → lightgbm
n_estimators: 200
min_child_samples: 50     # augmenter (défaut 20)
reg_alpha: 0.1            # L1
reg_lambda: 0.1           # L2
subsample: 0.8
colsample_bytree: 0.8
```

### 6.5 Features manquantes (Phase 3 roadmap)
- **ELO ratings** : différentiel ELO particulièrement utile pour prédire AwayWin (choc de niveaux)
- **Head-to-head récent** : certaines équipes dominent historiquement leur adversaire même à l'extérieur
- **Fatigue** : les équipes en déplacement consécutif perdent plus souvent

### 6.6 Modèle dédié AwayWin
Les signatures statistiques de HomeWin et AwayWin sont différentes. Considérer :
- Features asymétriques (différentiels calculés dans le bon sens selon le rôle H/A)
- Modèle séparé pour AwayWin avec features spécifiques aux équipes "voyageuses"

---

## 7. Recommandations par cas d'usage

| Usage                                  | Modèle recommandé HomeWin | Modèle recommandé AwayWin |
|----------------------------------------|---------------------------|---------------------------|
| Prédiction de victoire (confiance)     | CatBoost                  | CatBoost (mais calibrer)  |
| Détection maximale des victoires AW    | CatBoost                  | **Logistic Regression**   |
| Équilibre précision / rappel           | CatBoost                  | **AdaBoost**              |
| Ranking probabiliste (paris)           | CatBoost / LogReg          | CatBoost / LogReg         |
| Baseline robuste simple                | Logistic Regression        | Logistic Regression       |
| À éviter                               | KNN                        | Random Forest, Extra Trees|

---

## 8. Synthèse

| Tâche    | Meilleur modèle | Accuracy | ROC-AUC | Rappel positif | Verdict       |
|----------|-----------------|----------|---------|----------------|---------------|
| HomeWin  | CatBoost        | 66.5 %   | 0.714   | 55 %           | ✅ Satisfaisant |
| AwayWin* | CatBoost        | 72.8 %   | 0.719   | 29 %           | ⚠️ Trompeur    |
| AwayWin† | Logistic Reg.   | 65.7 %   | 0.716   | 67 %           | ✅ Plus honnête |

\* meilleur en accuracy brute
† meilleur en balanced accuracy et recall AwayWin

> **Conclusion** : La prédiction HomeWin est opérationnelle avec CatBoost (ROC-AUC 0.71, recall HW 55 %). La prédiction AwayWin nécessite impérativement un traitement du déséquilibre de classes et l'optimisation du seuil de décision — sans cela, les modèles de haute accuracy sont inutilisables pour détecter les vraies victoires à l'extérieur. Logistic Regression et AdaBoost sont les références les plus honnêtes pour AwayWin dans l'état actuel.
