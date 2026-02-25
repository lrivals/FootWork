# Analyse des résultats — Prédiction Multiclasse (HomeWin / Draw / AwayWin)
> Run : `multiclass_prediction2026-02-24_12-35-41` · Split temporel : train < 2022 / test ≥ 2022 · Test : **5 261 matchs**

---

## 1. Contexte et distribution des classes

| Classe   | Support test | Part (%) | Baseline naïve (majority vote) |
|----------|-------------|----------|-------------------------------|
| HomeWin  | 2 330       | 44,3 %   | —                              |
| AwayWin  | 1 593       | 30,3 %   | —                              |
| Draw     | 1 338       | 25,4 %   | —                              |
| **Total**| **5 261**   | 100 %    | **44,3 % (tout HomeWin)**      |

Le déséquilibre est modéré mais réel : HomeWin représente presque la moitié des matchs, le Draw le quart. Tout modèle qui prédit systématiquement HomeWin obtient **44,3 % d'accuracy** sans rien apprendre.

---

## 2. Tableau de classement global

| Rang | Modèle              | Accuracy | Bal. Acc | MCC    | Kappa  | Macro AUC |
|------|---------------------|----------|----------|--------|--------|-----------|
| 1    | CatBoost            | 0.5315   | 0.4592   | 0.2517 | 0.2252 | **0.6676**|
| 1    | AdaBoost            | 0.5315   | **0.4602**| **0.2532**| **0.2273**| 0.6533|
| 3    | Gradient Boosting   | 0.5250   | 0.4641   | 0.2404 | 0.2277 | 0.6588    |
| 4    | Logistic Regression | 0.5233   | 0.4610   | 0.2376 | 0.2235 | 0.6666    |
| 5    | Extra Trees         | 0.5191   | 0.4544   | 0.2292 | 0.2131 | 0.6451    |
| 6    | XGBoost             | 0.5151   | 0.4573   | 0.2255 | 0.2149 | 0.6532    |
| 7    | Random Forest       | 0.5136   | 0.4477   | 0.2182 | 0.2036 | 0.6458    |
| 7    | LightGBM            | 0.5136   | 0.4559   | 0.2220 | 0.2124 | 0.6546    |
| 9    | SVM                 | 0.4735   | 0.4656   | 0.2119 | 0.2091 | 0.6508    |
| 10   | KNN                 | 0.4421   | 0.4083   | 0.1273 | 0.1270 | 0.5847    |

**Observations générales :**
- La fourchette d'accuracy va de **44,2 % à 53,2 %** — le meilleur modèle dépasse la baseline naïve de seulement **+9 pts**.
- Le MCC oscille entre **0.13 et 0.25** : une corrélation faible à modérée. La tâche est intrinsèquement difficile.
- Tous les modèles (sauf KNN) ont un Macro AUC > 0.65, ce qui confirme qu'ils apprennent effectivement une information discriminante sur l'ensemble des classes, même si la décision finale reste incertaine.

---

## 3. Le problème critique : le Draw

C'est le point d'échec le plus flagrant de cette expérience.

| Modèle              | Draw Recall | Draw Precision | Draw F1 |
|---------------------|-------------|----------------|---------|
| SVM                 | **0.38**    | 0.29           | **0.33**|
| KNN                 | 0.23        | 0.26           | 0.25    |
| LightGBM            | 0.13        | 0.32           | 0.19    |
| Gradient Boosting   | 0.11        | 0.34           | 0.17    |
| Logistic Regression | 0.09        | 0.33           | 0.14    |
| Extra Trees         | 0.07        | 0.33           | 0.12    |
| XGBoost             | 0.10        | 0.31           | 0.16    |
| Random Forest       | 0.08        | 0.27           | 0.12    |
| CatBoost            | **0.01**    | 0.38           | 0.02    |
| AdaBoost            | **≈0.00**   | 0.16           | 0.00    |

**Le Draw est quasi ignoré par les meilleurs modèles en accuracy.** CatBoost ne prédit correctement que 15 matchs nuls sur 1 338 (recall = 1 %). AdaBoost en prédit 3.

Ce phénomène s'explique par :
1. **Déséquilibre des classes** : le Draw est la classe minoritaire (25,4 %) et sa pénalité d'erreur est faible face à HomeWin (44,3 %).
2. **Difficulté intrinsèque** : le Draw est le résultat le plus difficile à anticiper en football — les bookmakers eux-mêmes le prédisent moins bien que les victoires.
3. **Absence de pondération de classe** (class_weight) dans les hyperparamètres actuels.

**Le SVM fait exception** : il accepte de sacrifier de l'accuracy globale (47,4 %) pour prédire le Draw de façon plus équilibrée (recall 38 %, F1 = 33 %). Sa balanced accuracy (0.4656) est la **meilleure de tous les modèles** — preuve qu'il capture mieux la distribution réelle.

---

## 4. AUC par classe (OvR)

| Modèle              | AUC AwayWin | AUC Draw | AUC HomeWin |
|---------------------|-------------|----------|-------------|
| CatBoost            | **0.7192**  | **0.5683**| **0.7148** |
| Logistic Regression | 0.7161      | 0.5707   | 0.7127      |
| AdaBoost            | 0.7109      | 0.5427   | 0.7061      |
| Gradient Boosting   | 0.7016      | 0.5665   | 0.7078      |
| LightGBM            | 0.7010      | 0.5624   | 0.7000      |
| XGBoost             | 0.6995      | 0.5599   | 0.6998      |
| SVM                 | 0.6971      | 0.5570   | 0.6979      |
| Extra Trees         | 0.6945      | 0.5410   | 0.6996      |
| Random Forest       | 0.6944      | 0.5436   | 0.6993      |
| KNN                 | 0.6140      | 0.5174   | 0.6222      |

**Constats :**
- **AwayWin et HomeWin** : AUC ~0.71–0.72 pour les meilleurs modèles — la discrimination est raisonnable.
- **Draw** : AUC ~0.57 pour tous, proche du hasard (0.50). Indique que les scores de probabilité des modèles ne séparent pas vraiment les matchs nuls des autres. C'est cohérent avec le recall proche de zéro.
- La **régression logistique** obtient la meilleure AUC Draw (0.5707) malgré un recall Draw bas — ses probabilités sont plus calibrées.

---

## 5. Analyse des courbes d'entraînement (boosters)

| Modèle            | Métrique      | Best iter | Train final | Val final | Overfit |
|-------------------|---------------|-----------|-------------|-----------|---------|
| XGBoost           | mlogloss      | 28        | 0.7460      | 1.0018    | ⚠️ Fort |
| LightGBM          | multi_logloss | 27        | 0.7161      | 1.0019    | ⚠️ Fort |
| CatBoost          | MultiClass    | 1         | 0.9258      | 0.9836    | ❓       |
| Gradient Boosting | accuracy      | 55        | 0.5704      | 0.5250    | ✅ Modéré|

**XGBoost et LightGBM** : L'early stopping intervient très tôt (iter 27–28). La loss de validation (~1.00) est bien supérieure à la loss d'entraînement (~0.72–0.75) → **overfitting significatif**. Ces modèles mémorisent des patterns du train qui ne généralisent pas.

**CatBoost** : Le best_iter=1 est probablement un artefact de l'interprétation de la métrique "MultiClass" (traitée comme une accuracy dans le code au lieu d'une loss → argmax au lieu d'argmin). À vérifier. Les valeurs finales (0.92/0.98) semblent être une logloss, pas une accuracy.

**Gradient Boosting** : Le plus sain — best_iter=55, écart train/val contenu (0.5704 vs 0.5250). Pas d'early stopping abrupt.

---

## 6. Analyse des matrices de confusion

### CatBoost (meilleur en accuracy)
```
          AwayWin  Draw  HomeWin
AwayWin     871    11     711    ← 55 % recall
Draw        457    15     866    ← 1 % recall  ← PROBLÈME
HomeWin     406    14    1910    ← 82 % recall
```
Le modèle **sur-prédit HomeWin** massivement. 866 draws sont classés HomeWin, 457 sont classés AwayWin, seulement 15 sont correctement identifiés comme draws.

### SVM (meilleur en balanced accuracy)
```
          AwayWin  Draw  HomeWin
AwayWin     839    493    261    ← 53 % recall
Draw        457    506    375    ← 38 % recall  ← MEILLEUR
HomeWin     447    737   1146    ← 49 % recall
```
Le SVM distribue les erreurs plus équitablement. Il confond beaucoup HomeWin avec Draw (737 matchs), ce qui est cohérent avec sa meilleure détection des nuls.

---

## 7. Classement recommandé selon la métrique prioritaire

| Priorité                              | Meilleur modèle       |
|---------------------------------------|-----------------------|
| Accuracy brute                        | CatBoost / AdaBoost   |
| Discrimination probabiliste (AUC)     | CatBoost              |
| Équilibre entre classes               | **SVM**               |
| Prédiction du Draw                    | **SVM**               |
| MCC (corrélation globale)             | AdaBoost              |
| Confiance calibrée (proba)            | Logistic Regression   |
| À éviter                              | KNN                   |

---

## 8. Axes d'amélioration prioritaires

### 8.1 Rééquilibrage des classes (impact très fort attendu)
```python
# Exemple pour XGBoost / CatBoost
class_weight = {0: 1.0, 1: 1.74, 2: 1.0}  # boost Draw (1338 < 2330)
# Ou : class_weight='balanced'
```
Sans correction du déséquilibre, les modèles resteront biaisés vers HomeWin.

### 8.2 Seuils de décision asymétriques
Au lieu d'argmax sur les probabilités, définir des seuils par classe :
- Draw prédit si P(draw) > 0.25 (au lieu du seuil par défaut ~0.33)
- Optimiser les seuils sur F1-macro ou Kappa

### 8.3 Calibration des probabilités
Les AUC Draw sont ~0.57 mais le recall Draw est ~0–38 %. Les probabilités ne sont pas bien calibrées → utiliser `CalibratedClassifierCV` (Platt scaling ou isotonic regression).

### 8.4 Régularisation des boosters (overfitting XGB/LGB)
- Réduire `max_depth`, augmenter `min_child_weight`
- Augmenter `reg_alpha` / `reg_lambda`
- Laisser l'early stopping gérer le nombre d'itérations correctement

### 8.5 Features manquantes (Phase 3 roadmap)
- **ELO ratings** : capture la force relative en continu
- **Head-to-head** : certains affrontements sont structurellement plus propices aux nuls
- **Fatigue / densité de calendrier** : corrélée aux draws
- **Draw propensity** actuelle → à affiner avec des fenêtres plus longues

### 8.6 Ensemble / Stacking
Combiner les forces de CatBoost (AUC) et SVM (Draw recall) via un stacking de niveau 2 pourrait améliorer significativement la balanced accuracy.

---

## 9. Synthèse

| Dimension            | Résultat                        | Verdict      |
|----------------------|---------------------------------|--------------|
| Accuracy globale     | 53 % (baseline naïve : 44 %)   | ✅ Modeste mais réel  |
| Macro AUC            | 0.67 (hasard : 0.50)           | ✅ Signal solide       |
| Prédiction HomeWin   | Recall 75–82 % (meilleurs)     | ✅ Fonctionnel         |
| Prédiction AwayWin   | Recall 47–57 % (meilleurs)     | ⚠️ Acceptable         |
| Prédiction Draw      | Recall 0–38 %                  | ❌ Critique            |
| Overfitting boosters | XGBoost / LightGBM (best iter 27–28) | ⚠️ À corriger |
| Généralisation       | Écart acc train/val modéré      | ✅ Correct             |

> **Conclusion** : Les modèles apprennent effectivement à distinguer les issues footballistiques (AUC > 0.65 pour tous), mais le Draw reste le verrou principal. La priorité pour la prochaine itération est le **rééquilibrage des classes** + **calibration des probabilités** + **features ELO/h2h** spécifiquement utiles pour les nuls.
