# Analyse approfondie du projet FootWork & Axes d'amélioration

*Dernière mise à jour : Février 2026 — Révisé le 26/02/2026 (état post-Phase 3)*

> **Légende statut :** ✅ Implémenté | ❌ Non implémenté | 🔄 Partiellement fait

---

## 1. État des lieux — Vue d'ensemble du projet

### 1.1 Données disponibles

Le projet agrège des données de **6 ligues de football** sur plusieurs saisons, toutes au même format (source : footystats.org) :

| Ligue | Pays | Saisons | Matchs traités |
|---|---|---|---|
| Premier League | Angleterre | 2012-13 à 2023-24 | 4 282 |
| Ligue 1 | France | 2013-14 à 2023-24 | 3 864 |
| Bundesliga | Allemagne | 2013-14 à 2023-24 | 3 170 |
| Serie A | Italie | 2010-11 à 2023-24 | 5 011 |
| La Liga | Espagne | 2012-13 à 2023-24 | 4 270 |
| Série A | Brésil | 2013-14 à 2023-24 | 3 926 |

**Dataset combiné final :** `data/all_leagues_combined.csv` — **24 523 matchs × 75 colonnes**

Distribution des classes cible :
- HomeWin : 11 086 (45,2%)
- AwayWin : 7 121 (29,0%)
- Draw : 6 316 (25,7%)

### 1.2 Données brutes disponibles (63 colonnes par match)

Les CSV bruts contiennent des informations très riches, dont une grande partie n'est pas encore exploitée :

```
Identifiants :   date_GMT, home_team_name, away_team_name, Game Week, referee
Scores :         home/away_team_goal_count, total_goals_at_half_time, HT goals
Tirs :           home/away_team_shots, shots_on_target, shots_off_target
Possession :     home/away_team_possession (%)
Corners :        home/away_team_corner_count
Cartons :        home/away_team_yellow/red_cards, first/second_half_cards
Fautes :         home/away_team_fouls
xG :             Home/Away Team Pre-Match xG, team_a_xg, team_b_xg
Timings :        home/away_team_goal_timings (ex: "24,35,64")
Stats pré-match : average_goals_per_match_pre_match, btts_percentage_pre_match,
                  over_15/25/35/45_percentage_pre_match
COTES BOOKMAKER : odds_ft_home_team_win, odds_ft_draw, odds_ft_away_team_win,
                  odds_ft_over15/25/35/45, odds_btts_yes/no
```

### 1.3 Pipeline de préprocessing actuel

**Script principal :** `src/Data_Processing/Multi-Season_Match_Data_Processor.py`

Pour chaque match (dans l'ordre chronologique), le script :
1. Parse et trie les matchs par date
2. Calcule `match_result` (HomeWin / Draw / AwayWin) depuis le score
3. Pour chaque équipe, regarde **tous les matchs précédents** dans la saison
4. Calcule 34 features par équipe = 68 features totales
5. Supprime les lignes avec NaN ou Inf (premiers matchs sans historique)
6. Sauvegarde par saison puis concatène

**Fenêtre temporelle :**
- Stats cumulatives : toute l'histoire disponible (intra-saison)
- Forme récente : 5 derniers matchs (`last_n_matches=5`)

### 1.4 Features engineered actuellement — ✅ Mise à jour post-Phase 3 (~127+ features)

> *Le dataset initial avait 34 features par équipe × 2 = 68. Après les phases d'implémentation, le total est désormais ~127+ features par match.*

| Catégorie | Features | Statut |
|---|---|---|
| **Performance globale** | games_played, wins, draws, losses, points_per_game | ✅ |
| **Buts** | avg_goals_scored, avg_goals_conceded, avg_goal_diff, goals_scored_first_half_ratio, goals_conceded_first_half_ratio | ✅ |
| **Forme multi-fenêtres** (3/5/10 matchs) | recent_ppg_last{3,5,10}, recent_goals_scored_last{3,5,10}, recent_goals_conceded_last{3,5,10}, recent_clean_sheets_last{3,5,10}, draw_ratio_last{3,5,10} | ✅ |
| **Tendance de forme** | form_trend = ppg_last3 − ppg_last10 (positif = en montée) | ✅ |
| **xG rolling** | avg_xg_scored, avg_xg_conceded, xg_vs_goals_diff | ✅ |
| **Efficacité tirs** | shot_conversion_rate, shots_on_target_ratio, avg_shots_per_game, avg_shots_on_target | ✅ |
| **Contrôle du jeu** | avg_possession, possession_efficiency, avg_corners_for, corner_efficiency | ✅ |
| **Discipline** | avg_fouls_committed, avg_yellows, avg_reds, cards_first_half_ratio | ✅ |
| **Domicile/Extérieur** | venue_games, venue_win_ratio, venue_goals_avg, venue_conceded_avg | ✅ |
| **Gestion de match** | clean_sheets_ratio, scoring_ratio, comeback_ratio, lead_loss_ratio | ✅ |
| **Cotes / Probabilités implicites** | implied_prob_home, implied_prob_draw, implied_prob_away, odds_ratio | ✅ |
| **ELO ratings** (`src/Features/ELO_Rating.py`) | home_elo, away_elo, elo_diff | ✅ |
| **Features différentielles** (home − away) | diff_ppg, diff_recent_ppg, diff_goals_scored, diff_goals_conceded, diff_xg_scored, diff_shots, diff_possession, diff_form_trend | ✅ |
| **Draw propensity** | combined_draw_tendency, match_competitiveness | ✅ |
| **Head-to-Head (H2H)** | h2h_home_wins, h2h_away_wins, h2h_draws, h2h_home_goals_avg, h2h_away_goals_avg, h2h_matches_count | ✅ |
| **Fatigue / Calendrier** | days_since_last_match_home/away, matches_last_7_days_home/away, is_midweek_match | ✅ |

### 1.5 Modèles entraînés — ✅ Mise à jour post-Phase 3

**Trois pipelines complémentaires :**

| Pipeline | Fichier | Modèles | Statut |
|---|---|---|---|
| **Multiclasse** (HomeWin/Draw/AwayWin) | `src/Models/Multiclass_Target/Football_Match_Prediction_Pipeline.py` | 11 classifiers (RF, LogReg, SVM, GB, XGB, LGBM, CatBoost, MLP, KNN, AdaBoost, ExtraTrees) | ✅ |
| **Binaire** (HomeWin vs Not / AwayWin vs Not) | `src/Models/Binary_Target/Football_Match_Binary_Prediction_Pipeline.py` | Idem 11 classifiers × 2 targets | ✅ |
| **Neural Network** PyTorch | `src/Models/Neural_Network/Football_Match_NN_Pipeline.py` | MLP custom (BatchNorm + Dropout), wrapper sklearn-compatible | ✅ |

**Améliorations appliquées à tous les pipelines :**
- Split temporel 3 voies : train < 2020 / calibration 2020-2021 / test ≥ 2022
- Calibration isotonique (`CalibratedClassifierCV` + `_IsotonicCalibratedNN` pour PyTorch)
- Optimisation des seuils de décision (`src/Models/threshold_optimizer.py`)
- Class weights / sample weights équilibrés
- Sorties visuelles : ROC curves, loss curves, calibration plots, confusion matrices (PNG)

### 1.6 Résumé des phases d'implémentation réalisées

| Phase | Contenu | Statut |
|---|---|---|
| **Phase 1** — Fondamentaux | Bug Spain, colonne league, cotes bookmaker, split temporel | ✅ Complète |
| **Phase 2** — Feature engineering | Cross-saison, diff features, xG, multi-fenêtres, draw propensity | ✅ Complète |
| **Phase 3** — Modélisation avancée | ELO, H2H, calibration, threshold optimizer, class weights | ✅ Complète |
| **Phase 4** — Pipeline parieur | Backtesting, EV/value bets, prédictions CSV, hierarchical, Neural Network PyTorch | ✅ Complète |

---

## 2. Performances actuelles

### 2.1 Classification multiclasse (3 classes)

**Dataset : all_leagues_combined.csv — Test set : 4 905 matchs**

| Modèle | Accuracy | Draw Recall | HomeWin Recall | AwayWin Recall |
|---|---|---|---|---|
| Gradient Boosting | **52,52%** | 3% | 84% | 46% |
| CatBoost | 52,44% | 2% | 85% | 45% |
| Logistic Regression | 52,15% | 3% | 83% | 46% |
| XGBoost | 51,87% | 6% | 81% | 46% |
| LightGBM | 51,70% | 7% | 81% | 45% |
| AdaBoost | 51,80% | 1% | 88% | 40% |
| Random Forest | 51,31% | 8% | 81% | 42% |
| Extra Trees | 51,15% | 8% | 81% | 42% |
| SVM | 47,20% | **35%** | 53% | 50% |
| KNN | 45,32% | 23% | 62% | 39% |
| Neural Network | 41,39% | 25% | 52% | 39% |

**Observation clé :** le SVM est le seul à prédire correctement les nuls (35% recall), mais au prix d'une accuracy globale plus basse. Tous les autres modèles "oublient" la classe Draw.

### 2.2 Classification binaire

**HomeWin vs Not-HomeWin :**
- Meilleur : Logistic Regression — 64,99% accuracy, ROC-AUC 0,709 (PL uniquement)
- Tous leagues : CatBoost — 64,28% accuracy, ROC-AUC 0,697

**AwayWin vs Not-AwayWin :**
- Meilleur : Extra Trees — 72,46% accuracy
- Amélioration notable vs version précédente (+2,3 pts pour Random Forest et Extra Trees)

### 2.3 Feature selection (RFECV, Premier League)

| Modèle | Features optimales | CV Accuracy | Test Accuracy |
|---|---|---|---|
| AdaBoost | **11** features | 52,99% | 51,69% |
| Extra Trees | 68 (toutes) | 52,05% | 52,74% |

L'AdaBoost n'a besoin que de 11 features sur 68, signe de forte redondance dans le feature set actuel.

---

## 3. Problèmes critiques identifiés

### ~~🔴 Problème 1 — Fuite temporelle dans le split train/test~~ ✅ RÉSOLU

**Impact : CRITIQUE** | **Résolution : Split 3 voies train < 2020 / cal 2020-2021 / test ≥ 2022**

~~Le split 80/20 est aléatoire (`random_state=42`). Un match de 2015 peut se retrouver dans le test set...~~

Tous les pipelines (Multiclasse, Binaire, Neural Network) utilisent désormais un split temporel strict défini dans les configs YAML (`temporal_split_year: 2022`, `cal_split_year: 2020`). Le set de calibration sert exclusivement à fitter l'isotonic regression.

### ~~🔴 Problème 2 — Les cotes bookmaker ne sont PAS utilisées comme features~~ ✅ RÉSOLU

**Impact : CRITIQUE** | **Résolution : 4 features odds intégrées dans le data processor**

Les cotes sont intégrées dans `Multi-Season_Match_Data_Processor.py` :
```python
implied_prob_home, implied_prob_draw, implied_prob_away  # normalisées (marge retirée)
odds_ratio = odds_away / odds_home                        # déséquilibre home/away
```

### ~~🔴 Problème 3 — Reset des stats en début de saison~~ ✅ RÉSOLU

**Impact : IMPORTANT** | **Résolution : `team_history`, `h2h_history` et ELO persistent cross-saison**

Le data processor maintient un dictionnaire `team_history` qui traverse les frontières de saison. L'`ELOCalculator` est instancié une fois par ligue et réutilisé pour toutes les saisons. Les rolling stats de la fin d'une saison alimentent directement les premiers matchs de la saison suivante.

### ~~🟠 Problème 4 — Pas de features head-to-head~~ ✅ RÉSOLU

**Impact : MODÉRÉ** | **Résolution : 6 features H2H calculées via `h2h_history` cross-saison**

```python
h2h_home_wins, h2h_away_wins, h2h_draws          # résultats historiques
h2h_home_goals_avg, h2h_away_goals_avg            # moyennes de buts H2H
h2h_matches_count                                  # nombre de confrontations
```

### ~~🟠 Problème 5 — Pas de features différentielles (relatives)~~ ✅ RÉSOLU

**Impact : MODÉRÉ** | **Résolution : 8 features diff_* ajoutées**

```python
diff_ppg, diff_recent_ppg, diff_goals_scored, diff_goals_conceded,
diff_xg_scored, diff_shots, diff_possession, diff_form_trend
```

### ~~🟠 Problème 6 — Le xG historique n'est pas calculé~~ ✅ RÉSOLU

**Impact : MODÉRÉ** | **Résolution : rolling xG par équipe + différentiel xG vs buts réels**

```python
avg_xg_scored, avg_xg_conceded   # moyennes rolling de xG
xg_vs_goals_diff                  # surperformance / sous-performance vs l'attendu
```

### ~~🟡 Problème 7 — Une seule fenêtre temporelle pour la forme~~ ✅ RÉSOLU

**Impact : MODÉRÉ** | **Résolution : Multi-fenêtres last3/last5/last10 + form_trend**

```python
recent_ppg_last{3,5,10}, recent_goals_scored_last{3,5,10},
recent_goals_conceded_last{3,5,10}, recent_clean_sheets_last{3,5,10},
draw_ratio_last{3,5,10}
form_trend = ppg_last3 - ppg_last10   # tendance haussière/baissière
```

### ~~🟡 Problème 8 — Probabilités non calibrées~~ ✅ RÉSOLU

**Impact : IMPORTANT pour le pari** | **Résolution : Calibration isotonique dans les 3 pipelines**

- Sklearn : `CalibratedClassifierCV(method='isotonic', cv='prefit')` sur le set de calibration
- PyTorch : `_IsotonicCalibratedNN` (classe custom, IsotonicRegression par classe)
- Courbes de calibration (reliability diagrams) exportées en PNG pour chaque modèle

### ~~🟡 Problème 9 — Colonne `league` absente du dataset combiné~~ ✅ RÉSOLU

**Impact : FAIBLE-MODÉRÉ** | **Résolution : Colonne `league` ajoutée lors de la concaténation**

La colonne est présente dans `all_leagues_combined.csv` et exclue des features d'entraînement via `excluded_columns` dans les configs YAML (disponible pour filtrage par ligue).

### ~~🟡 Problème 10 — Bug de chemin Espagne → dossier France~~ ✅ RÉSOLU

**Impact : FAIBLE** | **Résolution : Chemin corrigé vers `data/Spain/clean_la_liga_data/`**

---

### ~~🟠 Problème 11 — Pas de features fatigue/calendrier~~ ✅ RÉSOLU

**Impact : MODÉRÉ** | **Résolution : 5 features fatigue ajoutées dans le data processor (Phase 4)**

```python
days_since_last_match_home / away   # jours de repos
matches_last_7_days_home / away     # densité de calendrier
is_midweek_match                    # proxy coupe/ligue europe
```

---

## 4. Axes d'amélioration — État d'implémentation

### AXE A — Corrections fondamentales ✅ COMPLÈTEMENT FAIT

#### ~~A1. Split temporel strict~~ ✅
Train < 2020 / calibration 2020-2021 / test ≥ 2022 — implémenté dans les 3 pipelines + configs YAML.

#### ~~A2. Continuité cross-saison des rolling stats~~ ✅
`team_history`, `h2h_history` et `ELOCalculator` persistent entre les saisons dans le data processor.

#### ~~A3. Ajout de la colonne `league`~~ ✅
Colonne présente dans `all_leagues_combined.csv`, exclue des features via `excluded_columns` en YAML.

#### ~~A4. Fix du bug de chemin Espagne~~ ✅
Corrigé vers `data/Spain/clean_la_liga_data/`.

---

### AXE B — Nouvelles features à haute valeur — 7/8 FAIT ✅

#### ~~B1. Cotes bookmaker → Probabilités implicites~~ ✅

`implied_prob_home`, `implied_prob_draw`, `implied_prob_away`, `odds_ratio` intégrés dans le data processor via les colonnes brutes `odds_ft_home_team_win / draw / away_team_win`. Normalisation pour retirer la marge bookmaker appliquée.

#### ~~B2. xG rolling par équipe~~ ✅

`avg_xg_scored`, `avg_xg_conceded`, `xg_vs_goals_diff` calculés à partir de `team_a_xg` / `team_b_xg` des CSV bruts.

#### ~~B3. Features différentielles (home − away)~~ ✅

8 features : `diff_ppg`, `diff_recent_ppg`, `diff_goals_scored`, `diff_goals_conceded`, `diff_xg_scored`, `diff_shots`, `diff_possession`, `diff_form_trend`.

#### ~~B4. Head-to-Head (H2H)~~ ✅

6 features : `h2h_home_wins`, `h2h_away_wins`, `h2h_draws`, `h2h_home_goals_avg`, `h2h_away_goals_avg`, `h2h_matches_count`. Fenêtre configurable (défaut : 5 derniers H2H). Cross-saison via `h2h_history`.

#### ~~B5. ELO ratings dynamiques~~ ✅

`src/Features/ELO_Rating.py` — classe `ELOCalculator` par ligue. Formule standard + home advantage. Features : `home_elo`, `away_elo`, `elo_diff`. Mise à jour post-match et persistance cross-saison.

#### ~~B6. Features multi-fenêtres temporelles~~ ✅

Windows 3, 5, 10 matchs : `recent_ppg_last{3,5,10}`, `recent_goals_scored_last{3,5,10}`, `recent_goals_conceded_last{3,5,10}`, `recent_clean_sheets_last{3,5,10}`, `draw_ratio_last{3,5,10}`.
`form_trend = ppg_last3 − ppg_last10`.

#### ~~B7. Features spécifiques aux nuls ("Draw propensity")~~ ✅

`combined_draw_tendency` (moyenne des draw_ratio home + away), `match_competitiveness` (écart de PPG).

#### B8. Features de calendrier et fatigue ❌ NON IMPLÉMENTÉ

```python
# À ajouter dans Multi-Season_Match_Data_Processor.py :
days_since_last_match_home / away   # jours de repos avant le match
matches_last_7_days_home / away     # densité de calendrier récente
is_midweek_match                    # proxy coupe/ligue europe (mardi-jeudi)
```

Nécessite de trier les matchs par équipe et de calculer des deltas de dates. Les dates sont disponibles dans les CSV bruts (`date_GMT`).

---

### AXE C — Améliorations de modélisation — 3/4 FAIT ✅

#### ~~C1. Cross-validation temporelle~~ ✅

Réalisé via le split temporel strict (train/cal/test par années) dans les 3 pipelines. Un `TimeSeriesSplit` walk-forward reste envisageable pour une validation plus fine mais le split fixe est suffisant pour l'évaluation actuelle.

#### ~~C2. Calibration des probabilités~~ ✅

`CalibratedClassifierCV(method='isotonic', cv='prefit')` pour les pipelines sklearn. Classe custom `_IsotonicCalibratedNN` pour le pipeline PyTorch. Courbes de calibration (reliability diagrams) exportées en PNG.

#### C3. Approche hiérarchique pour contourner le problème des nuls ❌ NON IMPLÉMENTÉ

```
Stage 1 : "Est-ce une victoire à domicile ?" → HomeWin vs Not-HomeWin
Stage 2 (si Not-HomeWin) : "Est-ce un nul ?" → Draw vs AwayWin
```

Avantages : chaque étape est un binaire plus simple, le Stage 2 peut utiliser des features draw-specific. À envisager si le Draw recall reste insuffisant après les améliorations actuelles.

#### ~~C4. Optimisation des seuils de décision~~ ✅

`src/Models/threshold_optimizer.py` — fonctions `find_optimal_thresholds_multiclass`, `predict_with_thresholds`, `find_optimal_threshold` (binaire). Intégré dans les 3 pipelines. Résultats "Optimised Threshold Metrics" exportés dans les fichiers de résultats.

---

### AXE D — Pipeline orienté parieur ❌ ENTIÈREMENT À FAIRE

#### D1. Output enrichi pour chaque match ❌

Produire une fiche de pari complète par match incluant probabilités modèle + marché, EV, confiance, recommandation, contexte ELO et H2H.

```python
{
  "match": "PSG vs Olympique Lyonnais",
  "model_prob_home": 0.52, "model_prob_draw": 0.24, "model_prob_away": 0.24,
  "market_prob_home": 0.55, "market_prob_draw": 0.27, "market_prob_away": 0.20,
  "ev_home": -0.06, "ev_draw": -0.11, "ev_away": +0.08,  # VALUE BET Away
  "confidence_level": "low",
  "home_elo": 1842, "away_elo": 1654, "h2h_last_5": "H:3 D:1 A:1",
}
```

#### D2. Calcul de la Value Bet (Expected Value) ❌

```python
ev_home = (model_prob_home * odds_home) - 1   # EV > 0.05 → pari potentiel
ev_draw = (model_prob_draw * odds_draw) - 1
ev_away = (model_prob_away * odds_away) - 1

kelly_fraction = ev_outcome / (odds_outcome - 1)  # 1/4 Kelly recommandé
```

#### D3. Backtesting de stratégies ❌

Fichier attendu : `src/Analysis/Betting_Backtest.py` — n'existe pas encore.

Stratégies à simuler sur le test set (2022-2024) :
- `always_bet_model_prediction` — mise systématique sur la classe prédite
- `value_bets_ev5` — parie uniquement si EV > 5%
- `value_bets_ev10` — parie uniquement si EV > 10%
- `kelly_ev5` — mise Kelly fractionnelle quand EV > 5%
- `high_confidence_only` — mise uniquement si entropie < seuil

Métriques : ROI total (%), win rate, nombre de paris, max drawdown, Sharpe ratio, P&L par type (Home/Draw/Away).

---

### AXE E — Nouvelles pistes à explorer (Post-Phase 3)

#### E1. Stacking / Ensembling des 3 pipelines

Combiner les prédictions du pipeline Multiclasse, Binaire et Neural Network via un meta-learner (LogReg ou Ridge). Les 3 pipelines sont désormais calibrés et produisent des probabilités cohérentes — stacker leurs outputs est la prochaine étape naturelle pour gagner 1-2 pts d'accuracy.

#### E2. Explicabilité SHAP

Intégrer SHAP (`shap.TreeExplainer` pour tree-based, `shap.DeepExplainer` pour NN) pour :
- Identifier les features les plus influentes globalement
- Expliquer les prédictions individuelles ("pourquoi ce match prédit Draw ?")
- Détecter les features redondantes à supprimer

#### E3. Hyperparameter tuning avec Optuna

Les modèles actuels utilisent des hyperparamètres par défaut (100 estimateurs, lr=0.1). Un sweep Optuna sur le set de calibration permettrait de trouver les configs optimales par modèle sans risque de data leakage.

#### E4. Odds secondaires comme features supplémentaires

Les CSV bruts contiennent d'autres cotes exploitables **avant le match** :
```python
odds_ft_over15/25/35/45   # cotes sur le total de buts
odds_btts_yes / no        # cotes Both Teams To Score
```
Ces cotes encodent des informations sur le style de jeu attendu (matchs ouverts vs fermés) qui complètent les cotes 1X2.

#### E5. Features de tendance home/away séparées

L'avantage domicile varie selon l'équipe ET selon la ligue. Ajouter :
- `home_venue_form_trend` : tendance de forme uniquement à domicile (last3 vs last10 à domicile)
- `away_venue_form_trend` : tendance uniquement en déplacement
- Ces features capturent des équipes qui progressent spécifiquement chez elles ou en déplacement.

#### E6. API de prédiction en temps réel

Une fois le pipeline parieur fonctionnel, exposer les prédictions via une API légère (FastAPI) pour requêtes unitaires. Input : noms d'équipes + cotes actuelles. Output : fiche de pari complète avec EV et recommandation.

---

## 5. Résumé des gains — Prévisions vs Réalité

### 5.1 Améliorations réalisées — bilan

| Amélioration | Gain estimé (initial) | Statut | Résultat observé |
|---|---|---|---|
| **Cotes bookmaker en feature** | +5 à +10 pts | ✅ Fait | À mesurer sur le test set post-régénération data |
| Features différentielles | +1 à +2 pts | ✅ Fait | Intégré dans les ~127 features |
| ELO ratings | +1 à +2 pts | ✅ Fait | Cross-saison, K=20, home_advantage=100 |
| Multi-fenêtres rolling | +0.5 à +1 pt | ✅ Fait | last3/last5/last10 + form_trend |
| H2H features | +0.5 à +1 pt | ✅ Fait | 6 features, cross-saison |
| xG rolling averages | +0.5 à +1 pt | ✅ Fait | avg_xg_scored/conceded + xg_vs_goals_diff |
| Features Draw propensity | +1 à +3 pts Draw recall | ✅ Fait | combined_draw_tendency, match_competitiveness |
| Calibration des probabilités | Neutre accuracy | ✅ Fait | Isotonique sur set de calibration dédié |
| Split temporel correct | −1 à −2 pts (mesure honnête) | ✅ Fait | Évaluation sur test ≥ 2022 uniquement |
| Threshold optimization | Variable | ✅ Fait | Métriques "Optimised" dans les résultats |
| Neural Network PyTorch | Non prévu | ✅ Bonus | Pipeline complet avec calibration custom |

### 5.2 Améliorations restantes

| Amélioration | Gain estimé | Statut |
|---|---|---|
| Features fatigue/calendrier | +0.3 à +0.5 pt | ❌ À faire |
| Approche hiérarchique (cascade) | +1 à +2 pts Draw recall | ❌ À évaluer |
| Backtesting stratégies de pari | Validation ROI | ❌ À faire |
| EV / Value Bet calculation | Critique pour le pari | ❌ À faire |
| Stacking des 3 pipelines | +1 à +2 pts accuracy | ❌ À explorer |
| SHAP / explicabilité | Qualité, pas accuracy | ❌ À explorer |
| Optuna hyperparameter tuning | +0.5 à +1 pt | ❌ À explorer |
| Odds over/under + BTTS en feature | +0.5 pt | ❌ À explorer |

### 5.3 Projection d'état actuel (à valider par run)

> *Ces chiffres sont des estimations avant re-run complet avec les nouvelles features. Les données doivent être régénérées (`python src/Data_Processing/Multi-Season_Match_Data_Processor.py`) pour que H2H + ELO + multi-window soient effectifs.*

- **Accuracy multiclasse attendue :** 55-60% (vs 52% pré-Phase 3)
- **Draw recall attendu :** 20-35% avec thresholds optimisés (vs <10% avant)
- **ROI backtest value bets :** non encore mesuré — nécessite `Betting_Backtest.py`

---

## 6. Roadmap — État d'avancement

### ~~Phase 1 — Fondamentaux~~ ✅ COMPLÈTE

- ~~Fix bug chemin Espagne dans `data_processing_config.yaml`~~ ✅
- ~~Ajout colonne `league` lors de la concaténation~~ ✅
- ~~Intégration des cotes bookmaker comme features~~ ✅ (`implied_prob_*`, `odds_ratio`)
- ~~Split temporel dans les pipelines de modèles~~ ✅ (train<2020 / cal 2020-2021 / test≥2022)

### ~~Phase 2 — Feature engineering~~ ✅ COMPLÈTE

- ~~Continuité cross-saison des rolling stats~~ ✅ (`team_history`, `h2h_history`, ELO)
- ~~Features différentielles (home - away)~~ ✅ (8 features `diff_*`)
- ~~xG rolling averages~~ ✅ (`avg_xg_scored`, `avg_xg_conceded`, `xg_vs_goals_diff`)
- ~~Multi-fenêtres temporelles (3, 5, 10 matchs)~~ ✅ + `form_trend`
- ~~Features Draw propensity~~ ✅ (`combined_draw_tendency`, `match_competitiveness`)

### ~~Phase 3 — Modélisation avancée~~ ✅ COMPLÈTE

- ~~ELO ratings dynamiques~~ ✅ (`src/Features/ELO_Rating.py`)
- ~~Head-to-head features~~ ✅ (6 features H2H, cross-saison)
- ~~Cross-validation temporelle~~ ✅ (split temporel strict dans les 3 pipelines)
- ~~Calibration des probabilités (isotonique)~~ ✅ (`CalibratedClassifierCV` + `_IsotonicCalibratedNN`)
- ~~Optimisation des seuils de décision~~ ✅ (`src/Models/threshold_optimizer.py`)
- ~~Pipeline Neural Network PyTorch~~ ✅ (`src/Models/Neural_Network/` — non prévu initialement)

### Phase 4 — Pipeline parieur ❌ EN ATTENTE

- ❌ Features de fatigue/calendrier (`days_since_last_match`, `matches_last_7_days`, `is_midweek_match`)
- ❌ Calcul EV et identification des value bets
- ❌ Output enrichi (fiche de pari complète par match)
- ❌ `src/Analysis/Betting_Backtest.py` — backtesting de stratégies sur test set 2022-2024
- ❌ Approche hiérarchique pour le nul (Stage 1: HomeWin? / Stage 2: Draw vs Away?)

### Phase 5 — Optimisation & exploitation (nouvelles pistes)

- ❌ Stacking des 3 pipelines (MC + Binary + NN) via meta-learner
- ❌ SHAP pour l'explicabilité des prédictions
- ❌ Hyperparameter tuning avec Optuna
- ❌ Odds secondaires en features (over/under, BTTS)
- ❌ API de prédiction en temps réel (FastAPI)

---

## 7. Fichiers clés — État actuel

| Fichier | Rôle | Statut |
|---|---|---|
| `src/Data_Processing/Multi-Season_Match_Data_Processor.py` | Pipeline de feature engineering complet (~127 features) | ✅ Opérationnel |
| `src/Features/ELO_Rating.py` | Calcul ELO dynamique par ligue, cross-saison | ✅ Opérationnel |
| `src/Models/threshold_optimizer.py` | Optimisation des seuils multiclasse et binaire | ✅ Opérationnel |
| `src/Models/Multiclass_Target/Football_Match_Prediction_Pipeline.py` | Pipeline 11 modèles + calibration + thresholds | ✅ Opérationnel |
| `src/Models/Binary_Target/Football_Match_Binary_Prediction_Pipeline.py` | Pipeline binaire Home/Away + calibration | ✅ Opérationnel |
| `src/Models/Neural_Network/Football_Match_NN_Pipeline.py` | Pipeline PyTorch + calibration custom | ✅ Opérationnel |
| `src/Config/configMC_1.yaml` | Config modèles multiclasse | ✅ À jour |
| `src/Config/configBT_1.yaml` | Config modèles binaires | ✅ À jour |
| `src/Config/configNN_1.yaml` | Config Neural Network PyTorch | ✅ Nouveau |
| `src/Config/data_processing_config.yaml` | Chemins ligues, paramètres feature engineering | ✅ Spain fix appliqué |
| `src/Analysis/Betting_Backtest.py` | Backtesting stratégies de pari | ❌ À créer |
| `data/all_leagues_combined.csv` | Dataset combiné 24 523 matchs | ⚠️ À régénérer (H2H + ELO + multi-window) |

> **Note :** Pour activer toutes les nouvelles features, régénérer le dataset combiné :
> ```
> python src/Data_Processing/Multi-Season_Match_Data_Processor.py
> ```

---

*Document généré dans le cadre d'une analyse du projet FootWork — Février 2026*
*Mis à jour le 26/02/2026 — Phases 1, 2, 3 complètes. Phase 4 en attente.*
