# Analyse approfondie du projet FootWork & Axes d'amélioration

*Dernière mise à jour : Février 2026*

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

### 1.4 Features engineered actuellement (34 par équipe × 2 = 68)

| Catégorie | Features |
|---|---|
| **Performance globale** | games_played, wins, draws, losses, points_per_game |
| **Buts** | avg_goals_scored, avg_goals_conceded, avg_goal_diff, goals_scored_first_half_ratio, goals_conceded_first_half_ratio |
| **Forme récente (5 matchs)** | recent_goals_scored, recent_goals_conceded, recent_points_per_game, recent_clean_sheets |
| **Efficacité tirs** | shot_conversion_rate, shots_on_target_ratio, avg_shots_per_game, avg_shots_on_target |
| **Contrôle du jeu** | avg_possession, possession_efficiency, avg_corners_for, corner_efficiency |
| **Discipline** | avg_fouls_committed, avg_yellows, avg_reds, cards_first_half_ratio |
| **Domicile/Extérieur** | venue_games, venue_win_ratio, venue_goals_avg, venue_conceded_avg |
| **Gestion de match** | clean_sheets_ratio, scoring_ratio, comeback_ratio, lead_loss_ratio |

### 1.5 Modèles entraînés

10 classifiers testés (Random Forest, Logistic Regression, SVM, Gradient Boosting, XGBoost, LightGBM, CatBoost, MLP, KNN, AdaBoost, Extra Trees) avec deux formulations :
- **Binaire** : HomeWin vs Not, AwayWin vs Not
- **Multiclasse** : HomeWin / Draw / AwayWin

**Préprocessing modèles :** StandardScaler (fitté sur train uniquement), split 80/20 aléatoire.

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

### 🔴 Problème 1 — Fuite temporelle dans le split train/test

**Impact : CRITIQUE**

Le split 80/20 est aléatoire (`random_state=42`). Un match de 2015 peut se retrouver dans le test set, un match de 2024 dans le train set. Or les features rolling sont calculées sur l'historique — le modèle voit potentiellement des statistiques futures.

```python
# Actuel — INCORRECT pour une évaluation réaliste
train_test_split(X, y, test_size=0.2, random_state=42)

# Correct pour simuler les conditions réelles
# Train : 2012-2021, Val : 2022, Test : 2023-2024
```

### 🔴 Problème 2 — Les cotes bookmaker ne sont PAS utilisées comme features

**Impact : CRITIQUE — gain estimé +5 à +10 pts d'accuracy**

Les cotes sont présentes dans chaque CSV brut (`odds_ft_home_team_win`, `odds_ft_draw`, `odds_ft_away_team_win`) et sont disponibles **avant le match**. Elles encodent l'opinion de marchés avec des milliards d'euros de liquidité et constituent empiriquement le prédicteur le plus puissant du résultat d'un match. Ne pas les utiliser est la lacune la plus impactante du projet.

La probabilité implicite se calcule comme :
```
prob_home = (1 / odds_home) / ((1/odds_home) + (1/odds_draw) + (1/odds_away))
```
(normalisation pour retirer la marge du bookmaker)

### 🔴 Problème 3 — Reset des stats en début de saison

**Impact : IMPORTANT**

Le préprocesseur traite chaque saison indépendamment. Les rolling stats d'une équipe remontent à 0 au match 1 de chaque saison. Or Manchester City arrive en août avec toute sa forme et son niveau de la saison précédente. Cette discontinuité artificielle dégrade la qualité des features en début de saison.

### 🟠 Problème 4 — Pas de features head-to-head

**Impact : MODÉRÉ**

Le pipeline ne regarde pas le passé entre les deux équipes qui se rencontrent. L'historique face-à-face (H2H) peut révéler des schémas spécifiques : certaines équipes bloquent systématiquement certains adversaires indépendamment du niveau global.

### 🟠 Problème 5 — Pas de features différentielles (relatives)

**Impact : MODÉRÉ**

On a `home_ppg = 2.1` et `away_ppg = 1.4` séparément, mais jamais `diff_ppg = +0.7`. Les modèles linéaires en particulier bénéficient des différences directes qui capturent le déséquilibre de niveau entre les équipes.

### 🟠 Problème 6 — Le xG historique n'est pas calculé

**Impact : MODÉRÉ**

`team_a_xg` et `team_b_xg` sont dans les données brutes pour chaque match. L'xG (Expected Goals) est un indicateur plus stable que les buts réels (réduit le bruit de la chance/malchance). Des moyennes rolling de xG seraient plus prédictives que des moyennes de buts.

### 🟡 Problème 7 — Une seule fenêtre temporelle pour la forme

**Impact : MODÉRÉ**

La forme récente est calculée uniquement sur les 5 derniers matchs. Capturer plusieurs horizons temporels (3, 5, 10 matchs) permettrait de distinguer la tendance à court terme du niveau à moyen terme, et de calculer des indicateurs de progression/régression.

### 🟡 Problème 8 — Probabilités non calibrées

**Impact : IMPORTANT pour le pari**

Pour un parieur, la **probabilité bien calibrée** est plus importante que la classe prédite. Si le modèle dit "60% HomeWin", cela doit signifier que l'équipe à domicile gagne dans 60% des cas ayant reçu cette prédiction. Sans calibration, les probabilités du modèle ne sont pas directement exploitables pour calculer la valeur espérée d'un pari.

### 🟡 Problème 9 — Colonne `league` absente du dataset combiné

**Impact : FAIBLE-MODÉRÉ**

Après fusion des ligues, l'identité de la ligue est perdue. Or l'avantage domicile varie significativement (Serie A ≠ Premier League), et certains styles de jeu sont spécifiques à une ligue.

### 🟡 Problème 10 — Bug de chemin Espagne → dossier France

`data_processing_config.yaml` route les données traitées de l'Espagne vers `data/France/clean_la_liga_data/` au lieu de `data/Spain/clean_la_liga_data/`.

---

## 4. Axes d'amélioration — Plan détaillé

### AXE A — Corrections fondamentales (Priorité 1)

#### A1. Split temporel strict
- Utiliser les saisons 2012-2021 comme train, 2022-23 comme validation, 2023-24 comme test
- Cela réplique les conditions réelles : on entraîne sur le passé, on prédit le futur
- Impact : l'accuracy "réelle" sera probablement légèrement plus basse qu'actuellement (mesure plus honnête)

#### A2. Continuité cross-saison des rolling stats
- Conserver l'état cumulatif de chaque équipe entre les saisons
- La forme récente (5 derniers matchs) traverse naturellement les frontières de saison
- Évite la chute de qualité des features en début de saison

#### A3. Ajout de la colonne `league`
- Lors de la concaténation, ajouter un identifiant de ligue
- L'encoder comme variable catégorielle dans les modèles
- Permettre des analyses et entraînements par ligue

#### A4. Fix du bug de chemin Espagne

---

### AXE B — Nouvelles features à haute valeur (Priorité 1)

#### B1. Cotes bookmaker → Probabilités implicites

C'est **l'amélioration la plus impactante possible**. Les cotes sont disponibles avant le match dans les données brutes.

```python
# Features à extraire de odds_ft_home_team_win, odds_ft_draw, odds_ft_away_team_win :

# Probabilités implicites brutes
raw_prob_home = 1 / odds_home
raw_prob_draw = 1 / odds_draw
raw_prob_away = 1 / odds_away
total = raw_prob_home + raw_prob_draw + raw_prob_away

# Normalisation (retrait de la marge bookmaker)
implied_prob_home = raw_prob_home / total
implied_prob_draw = raw_prob_draw / total
implied_prob_away = raw_prob_away / total

# Feature supplémentaire : déséquilibre home/away
odds_ratio = odds_away / odds_home  # ratio > 1 = favori à domicile
```

#### B2. xG rolling par équipe

```python
# Calculer rolling_xg_scored et rolling_xg_conceded
# A partir de team_a_xg / team_b_xg dans les données brutes

xg_vs_goals_diff = rolling_goals_scored - rolling_xg_scored
# Positif : surperformance (équipe "chanceuse")
# Négatif : sous-performance (équipe qui mérite mieux)
```

#### B3. Features différentielles (home - away)

Pour chaque paire de features symétrique, calculer la différence :

```python
diff_features = {
  'diff_ppg': home_ppg - away_ppg,
  'diff_recent_form': home_recent_ppg - away_recent_ppg,
  'diff_goals_scored': home_avg_goals_scored - away_avg_goals_scored,
  'diff_goals_conceded': home_avg_goals_conceded - away_avg_goals_conceded,
  'diff_xg': home_avg_xg_scored - away_avg_xg_scored,
  'diff_shots': home_avg_shots - away_avg_shots,
  'diff_possession': home_avg_possession - away_avg_possession,
  'diff_elo': home_elo - away_elo,  # si ELO implémenté
}
```

#### B4. Head-to-Head (H2H)

Pour chaque match, chercher dans l'historique tous les matchs précédents entre les deux équipes :

```python
h2h_features = {
  'h2h_matches': nombre de confrontations précédentes,
  'h2h_home_win_ratio': victoires domicile / total H2H,
  'h2h_draw_ratio': nuls / total H2H,
  'h2h_away_win_ratio': victoires extérieur / total H2H,
  'h2h_avg_goals': moyenne de buts par match H2H,
  # Fenêtre recommandée : 5 derniers H2H ou 3 dernières saisons
}
```

#### B5. ELO ratings dynamiques

L'ELO est un système de rating qui s'adapte après chaque match selon le résultat et le niveau des adversaires :

```
ELO_new = ELO_old + K × (résultat - probabilité_attendue)
résultat = 1 (victoire), 0.5 (nul), 0 (défaite)
probabilité_attendue = 1 / (1 + 10^((ELO_adverse - ELO_equipe) / 400))
```

Features issues de l'ELO :
- `home_elo`, `away_elo` : niveau absolu de chaque équipe
- `elo_diff` : différence de niveau (le prédicteur le plus fort dans cette famille)
- Calculer par ligue pour éviter les comparaisons inter-ligues

#### B6. Features multi-fenêtres temporelles

```python
# Plutôt qu'une seule fenêtre à 5 matchs :
for window in [3, 5, 10]:
    team_features[f'ppg_last_{window}'] = ...
    team_features[f'goals_scored_last_{window}'] = ...
    team_features[f'goals_conceded_last_{window}'] = ...

# Feature de tendance
form_trend = ppg_last_3 - ppg_last_10
# Positif = équipe en montée, Négatif = équipe en baisse de forme
```

#### B7. Features spécifiques aux nuls ("Draw propensity")

Le Draw est la classe la plus difficile à prédire. Créer des features qui capturent sa spécificité :

```python
draw_features = {
  'home_draw_ratio': % de nuls dans les N derniers matchs de l'équipe domicile,
  'away_draw_ratio': % de nuls dans les N derniers matchs de l'équipe visiteur,
  'combined_draw_tendency': (home_draw_ratio + away_draw_ratio) / 2,

  # Matchs serrés → plus de nuls
  'expected_match_competitiveness': abs(home_elo - away_elo),  # petit écart → nul plus probable
  'avg_goals_expected': (home_avg_goals_scored + away_avg_goals_conceded +
                         away_avg_goals_scored + home_avg_goals_conceded) / 2,

  # Styles de jeu
  'home_low_scoring_tendency': 1 if home_avg_goals_scored < threshold else 0,
  'away_defensive_style': away_clean_sheets_ratio,
}
```

#### B8. Features de calendrier et fatigue

```python
schedule_features = {
  'days_since_last_match_home': jours depuis le dernier match (domicile ou déplacement),
  'days_since_last_match_away': idem pour l'équipe visiteur,
  'matches_last_7_days_home': nombre de matchs dans les 7 jours précédents,
  'matches_last_7_days_away': idem,
  'is_midweek_match': 1 si le match est mardi/mercredi/jeudi (proxy coupe/ligue europe),
}
```

---

### AXE C — Améliorations de modélisation (Priorité 2)

#### C1. Cross-validation temporelle

```python
from sklearn.model_selection import TimeSeriesSplit
# Au lieu de train_test_split aléatoire :
tscv = TimeSeriesSplit(n_splits=5)
# Walk-forward : entraîne sur N saisons, teste sur N+1
```

Cela simule les conditions réelles de prédiction et évite tout leakage temporel.

#### C2. Calibration des probabilités

Pour un parieur, les probabilités doivent être fiables. Un modèle qui dit "70% HomeWin" doit avoir raison dans ~70% des cas :

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# Calibration isotonique (recommandée avec beaucoup de données)
calibrated_model = CalibratedClassifierCV(best_model, method='isotonic', cv=5)
calibrated_model.fit(X_train, y_train)

# Évaluation : reliability diagram (courbe de calibration)
prob_true, prob_pred = calibration_curve(y_test, probs[:, class_idx], n_bins=10)
```

#### C3. Approche hiérarchique pour contourner le problème des nuls

```
Stage 1 : "Est-ce une victoire à domicile ?" → HomeWin vs Not-HomeWin
Stage 2 (si Not-HomeWin) : "Est-ce un nul ?" → Draw vs AwayWin

Avantages :
- Chaque binaire est plus simple et mieux équilibré
- Le Stage 2 peut utiliser des features spécifiques aux nuls
- La cascade capture mieux la logique des résultats football
```

#### C4. Optimisation des seuils de décision

Par défaut, la classification se fait à seuil 0.5. Pour le pari, on peut ajuster :

```python
# Maximiser la précision sur les paris à haute confiance
thresholds = np.arange(0.4, 0.9, 0.05)
for t in thresholds:
    # Ne parier que si max(probas) > t
    high_conf_mask = probs.max(axis=1) > t
    precision = (y_pred[high_conf_mask] == y_test[high_conf_mask]).mean()
    coverage = high_conf_mask.mean()
    # Trouver le compromis precision/coverage optimal
```

---

### AXE D — Pipeline orienté parieur (Priorité 2-3)

#### D1. Output enrichi pour chaque match

Au lieu d'une simple classe prédite, produire une fiche de pari complète :

```python
{
  "match": "PSG vs Olympique Lyonnais",
  "date": "2024-03-15",
  "league": "Ligue 1",

  # Probabilités du modèle (calibrées)
  "model_prob_home": 0.52,
  "model_prob_draw": 0.24,
  "model_prob_away": 0.24,

  # Probabilités implicites des cotes bookmaker
  "market_prob_home": 0.55,
  "market_prob_draw": 0.27,
  "market_prob_away": 0.20,

  # Value bets : EV = (prob_modèle × cote) - 1
  # EV > 0 → value bet potentielle
  "ev_home": -0.06,   # PSG légèrement surévalué par le marché
  "ev_draw":  -0.11,
  "ev_away":  +0.08,  # Lyon sous-évalué par le marché → VALUE BET

  # Confiance du modèle
  "model_entropy": 1.52,       # proche du max (1.58) → forte incertitude
  "confidence_level": "low",   # low / medium / high

  # Recommandation parieur
  "recommendation": "AWAY VALUE BET (EV=+8%). Confiance modèle : FAIBLE → mise réduite recommandée",

  # Contexte
  "home_elo": 1842,
  "away_elo": 1654,
  "elo_diff": +188,  # Fort favori domicile selon ELO
  "h2h_last_5": "H:3 D:1 A:1",  # PSG domine historiquement
}
```

#### D2. Calcul de la Value Bet (Expected Value)

```python
# Pour chaque outcome :
ev_home = (model_prob_home × odds_home) - 1
ev_draw = (model_prob_draw × odds_draw) - 1
ev_away = (model_prob_away × odds_away) - 1

# EV > 0 → le marché sous-estime la probabilité de cet outcome
# EV > 0.05 (5%) → seuil minimal pour considérer un pari
# EV > 0.10 (10%) → pari intéressant

# Taille de mise selon Kelly Criterion :
kelly_fraction = ev_outcome / (odds_outcome - 1)
# En pratique : appliquer 1/4 Kelly pour limiter la variance
```

#### D3. Backtesting de stratégies

Simuler les stratégies de pari sur les données test (2023-24) :

```python
strategies = {
  "always_bet_model_prediction": mise sur la classe prédite à cote fixe,
  "value_bets_ev5": parie uniquement si EV > 5%,
  "value_bets_ev10": parie uniquement si EV > 10%,
  "kelly_ev5": mise Kelly quand EV > 5%,
  "high_confidence_only": mise uniquement si entropy < seuil,
}

# Métriques de backtest :
# - ROI total (%)
# - Win rate (%)
# - Nombre de paris
# - Maximum drawdown
# - Sharpe ratio des gains
# - Profit/Loss par type de résultat (Home/Draw/Away)
```

---

## 5. Résumé des gains attendus

| Amélioration | Gain accuracy estimé | Impact pour le pari |
|---|---|---|
| **Cotes bookmaker en feature** | **+5 à +10 pts** | Très élevé — aligne sur les meilleurs prédicteurs du marché |
| Features différentielles | +1 à +2 pts | Modéré |
| ELO ratings | +1 à +2 pts | Élevé — mesure directe du déséquilibre |
| Multi-fenêtres rolling | +0.5 à +1 pt | Modéré |
| H2H features | +0.5 à +1 pt | Modéré — utile pour les affiches récurrentes |
| xG rolling averages | +0.5 à +1 pt | Modéré — signal plus stable que les buts |
| Features de nul (Draw propensity) | +1 à +3 pts sur Draw recall | Élevé — les cotes Draw sont souvent value |
| Calibration des probabilités | Neutre sur accuracy | **Critique pour le calcul EV** |
| Split temporel correct | -1 à -2 pts (mesure honnête) | Critique — évaluation réaliste |
| Approche hiérarchique | +1 à +2 pts | Modéré |

**Projection réaliste après améliorations :**
- Accuracy multiclasse : 55-60% (vs 52% actuel)
- Draw recall : 25-35% (vs <10% actuel)
- ROI backtest value bets (EV > 5%) : à déterminer, mais positif si le modèle est meilleur que les cotes implicites

---

## 6. Roadmap d'implémentation suggérée

### Phase 1 — Fondamentaux (semaine 1)
1. Fix bug chemin Espagne dans `data_processing_config.yaml`
2. Ajout colonne `league` lors de la concaténation
3. Intégration des cotes bookmaker comme features dans `Multi-Season_Match_Data_Processor.py`
4. Split temporel dans les pipelines de modèles

### Phase 2 — Feature engineering (semaine 2)
5. Continuité cross-saison des rolling stats
6. Features différentielles (home - away)
7. xG rolling averages
8. Multi-fenêtres temporelles (3, 5, 10 matchs)
9. Features Draw propensity

### Phase 3 — Modélisation avancée (semaine 3)
10. ELO ratings dynamiques
11. Head-to-head features
12. Features de fatigue/calendrier
13. Cross-validation temporelle (TimeSeriesSplit)
14. Calibration des probabilités (isotonique)

### Phase 4 — Pipeline parieur (semaine 4)
15. Calcul EV et identification des value bets
16. Output enrichi (fiche de pari complète)
17. Backtesting de stratégies sur 2023-24
18. Approche hiérarchique pour le nul (optionnel)

---

## 7. Fichiers clés à modifier

| Fichier | Modifications |
|---|---|
| `src/Data_Processing/Multi-Season_Match_Data_Processor.py` | Ajout cotes, xG rolling, H2H, multi-window, diff features, continuité cross-saison |
| `src/Config/data_processing_config.yaml` | Fix Spain path, nouveaux paramètres |
| `src/Models/Multiclass_Target/Football_Match_Prediction_Pipeline.py` | Split temporel, calibration, output enrichi |
| `src/Models/Binary_Target/Football_Match_Binary_Prediction_Pipeline.py` | Split temporel |
| `src/Features/ELO_Rating.py` *(nouveau)* | Calcul ELO dynamique |
| `src/Analysis/Betting_Backtest.py` *(nouveau)* | Backtesting stratégies de pari |
| `src/Config/configMC_2.yaml` *(nouveau)* | Config pour le nouveau pipeline amélioré |

---

*Document généré dans le cadre d'une analyse du projet FootWork — Février 2026*
