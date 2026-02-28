# Football Match Prediction — Data Processing & Feature Engineering

*Dernière mise à jour : Février 2026 (état post-Phase 4)*

**Script principal :** `src/Data_Processing/Multi-Season_Match_Data_Processor.py`
**Config :** `src/Config/data_processing_config.yaml`

---

## Vue d'ensemble

Le pipeline de traitement transforme des CSV bruts (footystats.org) en matrices de features prêtes pour le machine learning. Il maintient une continuité cross-saison pour les stats d'équipes, les confrontations H2H et les classements ELO.

---

## Feature Engineering

### 1. Features de performance globale (par équipe × 2 = colonnes `home_*` et `away_*`)

**48 features par équipe, calculées uniquement sur l'historique avant le match :**

| Catégorie | Features |
|-----------|----------|
| **Performance globale** | `games_played`, `wins`, `draws`, `losses`, `points_per_game` |
| **Buts** | `avg_goals_scored`, `avg_goals_conceded`, `avg_goal_diff`, `goals_scored_first_half_ratio`, `goals_conceded_first_half_ratio` |
| **Forme multi-fenêtres** (3/5/10 matchs) | `recent_ppg_last{3,5,10}`, `recent_goals_scored_last{3,5,10}`, `recent_goals_conceded_last{3,5,10}`, `recent_clean_sheets_last{3,5,10}`, `draw_ratio_last{3,5,10}` |
| **Tendance de forme** | `form_trend = ppg_last3 − ppg_last10` (positif = en montée) |
| **xG rolling** | `avg_xg_scored`, `avg_xg_conceded`, `xg_vs_goals_diff` |
| **Efficacité tirs** | `shot_conversion_rate`, `shots_on_target_ratio`, `avg_shots_per_game`, `avg_shots_on_target` |
| **Contrôle du jeu** | `avg_possession`, `possession_efficiency`, `avg_corners_for`, `corner_efficiency` |
| **Discipline** | `avg_fouls_committed`, `avg_yellows`, `avg_reds`, `cards_first_half_ratio` |
| **Domicile/Extérieur** | `venue_games`, `venue_win_ratio`, `venue_goals_avg`, `venue_conceded_avg` |
| **Gestion de match** | `clean_sheets_ratio`, `scoring_ratio`, `comeback_ratio`, `lead_loss_ratio` |

### 2. Cotes bookmaker / Probabilités implicites (4 features)

| Feature | Description |
|---------|-------------|
| `implied_prob_home` | Probabilité implicite normalisée victoire domicile (marge bookmaker retirée) |
| `implied_prob_draw` | Probabilité implicite normalisée nul |
| `implied_prob_away` | Probabilité implicite normalisée victoire extérieure |
| `odds_ratio` | `odds_away / odds_home` — déséquilibre home/away selon le bookmaker |

### 3. Cotes brutes (3 colonnes — stockées mais exclues du training)

| Feature | Description |
|---------|-------------|
| `raw_odds_home` | Cote décimale brute victoire domicile |
| `raw_odds_draw` | Cote décimale brute nul |
| `raw_odds_away` | Cote décimale brute victoire extérieure |

Ces colonnes sont stockées dans les CSV mais exclues des features d'entraînement via `excluded_columns` dans les configs YAML. Elles servent au calcul de l'Expected Value (EV) et du Kelly criterion dans le module de backtest.

### 4. Features ELO (3 features)

| Feature | Description |
|---------|-------------|
| `home_elo` | Classement ELO de l'équipe domicile avant le match |
| `away_elo` | Classement ELO de l'équipe extérieure avant le match |
| `elo_diff` | `home_elo − away_elo` |

Calculées via `src/Features/ELO_Rating.py` — voir [elo_h2h_features.md](elo_h2h_features.md).
Paramètres : `initial_elo=1500`, `k_factor=20`, `home_advantage=100`.

### 5. Features différentielles (8 features, home − away)

```
diff_ppg, diff_recent_ppg, diff_goals_scored, diff_goals_conceded,
diff_xg_scored, diff_shots, diff_possession, diff_form_trend
```

### 6. Draw propensity (2 features)

| Feature | Description |
|---------|-------------|
| `combined_draw_tendency` | Moyenne des draw_ratio_last5 des deux équipes |
| `match_competitiveness` | Mesure de l'équilibre du match selon les stats |

### 7. Features Head-to-Head (6 features)

| Feature | Description |
|---------|-------------|
| `h2h_home_wins` | Victoires domicile dans les 5 derniers face-à-face |
| `h2h_away_wins` | Victoires extérieur dans les 5 derniers face-à-face |
| `h2h_draws` | Nuls dans les 5 derniers face-à-face |
| `h2h_home_goals_avg` | Buts moyens de l'équipe domicile en H2H |
| `h2h_away_goals_avg` | Buts moyens de l'équipe extérieure en H2H |
| `h2h_matches_count` | Nombre de confrontations historiques disponibles |

Voir [elo_h2h_features.md](elo_h2h_features.md) pour les détails d'implémentation.

### 8. Features Fatigue / Calendrier (5 features)

| Feature | Description |
|---------|-------------|
| `days_since_last_match_home` | Jours depuis le dernier match de l'équipe domicile |
| `days_since_last_match_away` | Jours depuis le dernier match de l'équipe extérieure |
| `matches_last_7_days_home` | Nombre de matchs de l'équipe domicile dans les 7 derniers jours |
| `matches_last_7_days_away` | Nombre de matchs de l'équipe extérieure dans les 7 derniers jours |
| `is_midweek_match` | Booléen — match en semaine (lundi–jeudi) |

Ces features capturent la fatigue physique et les contraintes de calendrier.

---

## Résumé des features

| Catégorie | Nombre de features |
|-----------|-------------------|
| Features par équipe (home + away) | 48 × 2 = **96** |
| Cotes / probabilités implicites | **4** |
| ELO | **3** |
| Différentielles (home − away) | **8** |
| Draw propensity | **2** |
| H2H | **6** |
| Fatigue / Calendrier | **5** |
| **Total features d'entraînement** | **~124** |
| Cotes brutes (stockées, non entraînées) | +3 |
| Métadonnées (date, équipes, saison, league, targets) | +7 |
| **Total colonnes dans le CSV** | **~134** |

---

## Fenêtres temporelles

- **Stats globales :** toute l'histoire disponible (cross-saison, aucun reset)
- **Forme récente :** fenêtres last3, last5, last10 matchs
- **H2H :** 5 dernières confrontations directes
- **Fatigue :** 7 derniers jours calendaires

---

## Cibles de prédiction

### Multiclasse (3 classes)
- `target_result` : `HomeWin` / `Draw` / `AwayWin`

### Binaire (2 modèles distincts)
- `HomeWin vs Not-HomeWin`
- `AwayWin vs Not-AwayWin`

---

## Sources de données brutes

Les CSV bruts (footystats.org) contiennent par match :

```
Identifiants :   date_GMT, home_team_name, away_team_name
Scores :         home/away_team_goal_count, HT goals
Tirs :           home/away_team_shots, shots_on_target, shots_off_target
Possession :     home/away_team_possession (%)
Corners :        home/away_team_corner_count
Cartons :        home/away_team_yellow/red_cards, first/second_half_cards
Fautes :         home/away_team_fouls
xG :             Home/Away Team Pre-Match xG, team_a_xg, team_b_xg
Cotes :          odds_ft_home_team_win, odds_ft_draw, odds_ft_away_team_win
```

---

## Exécution

```bash
python src/Data_Processing/Multi-Season_Match_Data_Processor.py
```

Traitement parallèle des 6 ligues via `ProcessPoolExecutor`. Sortie :
- `data/<League>/clean_<league>_data/processed_matches_<years>.csv` — par saison
- `data/all_leagues_combined.csv` — dataset combiné (~24 500 matchs)

---

## Garantie anti-fuite temporelle

Pour chaque match, les features sont calculées **exclusivement** à partir des données disponibles avant ce match :
1. `get_prematch_elos()` → ELO **avant** le match
2. `calculate_team_features()` → stats équipe sur l'historique passé uniquement
3. `compute_h2h_features()` → H2H sur les confrontations passées uniquement
4. Mise à jour ELO + team_history + h2h_history → **après** avoir capturé toutes les features

La continuité cross-saison (`team_history`, `h2h_history`, `ELOCalculator`) est maintenue sans jamais utiliser d'information future.
