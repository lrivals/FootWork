# Betting Backtest — Documentation

**Module :** `src/Analysis/Betting_Backtest.py`
**Dépendances :** pandas, numpy, matplotlib
**Entrée :** CSV de prédictions généré par le pipeline multiclass
**Sortie :** rapport texte + plots + CSV des paris

---

## Vue d'ensemble

Le module `Betting_Backtest.py` simule 7 stratégies de pari sur le set de test (2022-2024) à partir des prédictions du pipeline multiclasse. Il évalue la rentabilité de chaque stratégie via des métriques financières standards.

---

## Classe `BettingBacktester`

### Constructeur

```python
BettingBacktester(predictions_df: pd.DataFrame)
```

Trie le DataFrame par date de manière chronologique pour simuler une application réelle des stratégies dans le temps.

### Format d'entrée CSV attendu

Le CSV doit contenir les colonnes suivantes (produites par `save_prediction_csv()` du pipeline multiclass) :

| Colonne | Type | Description |
|---------|------|-------------|
| `date` | str/date | Date du match |
| `home_team` | str | Équipe à domicile |
| `away_team` | str | Équipe extérieure |
| `target_result` | str | Résultat réel (`HomeWin`, `Draw`, `AwayWin`) |
| `league` | str | Nom de la ligue (optionnel, pour filtre) |
| `prob_homewin` | float | Probabilité calibrée victoire domicile |
| `prob_draw` | float | Probabilité calibrée nul |
| `prob_awaywin` | float | Probabilité calibrée victoire extérieure |
| `pred_default` | str | Prédiction argmax (non ajustée) |
| `pred_opt` | str | Prédiction avec seuils optimisés |
| `raw_odds_home` | float | Cote décimale bookmaker victoire domicile |
| `raw_odds_draw` | float | Cote décimale bookmaker nul |
| `raw_odds_away` | float | Cote décimale bookmaker victoire extérieure |
| `ev_home` | float | Expected Value = prob_home × odds_home − 1 |
| `ev_draw` | float | Expected Value = prob_draw × odds_draw − 1 |
| `ev_away` | float | Expected Value = prob_away × odds_away − 1 |
| `kelly_home` | float | Fraction Kelly quart, cappé à 10% |
| `kelly_draw` | float | Fraction Kelly quart, cappé à 10% |
| `kelly_away` | float | Fraction Kelly quart, cappé à 10% |

---

## Les 7 Stratégies de Pari

### Constante commune
- **Mise flat :** `FLAT_STAKE_PCT = 0.01` (1% de la bankroll courante par pari)

### 1. `always_bet_prediction`
Parie systématiquement sur le résultat prédit (`pred_opt` ou `pred_default`), mise flat 1%.
**Usage :** baseline de référence — évalue la rentabilité brute des prédictions du modèle.

### 2. `value_bets_ev5`
Parie uniquement sur l'issue avec le meilleur EV parmi les trois, si ce dernier ≥ 5%. Mise flat 1%.
**Usage :** value betting conservateur — filtre les paris sans valeur attendue positive.

### 3. `value_bets_ev10`
Identique à `value_bets_ev5` avec un seuil EV plus élevé ≥ 10%.
**Usage :** value betting strict — réduit le volume de paris, cible les opportunités à forte valeur.

### 4. `kelly_ev5`
Parie sur l'issue avec le meilleur EV ≥ 5%, mise proportionnelle à la fraction Kelly quart.
La fraction Kelly est calculée : `kelly = (EV / (odds - 1)) × 0.25`, cappée à 10% de la bankroll.
**Usage :** gestion optimale de la bankroll selon le critère de Kelly fractionné.

### 5. `high_confidence`
Parie si **les trois conditions** sont réunies :
- La probabilité du meilleur résultat ≥ 55%
- `pred_default == pred_opt == best_outcome`

Mise flat 1%.
**Usage :** filtre les paris sur lesquels le modèle est très certain et cohérent entre seuils.

### 6. `draw_only_ev5`
Parie **uniquement sur les nuls** si `ev_draw` ≥ 5%. Mise flat 1%.
**Usage :** stratégie spécialisée Draw — teste si le modèle trouve de la valeur sur les nuls.

### 7. `draw_only_ev10`
Identique à `draw_only_ev5` avec un seuil EV plus élevé ≥ 10%.
**Usage :** version stricte de la stratégie Draw.

---

## Méthodes

### `run_strategy(strategy, bankroll=1000) → DataFrame`

Simule la stratégie chronologiquement. Retourne un DataFrame de tous les paris placés :

| Colonne | Description |
|---------|-------------|
| `date` | Date du match |
| `home_team`, `away_team` | Équipes |
| `league` | Ligue |
| `bet_on` | Issue pariée (`HomeWin`, `Draw`, `AwayWin`) |
| `odds` | Cote décimale au moment du pari |
| `stake` | Montant misé (en unités de bankroll) |
| `actual` | Résultat réel |
| `won` | Booléen — pari gagné ou non |
| `profit` | Profit/perte net du pari |
| `bankroll_after` | Bankroll après ce pari |

### `compute_metrics(bets_df, initial_bankroll=1000) → dict`

Calcule les métriques financières sur un DataFrame de paris :

| Métrique | Description |
|----------|-------------|
| `n_bets` | Nombre total de paris |
| `win_rate` | Taux de succès |
| `roi` | Return on Investment = profit_total / montant_total_misé × 100 |
| `total_staked` | Capital total engagé |
| `total_profit` | Profit/perte net total |
| `final_bankroll` | Bankroll finale |
| `max_drawdown_pct` | Drawdown maximal depuis le pic de bankroll (%) |
| `sharpe` | Ratio de Sharpe annualisé sur la base mensuelle (`√12 × μ_mois / σ_mois`) |
| `pnl_by_outcome` | P&L agrégé par type de résultat (HomeWin/Draw/AwayWin) |
| `pnl_by_league` | P&L agrégé par ligue |

### `run_all_strategies(bankroll=1000) → dict`

Exécute les 7 stratégies et retourne `{strategy_name: {'bets_df': ..., 'metrics': ...}}`.

### `save_report(all_results, output_dir, initial_bankroll, league_filter)`

Sauvegarde :
- `backtest_report.txt` — tableau récapitulatif + P&L par outcome + P&L par ligue
- `bankroll_evolution.png` — évolution de la bankroll dans le temps pour chaque stratégie
- `pnl_by_outcome.png` — bar chart P&L par type de résultat et par stratégie
- `bets_<best_strategy>.csv` — CSV des paris de la stratégie la plus rentable

---

## Utilisation

### Ligne de commande

```bash
python src/Analysis/Betting_Backtest.py \
    --predictions results/All_Leagues/Multiclass_Target/<run>/predictions_CatBoost.csv \
    --bankroll 1000 \
    --output results/Backtest/

# Avec filtre par ligue
python src/Analysis/Betting_Backtest.py \
    --predictions results/.../predictions_CatBoost.csv \
    --bankroll 1000 \
    --output results/Backtest/ \
    --league Spain
```

**Arguments CLI :**
| Argument | Défaut | Description |
|----------|--------|-------------|
| `--predictions` | (requis) | Chemin vers le CSV de prédictions |
| `--bankroll` | `1000` | Bankroll initiale en unités |
| `--output` | `results/Backtest/` | Répertoire de sortie |
| `--league` | `All` | Filtre par ligue (`Germany`, `Italy`, `France`, `Spain`, `Premier_League`, `Brazil`, `All`) |

### Usage programmatique

```python
import pandas as pd
from src.Analysis.Betting_Backtest import BettingBacktester

df = pd.read_csv("results/.../predictions_CatBoost.csv")
backtester = BettingBacktester(df)

# Exécuter toutes les stratégies
all_results = backtester.run_all_strategies(bankroll=1000)

# Sauvegarder le rapport
backtester.save_report(all_results, "results/Backtest/", initial_bankroll=1000)
```

---

## Ordre d'exécution complet

1. Générer les données : `python src/Data_Processing/Multi-Season_Match_Data_Processor.py`
2. Entraîner le modèle : `python src/Models/Multiclass_Target/Football_Match_Prediction_Pipeline.py`
3. Lancer le backtest :
```bash
python src/Analysis/Betting_Backtest.py \
    --predictions results/All_Leagues/Multiclass_Target/<timestamp>/predictions_CatBoost.csv \
    --bankroll 1000 \
    --output results/Backtest/
```

---

## Notes importantes

- Les cotes `raw_odds_*` sont stockées dans le CSV de prédictions mais **exclues du training** (via `excluded_columns` dans la config YAML). Elles servent uniquement au calcul de l'EV et au backtest.
- La simulation est strictement **chronologique** — pas d'information future utilisée.
- Les cotes ≤ 1.0 sont ignorées (données manquantes ou invalides).
- Le Kelly quart (`× 0.25`) est une approximation conservatrice du critère de Kelly pour limiter la variance.
