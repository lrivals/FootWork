# ELO Ratings & Features Head-to-Head — Documentation

Ce document couvre :
- **`src/Features/ELO_Rating.py`** — système de notation ELO
- **Features H2H** — calculées dans `src/Data_Processing/Multi-Season_Match_Data_Processor.py`

---

## 1. Système ELO (`ELO_Rating.py`)

### Principe

Le système ELO est un système de classement basé sur les résultats de matchs. Après chaque match, le classement des deux équipes est mis à jour selon :

```
ELO_new = ELO_old + K × (score_réel - score_attendu)

score_attendu = 1 / (1 + 10^((ELO_adversaire − ELO_équipe − bonus) / 400))
score_réel    = 1.0 (victoire), 0.5 (nul), 0.0 (défaite)
```

L'équipe qui joue à domicile reçoit un bonus de `home_advantage` points dans le calcul du score attendu.

---

### Classe `ELOCalculator`

```python
from src.Features.ELO_Rating import ELOCalculator

elo = ELOCalculator(initial_elo=1500, k_factor=20, home_advantage=100)
```

#### Paramètres de construction

| Paramètre | Valeur par défaut | Description |
|-----------|-------------------|-------------|
| `initial_elo` | 1500 | Cote ELO de départ pour toute équipe inconnue |
| `k_factor` | 20 | Sensibilité des mises à jour (standard football de club) |
| `home_advantage` | 100 | Bonus ELO pour l'équipe à domicile dans le score attendu |

#### Méthodes

##### `get_elo(team) → float`
Retourne le classement ELO actuel d'une équipe (`initial_elo` si jamais vue).

##### `expected_score(elo_team, elo_opponent, team_is_home=False) → float`
Calcule le score attendu (proxy de probabilité de victoire) pour une équipe.
L'équipe à domicile reçoit un bonus de `+home_advantage` dans la formule.

##### `get_prematch_elos(home_team, away_team) → (elo_home, elo_away, elo_diff)`
Retourne les ELO **avant** le match — à appeler **avant** `update()`.
Produit 3 features : `home_elo`, `away_elo`, `elo_diff = home_elo - away_elo`.

##### `update(home_team, away_team, home_goals, away_goals) → None`
Met à jour les ELO des deux équipes **après** le match.

```python
# Usage correct (ordre strict)
elo_h, elo_a, elo_diff = elo_calc.get_prematch_elos(home_team, away_team)
# → features utilisées pour ce match
elo_calc.update(home_team, away_team, home_goals, away_goals)
# → mise à jour pour les matchs suivants
```

##### `snapshot() → dict`
Retourne une copie du dictionnaire `{team: elo}` pour debug/export.

---

### Continuité cross-saison

Une instance `ELOCalculator` est créée **une fois par ligue** et réutilisée pour toutes les saisons. Les classements ELO ne sont jamais réinitialisés entre saisons — ils reflètent l'historique complet de la ligue depuis la première saison disponible.

```python
# Dans process_league_data() :
elo_calculator = ELOCalculator()  # une seule instance par ligue
for season_file in sorted_season_files:
    process_season(season_file, ..., elo_calculator=elo_calculator)  # persistance
```

---

### Features ELO générées

| Feature | Description |
|---------|-------------|
| `home_elo` | Classement ELO de l'équipe à domicile avant le match |
| `away_elo` | Classement ELO de l'équipe extérieure avant le match |
| `elo_diff` | Différence `home_elo − away_elo` (positif = favori domicile) |

Ces 3 features sont exclues du calcul des features différentielles `diff_*` (déjà différentielle par nature).

---

## 2. Features Head-to-Head (H2H)

### Principe

Les features H2H capturent l'historique des confrontations directes entre deux équipes spécifiques, indépendamment de leurs performances générales. Un historique de buts favorables en H2H peut indiquer un avantage psychologique ou tactique.

### Implémentation

Le `h2h_history` est un dictionnaire persistant à travers toutes les saisons d'une ligue :

```python
h2h_key = tuple(sorted([home_team, away_team]))  # paire canonique
h2h_history[h2h_key].append({match_info})        # après chaque match
```

La clé canonique `tuple(sorted(...))` garantit que les confrontations dans les deux sens sont regroupées.

### Paramètre de fenêtre

```yaml
# data_processing_config.yaml
processing_params:
  h2h_window: 5   # nombre de dernières confrontations considérées
```

Seuls les `h2h_window` (par défaut 5) derniers face-à-face sont utilisés pour calculer les features.

### Features H2H générées

| Feature | Description |
|---------|-------------|
| `h2h_home_wins` | Nombre de victoires à domicile dans les derniers H2H |
| `h2h_away_wins` | Nombre de victoires extérieures dans les derniers H2H |
| `h2h_draws` | Nombre de nuls dans les derniers H2H |
| `h2h_home_goals_avg` | Moyenne de buts marqués par l'équipe actuelle à domicile en H2H |
| `h2h_away_goals_avg` | Moyenne de buts marqués par l'équipe actuelle extérieure en H2H |
| `h2h_matches_count` | Nombre total de confrontations dans la fenêtre |

Si une paire d'équipes n'a pas d'historique H2H disponible, les 6 features sont initialisées à `0`.

### Exemple d'interprétation

Pour le match PSG (domicile) vs Lyon (extérieur) :
- `h2h_home_wins = 3` → PSG a gagné 3 des 5 derniers face-à-face à la maison
- `h2h_draws = 1` → 1 nul sur les 5 derniers
- `h2h_home_goals_avg = 2.2` → PSG marque en moyenne 2.2 buts par H2H à domicile

---

## 3. Intégration dans le pipeline de traitement

### Ordre des opérations par match

Pour chaque match, dans `prepare_match_data_for_prediction()` :

```python
# 1. Récupérer ELO AVANT le match
home_elo, away_elo, elo_diff = elo_calculator.get_prematch_elos(home_team, away_team)

# 2. Calculer features équipe (depuis team_history)
home_features = calculate_team_features(history, home_team, 'home')
away_features = calculate_team_features(history, away_team, 'away')

# 3. Calculer features H2H (depuis h2h_history)
h2h_feats = compute_h2h_features(h2h_history, home_team, away_team, h2h_window)

# 4. Mettre à jour ELO APRÈS (pour les matchs suivants)
elo_calculator.update(home_team, away_team, home_goals, away_goals)

# 5. Mettre à jour team_history et h2h_history
```

Cette séquence stricte garantit l'absence de fuite temporelle : toutes les features d'un match n'utilisent que des informations antérieures à ce match.

---

## 4. Persistance cross-saison

Trois objets persistent à travers les saisons pour chaque ligue :

| Objet | Type | Contenu |
|-------|------|---------|
| `global_team_history` | `dict[team → list[match]]` | Tous les matchs précédents de chaque équipe |
| `global_h2h_history` | `dict[pair → list[match]]` | Tous les face-à-face précédents |
| `elo_calculator` | `ELOCalculator` | Classements ELO cumulés |

Ces objets sont passés de saison en saison dans `process_league_data()`, ce qui élimine l'effet "cold start" du début de chaque nouvelle saison.

---

## 5. Constantes et choix de conception

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `initial_elo` | 1500 | Valeur neutre standard (FIFA utilise 1000-2000) |
| `k_factor` | 20 | Standard football de club (FiveThirtyEight utilise 20-60) |
| `home_advantage` | 100 | Avantage domicile typique pour le football européen |
| `h2h_window` | 5 | Équilibre entre historique suffisant et pertinence récente |
| Clé H2H | `tuple(sorted(...))` | Clé canonique indépendante de l'équipe qui reçoit |
