# Football Match Prediction Project

Last update 18.12.2024



## Data Processing

### Feature Engineering
The data preprocessing pipeline creates comprehensive features from historical match data, focusing on several key areas:

1. Goal Scoring Patterns
- Average goals scored/conceded
- First/second half scoring ratios
- Recent goal-scoring form (last 5 matches)
- Clean sheet performance
- Goals per game ratios
- Historical goal difference

2. Game Control Metrics
- Possession statistics and effectiveness
- Corner count and conversion rates
- Shot accuracy and efficiency
  - Shot conversion rate
  - Shots on target ratio
  - Total shots per game
- Ball control indicators

3. Team Form and Momentum
- Points per game
- Recent match results (last 5 games)
- Venue-specific performance
  - Home/Away win ratios
  - Venue-specific scoring rates
  - Venue-specific concession rates
- Form against similar-ranked opponents

4. Team Behavior Patterns
- Disciplinary record
  - Yellow/Red card averages
  - Fouls per game
  - First/Second half card distribution
- Game management stats
  - Comeback ratios
  - Lead loss percentages
  - Clean sheet frequency

### Feature Calculation Window
- All features are calculated using only historical data available before each match
- Rolling statistics typically use the last 5 matches for recent form
- Season-long statistics are maintained separately from recent form metrics

### Target Variables
The project supports two prediction approaches:

1. Binary Classification
- Home Win vs Not Home Win
- Away Win vs Not Away Win

2. Multiclass Classification
- Home Win
- Draw
- Away Win

## Data Source Structure
The raw data includes the following key information for each match:

```
Match Information:
- Date and timestamp
- Home/Away teams
- Final score
- Half-time score

Performance Metrics:
- Shot statistics (total, on target, off target)
- Corner counts
- Possession percentages
- Cards (yellow/red)
- Fouls committed

Game Flow:
- Goal timings
- Card timings
- First/Second half statistics
```

## Model Features Summary

### Team Performance Features (calculated for both home and away teams)
```
Basic Statistics:
- games_played
- wins, draws, losses
- points_per_game
- goals_scored/conceded averages

Recent Form:
- last_5_matches_points
- recent_goals_scored/conceded
- recent_clean_sheets

Efficiency Metrics:
- shot_conversion_rate
- possession_efficiency
- corner_efficiency

Game Management:
- comeback_ratio
- lead_loss_ratio
- clean_sheets_ratio
```
