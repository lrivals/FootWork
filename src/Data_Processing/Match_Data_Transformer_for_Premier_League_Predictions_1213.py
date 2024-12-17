import pandas as pd
import numpy as np
from datetime import datetime

def prepare_match_data_for_prediction(df):
    """
    Transform match data into a format suitable for prediction by including
    rolling statistics from previous matches for both teams.
    
    Args:
        df: DataFrame with Premier League match data
    """
    # Convert date and sort chronologically
    df['date'] = pd.to_datetime(df['date_GMT'], format='%b %d %Y - %I:%M%p')
    df = df.sort_values('date').reset_index(drop=True)
    
    # Calculate goal difference and match result
    df['goal_difference'] = df['home_team_goal_count'] - df['away_team_goal_count']
    df['match_result'] = df['goal_difference'].apply(lambda x: 
        'HomeWin' if x > 0 else ('AwayWin' if x < 0 else 'Draw'))
    
    # Initialize list to store processed matches
    processed_matches = []
    
    # Process each match
    for idx in range(len(df)):
        current_match = df.iloc[idx]
        # Get matches before current match
        prev_matches = df.iloc[:idx]
        
        # Calculate features for home team
        home_features = calculate_team_features(
            prev_matches, 
            current_match['home_team_name'],
            'home',
            last_n_matches=5  # Use last 5 matches for form
        )
        
        # Calculate features for away team
        away_features = calculate_team_features(
            prev_matches, 
            current_match['away_team_name'],
            'away',
            last_n_matches=5
        )
        
        # Combine features with match information
        match_data = {
            'date': current_match['date'],
            'home_team': current_match['home_team_name'],
            'away_team': current_match['away_team_name'],
            'target_result': current_match['match_result'],
            'target_home_goals': current_match['home_team_goal_count'],
            'target_away_goals': current_match['away_team_goal_count'],
        }
        
        # Add team features
        for key, value in home_features.items():
            match_data[f'home_{key}'] = value
        for key, value in away_features.items():
            match_data[f'away_{key}'] = value
            
        processed_matches.append(match_data)
    
    # Convert to DataFrame
    prediction_df = pd.DataFrame(processed_matches)
    
    # Remove first few matches where not enough history is available
    prediction_df = prediction_df.replace([np.inf, -np.inf], np.nan)
    prediction_df = prediction_df.dropna()
    
    return prediction_df

def calculate_team_features(previous_matches, team_name, venue_type, last_n_matches=5):
    """
    Calculate rolling statistics for a team based on their previous matches.
    
    Args:
        previous_matches: DataFrame containing matches before current match
        team_name: Name of the team to calculate features for
        venue_type: 'home' or 'away' indicating current match venue
        last_n_matches: Number of recent matches to consider for form
    """
    if len(previous_matches) == 0:
        return get_empty_features()

    # Get team's matches (both home and away)
    team_matches = previous_matches[
        (previous_matches['home_team_name'] == team_name) |
        (previous_matches['away_team_name'] == team_name)
    ].copy()
    
    if len(team_matches) == 0:
        return get_empty_features()
    
    # Calculate basic stats for each match
    team_matches['team_goals'] = np.where(
        team_matches['home_team_name'] == team_name,
        team_matches['home_team_goal_count'],
        team_matches['away_team_goal_count']
    )
    
    team_matches['opponent_goals'] = np.where(
        team_matches['home_team_name'] == team_name,
        team_matches['away_team_goal_count'],
        team_matches['home_team_goal_count']
    )
    
    team_matches['team_shots'] = np.where(
        team_matches['home_team_name'] == team_name,
        team_matches['home_team_shots'],
        team_matches['away_team_shots']
    )
    
    team_matches['team_shots_on_target'] = np.where(
        team_matches['home_team_name'] == team_name,
        team_matches['home_team_shots_on_target'],
        team_matches['away_team_shots_on_target']
    )
    
    team_matches['team_corners'] = np.where(
        team_matches['home_team_name'] == team_name,
        team_matches['home_team_corner_count'],
        team_matches['away_team_corner_count']
    )
    
    # Get recent matches for form
    recent_matches = team_matches.tail(last_n_matches)
    
    # Calculate venue-specific stats
    venue_matches = previous_matches[
        previous_matches[f'{venue_type}_team_name'] == team_name
    ]
    
    features = {
        # Overall performance
        'games_played': len(team_matches),
        'avg_goals_scored': team_matches['team_goals'].mean(),
        'avg_goals_conceded': team_matches['opponent_goals'].mean(),
        'avg_goal_diff': (team_matches['team_goals'] - team_matches['opponent_goals']).mean(),
        
        # Recent form (last N matches)
        'recent_avg_goals': recent_matches['team_goals'].mean(),
        'recent_avg_conceded': recent_matches['opponent_goals'].mean(),
        'recent_avg_goal_diff': (recent_matches['team_goals'] - recent_matches['opponent_goals']).mean(),
        
        # Shot statistics
        'avg_shots': team_matches['team_shots'].mean(),
        'avg_shots_on_target': team_matches['team_shots_on_target'].mean(),
        'shot_accuracy': team_matches['team_shots_on_target'].sum() / team_matches['team_shots'].sum()
            if team_matches['team_shots'].sum() > 0 else 0,
        
        # Venue-specific performance
        'venue_games': len(venue_matches),
        'venue_goals_avg': venue_matches[f'{venue_type}_team_goal_count'].mean() if len(venue_matches) > 0 else 0,
        'venue_conceded_avg': venue_matches[f'{"away" if venue_type == "home" else "home"}_team_goal_count'].mean()
            if len(venue_matches) > 0 else 0,
        
        # Form indicators
        'clean_sheets_ratio': (team_matches['opponent_goals'] == 0).mean(),
        'scoring_ratio': (team_matches['team_goals'] > 0).mean(),
        'avg_corners': team_matches['team_corners'].mean()
    }
    
    return features

def get_empty_features():
    """Return dictionary of features with null values for teams with no history"""
    return {
        'games_played': 0,
        'avg_goals_scored': np.nan,
        'avg_goals_conceded': np.nan,
        'avg_goal_diff': np.nan,
        'recent_avg_goals': np.nan,
        'recent_avg_conceded': np.nan,
        'recent_avg_goal_diff': np.nan,
        'avg_shots': np.nan,
        'avg_shots_on_target': np.nan,
        'shot_accuracy': np.nan,
        'venue_games': 0,
        'venue_goals_avg': np.nan,
        'venue_conceded_avg': np.nan,
        'clean_sheets_ratio': np.nan,
        'scoring_ratio': np.nan,
        'avg_corners': np.nan
    }

if __name__ == '__main__':
    # Read the data
    df = pd.read_csv('premier-league-matches-2007-2023/england-premier-league-matches-2012-to-2013-stats.csv')
    
    # Process the data
    prediction_data = prepare_match_data_for_prediction(df)
    
    # Save processed data
    prediction_data.to_csv('processed_matches_for_prediction.csv', index=False)
    
    print(f"Created dataset with {len(prediction_data)} matches")
    print("\nFeatures created for each team:")
    home_features = [col for col in prediction_data.columns if col.startswith('home_')]
    print("\n".join(f"- {feat}" for feat in home_features))
