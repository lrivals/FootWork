import pandas as pd
import numpy as np

def prepare_match_data_for_prediction(season_file):
    """
    Transform match data into a format suitable for prediction by including
    rolling statistics from previous matches for both teams.
    """
    # Read the season data
    df = pd.read_csv(season_file)
    
    # Sort matches by date to ensure chronological order
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Calculate match result if not present
    if 'match_result' not in df.columns:
        df['goal_difference'] = df['home_team_goal_count'] - df['away_team_goal_count']
        df['match_result'] = df['goal_difference'].apply(lambda x: 
            'HomeWin' if x > 0 else ('AwayWin' if x < 0 else 'Draw'))
    
    # Initialize list to store processed matches
    processed_matches = []
    
    # Process each match
    for idx, match in df.iterrows():
        # Get previous matches for both teams
        prev_matches = df.loc[:idx-1]  # All matches before current match
        
        # Calculate features for home team
        home_features = calculate_team_features(
            prev_matches, 
            match['home_team_name'], 
            'home'
        )
        
        # Calculate features for away team
        away_features = calculate_team_features(
            prev_matches, 
            match['away_team_name'], 
            'away'
        )
        
        # Combine features with match information
        match_data = {
            'date': match['date'],
            'home_team': match['home_team_name'],
            'away_team': match['away_team_name'],
            'target_result': match['match_result'],
            
            # Add actual match stats (can be used to evaluate feature importance)
            'actual_home_goals': match['home_team_goal_count'],
            'actual_away_goals': match['away_team_goal_count']
        }
        
        # Add home team features with 'home_' prefix
        for key, value in home_features.items():
            match_data[f'home_{key}'] = value
            
        # Add away team features with 'away_' prefix
        for key, value in away_features.items():
            match_data[f'away_{key}'] = value
        
        processed_matches.append(match_data)
    
    # Convert to DataFrame
    prediction_df = pd.DataFrame(processed_matches)
    
    # Remove first few matches where not enough history is available
    prediction_df = prediction_df.replace([np.inf, -np.inf], np.nan)
    prediction_df = prediction_df.dropna()
    
    return prediction_df

def calculate_team_features(previous_matches, team_name, venue_type):
    """
    Calculate rolling statistics for a team based on their previous matches.
    
    Args:
        previous_matches: DataFrame containing matches before current match
        team_name: Name of the team to calculate features for
        venue_type: 'home' or 'away' indicating current match venue
    """
    # Filter matches where team played
    team_matches = previous_matches[
        (previous_matches['home_team_name'] == team_name) |
        (previous_matches['away_team_name'] == team_name)
    ]
    
    if len(team_matches) == 0:
        return {
            'avg_goals_scored': np.nan,
            'avg_goals_conceded': np.nan,
            'form_points': np.nan,
            'win_rate': np.nan,
            'last_5_wins': np.nan,
            'goals_scored_last_5': np.nan,
            'goals_conceded_last_5': np.nan,
            'clean_sheets_ratio': np.nan,
            'venue_win_rate': np.nan  # win rate at current venue type
        }
    
    # Calculate goals statistics
    goals_scored = []
    goals_conceded = []
    results = []
    
    for _, match in team_matches.iterrows():
        if match['home_team_name'] == team_name:
            goals_scored.append(match['home_team_goal_count'])
            goals_conceded.append(match['away_team_goal_count'])
            results.append(1 if match['match_result'] == 'HomeWin' 
                         else 0 if match['match_result'] == 'Draw' 
                         else -1)
        else:
            goals_scored.append(match['away_team_goal_count'])
            goals_conceded.append(match['home_team_goal_count'])
            results.append(1 if match['match_result'] == 'AwayWin' 
                         else 0 if match['match_result'] == 'Draw' 
                         else -1)
    
    # Calculate venue-specific statistics
    venue_matches = previous_matches[
        previous_matches[f'{venue_type}_team_name'] == team_name
    ]
    venue_wins = len(venue_matches[
        venue_matches['match_result'] == f'{venue_type.capitalize()}Win'
    ])
    
    # Calculate features
    features = {
        'avg_goals_scored': np.mean(goals_scored),
        'avg_goals_conceded': np.mean(goals_conceded),
        'form_points': sum(results[-5:]) if len(results) >= 5 else np.nan,
        'win_rate': sum(1 for r in results if r == 1) / len(results),
        'last_5_wins': sum(1 for r in results[-5:] if r == 1) if len(results) >= 5 else np.nan,
        'goals_scored_last_5': sum(goals_scored[-5:]) if len(goals_scored) >= 5 else np.nan,
        'goals_conceded_last_5': sum(goals_conceded[-5:]) if len(goals_conceded) >= 5 else np.nan,
        'clean_sheets_ratio': sum(1 for g in goals_conceded if g == 0) / len(goals_conceded),
        'venue_win_rate': venue_wins / len(venue_matches) if len(venue_matches) > 0 else np.nan
    }
    
    return features

def process_all_seasons():
    """
    Process all season files and combine them into a single dataset
    """
    all_seasons_data = []
    
    for i in range(12):  # 12 seasons from 2012-2024
        season_file = f'premier-league-matches-2007-2023/england-premier-league-matches-20{12+i}-to-20{13+i}-stats.csv'
        try:
            season_data = prepare_match_data_for_prediction(season_file)
            all_seasons_data.append(season_data)
            print(f"Processed season 20{12+i}-20{13+i}")
        except Exception as e:
            print(f"Error processing season 20{12+i}-20{13+i}: {e}")
    
    if all_seasons_data:
        combined_data = pd.concat(all_seasons_data, ignore_index=True)
        combined_data.to_csv('matches_for_prediction.csv', index=False)
        print(f"Created dataset with {len(combined_data)} matches")
        return combined_data
    return None

if __name__ == '__main__':
    process_all_seasons()