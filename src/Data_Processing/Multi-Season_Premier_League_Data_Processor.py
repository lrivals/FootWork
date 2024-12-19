from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import os
from pathlib import Path
import sys
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
from src.Config.Config_Manager import ConfigManager

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
            last_n_matches=5
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
            'season': current_match['season'],  # Added season tracking
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
    Calculate comprehensive team statistics based on previous matches.
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
    team_matches['is_home'] = team_matches['home_team_name'] == team_name
    
    # Goals and Scoring
    team_matches['team_goals'] = np.where(
        team_matches['is_home'],
        team_matches['home_team_goal_count'],
        team_matches['away_team_goal_count']
    )
    team_matches['opponent_goals'] = np.where(
        team_matches['is_home'],
        team_matches['away_team_goal_count'],
        team_matches['home_team_goal_count']
    )
    team_matches['team_goals_ht'] = np.where(
        team_matches['is_home'],
        team_matches['home_team_goal_count_half_time'],
        team_matches['away_team_goal_count_half_time']
    )
    team_matches['opponent_goals_ht'] = np.where(
        team_matches['is_home'],
        team_matches['away_team_goal_count_half_time'],
        team_matches['home_team_goal_count_half_time']
    )
    
    # Shots
    team_matches['team_shots'] = np.where(
        team_matches['is_home'],
        team_matches['home_team_shots'],
        team_matches['away_team_shots']
    )
    team_matches['team_shots_on_target'] = np.where(
        team_matches['is_home'],
        team_matches['home_team_shots_on_target'],
        team_matches['away_team_shots_on_target']
    )
    team_matches['team_shots_off_target'] = np.where(
        team_matches['is_home'],
        team_matches['home_team_shots_off_target'],
        team_matches['away_team_shots_off_target']
    )
    
    # Cards and Fouls
    team_matches['team_yellows'] = np.where(
        team_matches['is_home'],
        team_matches['home_team_yellow_cards'],
        team_matches['away_team_yellow_cards']
    )
    team_matches['team_reds'] = np.where(
        team_matches['is_home'],
        team_matches['home_team_red_cards'],
        team_matches['away_team_red_cards']
    )
    team_matches['team_fouls'] = np.where(
        team_matches['is_home'],
        team_matches['home_team_fouls'],
        team_matches['away_team_fouls']
    )
    team_matches['team_first_half_cards'] = np.where(
        team_matches['is_home'],
        team_matches['home_team_first_half_cards'],
        team_matches['away_team_first_half_cards']
    )
    team_matches['team_second_half_cards'] = np.where(
        team_matches['is_home'],
        team_matches['home_team_second_half_cards'],
        team_matches['away_team_second_half_cards']
    )
    
    # Corners and Possession
    team_matches['team_corners'] = np.where(
        team_matches['is_home'],
        team_matches['home_team_corner_count'],
        team_matches['away_team_corner_count']
    )
    team_matches['team_possession'] = np.where(
        team_matches['is_home'],
        team_matches['home_team_possession'],
        team_matches['away_team_possession']
    )
    
    # Get recent matches for form
    recent_matches = team_matches.tail(last_n_matches)
    
    # Calculate venue-specific stats
    venue_matches = previous_matches[
        previous_matches[f'{venue_type}_team_name'] == team_name
    ]
    
    # Calculate features
    features = {
        # Overall Performance Metrics
        'games_played': len(team_matches),
        'wins': sum((team_matches['team_goals'] > team_matches['opponent_goals'])),
        'draws': sum((team_matches['team_goals'] == team_matches['opponent_goals'])),
        'losses': sum((team_matches['team_goals'] < team_matches['opponent_goals'])),
        'points_per_game': (sum((team_matches['team_goals'] > team_matches['opponent_goals'])) * 3 + 
                          sum((team_matches['team_goals'] == team_matches['opponent_goals']))) / len(team_matches),
        
        # Goal Scoring Patterns
        'avg_goals_scored': team_matches['team_goals'].mean(),
        'avg_goals_conceded': team_matches['opponent_goals'].mean(),
        'avg_goal_diff': (team_matches['team_goals'] - team_matches['opponent_goals']).mean(),
        'goals_scored_first_half_ratio': (team_matches['team_goals_ht'] / team_matches['team_goals']).mean(),
        'goals_conceded_first_half_ratio': (team_matches['opponent_goals_ht'] / team_matches['opponent_goals']).mean(),
        
        # Recent Form
        'recent_goals_scored': recent_matches['team_goals'].mean(),
        'recent_goals_conceded': recent_matches['opponent_goals'].mean(),
        'recent_points_per_game': (sum((recent_matches['team_goals'] > recent_matches['opponent_goals'])) * 3 + 
                                 sum((recent_matches['team_goals'] == recent_matches['opponent_goals']))) / len(recent_matches),
        'recent_clean_sheets': sum(recent_matches['opponent_goals'] == 0) / len(recent_matches),
        
        # Shooting Efficiency
        'shot_conversion_rate': team_matches['team_goals'].sum() / team_matches['team_shots'].sum() 
            if team_matches['team_shots'].sum() > 0 else 0,
        'shots_on_target_ratio': team_matches['team_shots_on_target'].sum() / team_matches['team_shots'].sum()
            if team_matches['team_shots'].sum() > 0 else 0,
        'avg_shots_per_game': team_matches['team_shots'].mean(),
        'avg_shots_on_target': team_matches['team_shots_on_target'].mean(),
        
        # Game Control
        'avg_possession': team_matches['team_possession'].mean(),
        'possession_efficiency': (team_matches['team_goals'].sum() / team_matches['team_possession'].sum() * 100)
            if team_matches['team_possession'].sum() > 0 else 0,
        'avg_corners_for': team_matches['team_corners'].mean(),
        'corner_efficiency': team_matches['team_goals'].sum() / team_matches['team_corners'].sum()
            if team_matches['team_corners'].sum() > 0 else 0,
        
        # Discipline
        'avg_fouls_committed': team_matches['team_fouls'].mean(),
        'avg_yellows': team_matches['team_yellows'].mean(),
        'avg_reds': team_matches['team_reds'].mean(),
        'cards_first_half_ratio': (team_matches['team_first_half_cards'].sum() / 
            (team_matches['team_first_half_cards'].sum() + team_matches['team_second_half_cards'].sum()))
            if (team_matches['team_first_half_cards'].sum() + team_matches['team_second_half_cards'].sum()) > 0 else 0,
        
        # Venue-specific Performance
        'venue_games': len(venue_matches),
        'venue_win_ratio': sum((venue_matches[f'{venue_type}_team_goal_count'] > 
                              venue_matches[f'{"away" if venue_type == "home" else "home"}_team_goal_count'])) / len(venue_matches)
            if len(venue_matches) > 0 else 0,
        'venue_goals_avg': venue_matches[f'{venue_type}_team_goal_count'].mean() 
            if len(venue_matches) > 0 else 0,
        'venue_conceded_avg': venue_matches[f'{"away" if venue_type == "home" else "home"}_team_goal_count'].mean()
            if len(venue_matches) > 0 else 0,
        
        # Form Indicators
        'clean_sheets_ratio': (team_matches['opponent_goals'] == 0).mean(),
        'scoring_ratio': (team_matches['team_goals'] > 0).mean(),
        'comeback_ratio': sum((team_matches['opponent_goals_ht'] > team_matches['team_goals_ht']) & 
                            (team_matches['team_goals'] > team_matches['opponent_goals'])) / len(team_matches),
        'lead_loss_ratio': sum((team_matches['team_goals_ht'] > team_matches['opponent_goals_ht']) & 
                              (team_matches['opponent_goals'] > team_matches['team_goals'])) / len(team_matches)
    }
    
    return features

def get_empty_features():
    """Return dictionary of features with null values for teams with no history"""
    return {
        'games_played': 0,
        'wins': np.nan,
        'draws': np.nan,
        'losses': np.nan,
        'points_per_game': np.nan,
        'avg_goals_scored': np.nan,
        'avg_goals_conceded': np.nan,
        'avg_goal_diff': np.nan,
        'goals_scored_first_half_ratio': np.nan,
        'goals_conceded_first_half_ratio': np.nan,
        'recent_goals_scored': np.nan,
        'recent_goals_conceded': np.nan,
        'recent_points_per_game': np.nan,
        'recent_clean_sheets': np.nan,
        'shot_conversion_rate': np.nan,
        'shots_on_target_ratio': np.nan,
        'avg_shots_per_game': np.nan,
        'avg_shots_on_target': np.nan,
        'avg_possession': np.nan,
        'possession_efficiency': np.nan,
        'avg_corners_for': np.nan,
        'corner_efficiency': np.nan,
        'avg_fouls_committed': np.nan,
        'avg_yellows': np.nan,
        'avg_reds': np.nan,
        'cards_first_half_ratio': np.nan,
        'venue_games': 0,
        'venue_win_ratio': np.nan,
        'venue_goals_avg': np.nan,
        'venue_conceded_avg': np.nan,
        'clean_sheets_ratio': np.nan,
        'scoring_ratio': np.nan,
        'comeback_ratio': np.nan,
        'lead_loss_ratio': np.nan
    }
def process_season(input_file: Path, output_folder: Path, league: str) -> pd.DataFrame:
    """
    Process a single season's data file and save the processed version.
    
    Args:
        input_file: Path to the input CSV file
        output_folder: Path to save the processed data
        league: Name of the league being processed
    
    Returns:
        DataFrame with processed data
    """
    try:
        # Extract season from filename
        filename = input_file.name
        season_match = filename.split('matches-')[1].split('-stats')[0]
        
        print(f"Processing {league} season {season_match}...")
        df = pd.read_csv(input_file)
        df['season'] = season_match
        df['league'] = league
        
        prediction_data = prepare_match_data_for_prediction(df)
        
        # Create league-specific output directory
        os.makedirs(output_folder, exist_ok=True)
        
        # Clean up season string for filename
        season_clean = season_match.replace('-to-', '_to_')
        output_file = output_folder / f'processed_matches_{season_clean}.csv'
        prediction_data.to_csv(output_file, index=False)
        
        print(f"Processed {league} {season_match} season: {len(prediction_data)} matches")
        return prediction_data
        
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        return None   
     

def process_league_data(league: str, config_manager: ConfigManager) -> pd.DataFrame:
    """
    Process all seasons for a specific league
    
    Args:
        league: Name of the league to process
        config_manager: ConfigManager instance
    """
    # Get league-specific configuration
    league_config = config_manager.get_config_value('data_paths', 'leagues', league)
    if not league_config:
        print(f"No configuration found for {league}")
        return None
    
    input_path = Path(league_config['raw_data'])
    output_path = Path(league_config['processed_data'])
    start_year = league_config['start_year']
    end_year = league_config['end_year']
    
    # Get file pattern for the league
    pattern = config_manager.get_config_value('league_patterns', league)
    
    print(f"\nProcessing {league} data...")
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Processing seasons from {start_year} to {end_year}")
    
    all_seasons_data = []
    
    # Process each file in the league directory
    for input_file in input_path.glob(pattern):
        try:
            # Extract year from filename to check if it's within range
            season_start = int(input_file.name.split('-')[-4])
            if start_year <= season_start < end_year:
                season_data = process_season(input_file, output_path, league)
                if season_data is not None:
                    all_seasons_data.append(season_data)
        except Exception as e:
            print(f"Error processing {input_file}: {str(e)}")
            continue
    
    if all_seasons_data:
        combined_data = pd.concat(all_seasons_data, ignore_index=True)
        combined_output = output_path / 'all_seasons_combined.csv'
        os.makedirs(output_path, exist_ok=True)
        combined_data.to_csv(combined_output, index=False)
        
        print(f"\nProcessed all {league} seasons:")
        print(f"Total matches: {len(combined_data)}")
        print(f"Seasons covered: {combined_data['season'].nunique()}")
        print(f"Files saved in: {output_path}")
        
        return combined_data
    else:
        print(f"No data was processed for {league}")
        return None
       
def main():
    # Initialize config manager
    config_manager = ConfigManager("src/Config/data_processing_config.yaml")
    
    # Get list of leagues to process
    leagues = list(config_manager.get_config_value('data_paths', 'leagues').keys())
    
    all_leagues_data = {}
    
    # Process each league
    for league in leagues:
        league_data = process_league_data(league, config_manager)
        if league_data is not None:
            all_leagues_data[league] = league_data
    
    # Optionally combine all leagues data
    if all_leagues_data:
        combined_all_leagues = pd.concat(all_leagues_data.values(), ignore_index=True)
        base_path = Path(config_manager.get_config_value('data_paths', 'base_path'))
        combined_output = base_path / 'all_leagues_combined.csv'
        os.makedirs(base_path, exist_ok=True)
        combined_all_leagues.to_csv(combined_output, index=False)
        
        print("\nCombined statistics for all leagues:")
        for league in all_leagues_data:
            print(f"\n{league}:")
            print(f"Matches: {len(all_leagues_data[league])}")
            print(f"Seasons: {all_leagues_data[league]['season'].nunique()}")
        
        print(f"\nAll leagues combined data saved to: {combined_output}")
        print(f"Total matches across all leagues: {len(combined_all_leagues)}")

if __name__ == '__main__':
    main()