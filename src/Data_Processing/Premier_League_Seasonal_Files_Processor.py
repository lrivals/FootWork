import pandas as pd
import numpy as np
import os

def process_seasonal_files():
    # List to store processed season DataFrames
    processed_seasons = []
    
    # Process each season file
    for i in range(12):
        # Construct input and output file names
        input_file = f'premier-league-matches-2007-2023/england-premier-league-matches-20{12+i}-to-20{13+i}-stats.csv'
        output_file = f'seasonal_performance_20{12+i}_to_20{13+i}.csv'
        
        try:
            # Read the seasonal CSV file
            df = pd.read_csv(input_file)
            
            # Determine match result if not already present
            if 'match_result' not in df.columns:
                # Ensure goal_difference column exists
                if 'goal_difference' not in df.columns:
                    # Calculate goal difference (home team goals - away team goals)
                    df['goal_difference'] = df['home_team_goal_count'] - df['away_team_goal_count']
                
                # Define result determination function
                def determine_result(goal_diff):
                    if goal_diff > 0:
                        return 'HomeWin'
                    elif goal_diff < 0:
                        return 'AwayWin'
                    else:
                        return 'Draw'
                
                # Apply result determination
                df['match_result'] = df['goal_difference'].apply(determine_result)
            
            # Add season identifier
            df['season'] = f'20{12+i}-20{13+i}'
            
            # Process team performance for this season
            season_performance = calculate_seasonal_performance(df)
            
            # Save seasonal performance to CSV
            season_performance.to_csv(output_file, index=False)
            
            # Append to list of processed seasons
            processed_seasons.append(season_performance)
            
            print(f"Processed season {input_file}")
        
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Check if any seasons were processed
    if not processed_seasons:
        print("No seasons could be processed. Check your input files.")
        return None
    
    # Combine all seasonal performances
    combined_performance = pd.concat(processed_seasons, ignore_index=True)
    combined_performance.to_csv('all_seasons_performance.csv', index=False)
    
    return combined_performance

def calculate_seasonal_performance(df):
    """
    Calculate comprehensive performance metrics for a specific season
    """
    # Performance metrics calculation
    team_performance = {}
    
    # Unique teams in the season
    teams = set(df['home_team_name'].unique()).union(set(df['away_team_name'].unique()))
    
    for team in teams:
        # Home and away matches for the team
        home_matches = df[df['home_team_name'] == team]
        away_matches = df[df['away_team_name'] == team]
        
        # Aggregate performance metrics
        team_stats = {
            'team_name': team,
            'season': df['season'].iloc[0],  # Add season identifier
            
            # Match outcomes
            'total_matches': len(home_matches) + len(away_matches),
            'home_matches': len(home_matches),
            'away_matches': len(away_matches),
            
            # Goal statistics
            'total_goals_scored': home_matches['home_team_goal_count'].sum() + away_matches['away_team_goal_count'].sum(),
            'total_goals_conceded': home_matches['away_team_goal_count'].sum() + away_matches['home_team_goal_count'].sum(),
            
            # Win/Draw/Loss breakdown
            'home_wins': len(home_matches[home_matches['match_result'] == 'HomeWin']),
            'away_wins': len(away_matches[away_matches['match_result'] == 'AwayWin']),
            'draws': len(df[((df['home_team_name'] == team) | (df['away_team_name'] == team)) & (df['match_result'] == 'Draw')]),
            
            # Performance percentages
            'win_percentage': calculate_win_percentage(home_matches, away_matches),
            'draw_percentage': calculate_draw_percentage(home_matches, away_matches),
            
            # Advanced metrics (add error handling)
            'avg_possession': calculate_average_metric(home_matches, away_matches, 'possession') if 'home_team_possession' in df.columns and 'away_team_possession' in df.columns else 0,
            'avg_shots': calculate_average_metric(home_matches, away_matches, 'shots') if 'home_team_shots' in df.columns and 'away_team_shots' in df.columns else 0,
            'avg_shots_on_target': calculate_average_metric(home_matches, away_matches, 'shots_on_target') if 'home_team_shots_on_target' in df.columns and 'away_team_shots_on_target' in df.columns else 0,
            'avg_corner_count': calculate_average_metric(home_matches, away_matches, 'corner_count') if 'home_team_corner_count' in df.columns and 'away_team_corner_count' in df.columns else 0,
            'avg_yellow_cards': calculate_average_metric(home_matches, away_matches, 'yellow_cards') if 'home_team_yellow_cards' in df.columns and 'away_team_yellow_cards' in df.columns else 0,
            
            # Goal-related metrics
            'goals_per_match': (home_matches['home_team_goal_count'].sum() + away_matches['away_team_goal_count'].sum()) / (len(home_matches) + len(away_matches)),
            'goals_conceded_per_match': (home_matches['away_team_goal_count'].sum() + away_matches['home_team_goal_count'].sum()) / (len(home_matches) + len(away_matches))
        }
        
        team_performance[team] = team_stats
    
    # Convert to DataFrame
    performance_df = pd.DataFrame.from_dict(team_performance, orient='index').reset_index(drop=True)
    
    return performance_df

def calculate_win_percentage(home_matches, away_matches):
    """Calculate team's win percentage"""
    home_wins = len(home_matches[home_matches['match_result'] == 'HomeWin'])
    away_wins = len(away_matches[away_matches['match_result'] == 'AwayWin'])
    total_matches = len(home_matches) + len(away_matches)
    return (home_wins + away_wins) / total_matches * 100 if total_matches > 0 else 0

def calculate_draw_percentage(home_matches, away_matches):
    """Calculate team's draw percentage"""
    draws = len(home_matches[home_matches['match_result'] == 'Draw']) + \
            len(away_matches[away_matches['match_result'] == 'Draw'])
    total_matches = len(home_matches) + len(away_matches)
    return draws / total_matches * 100 if total_matches > 0 else 0

def calculate_average_metric(home_matches, away_matches, metric):
    """Calculate average of a specific metric"""
    home_metric = home_matches[f'home_team_{metric}'].mean()
    away_metric = away_matches[f'away_team_{metric}'].mean()
    return (home_metric + away_metric) / 2 if len(home_matches) + len(away_matches) > 0 else 0

def create_cross_seasonal_analysis():
    """
    Create a comprehensive analysis across all seasons
    """
    try:
        # Read the all seasons performance file
        all_seasons_df = pd.read_csv('all_seasons_performance.csv')
        
        # Group by team name and calculate overall performance
        overall_team_performance = all_seasons_df.groupby('team_name').agg({
            'total_matches': 'sum',
            'total_goals_scored': 'sum',
            'total_goals_conceded': 'sum',
            'win_percentage': 'mean',
            'draw_percentage': 'mean',
            'avg_possession': 'mean',
            'avg_shots': 'mean',
            'avg_shots_on_target': 'mean',
            'avg_corner_count': 'mean'
        }).reset_index()
        
        # Save cross-seasonal analysis
        overall_team_performance.to_csv('cross_seasonal_team_performance.csv', index=False)
        
        return overall_team_performance
    except FileNotFoundError:
        print("No seasonal performance file found. Run process_seasonal_files() first.")
        return None

# Main execution
if __name__ == '__main__':
    # Process individual seasonal files
    seasonal_performances = process_seasonal_files()
    
    # Create cross-seasonal analysis if seasonal performances were processed
    if seasonal_performances is not None:
        cross_seasonal_performance = create_cross_seasonal_analysis()
        
        print("Seasonal and cross-seasonal analysis completed.")