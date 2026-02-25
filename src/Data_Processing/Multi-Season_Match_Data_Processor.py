from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import os
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
from src.Config.Config_Manager import ConfigManager
from src.Features.ELO_Rating import ELOCalculator

def prepare_match_data_for_prediction(df, league_name=None, team_history=None,
                                      elo_calculator=None, h2h_history=None,
                                      h2h_window=5):
    """
    Transform match data into a format suitable for prediction by including
    rolling statistics from previous matches for both teams.

    Supports cross-season continuity via team_history and h2h_history dicts
    that persist across seasons.

    Args:
        df: DataFrame with match data for one season
        league_name: Name of the league (added as a column)
        team_history: dict mapping team_name -> DataFrame of all past raw match rows.
                      If None, starts fresh (intra-season only).
        elo_calculator: ELOCalculator instance (persists across seasons).
                        If None, ELO features will be NaN.
        h2h_history: dict mapping frozenset({team_a, team_b}) -> list of result dicts.
                     If None, starts fresh.
        h2h_window: Number of previous H2H encounters to consider (default 5).

    Returns:
        (prediction_df, team_history, h2h_history): processed DataFrame and updated histories
    """
    if team_history is None:
        team_history = {}
    if h2h_history is None:
        h2h_history = {}

    # Convert date and sort chronologically
    df['date'] = pd.to_datetime(df['date_GMT'], format='%b %d %Y - %I:%M%p')
    df = df.sort_values('date').reset_index(drop=True)

    # Calculate goal difference and match result
    df['goal_difference'] = df['home_team_goal_count'] - df['away_team_goal_count']
    df['match_result'] = df['goal_difference'].apply(lambda x:
        'HomeWin' if x > 0 else ('AwayWin' if x < 0 else 'Draw'))

    processed_matches = []

    for idx in range(len(df)):
        current_match = df.iloc[idx]
        home_team = current_match['home_team_name']
        away_team = current_match['away_team_name']

        # Use cross-season team history if available
        home_prev = team_history.get(home_team, pd.DataFrame())
        away_prev = team_history.get(away_team, pd.DataFrame())

        # Calculate features for each team from their respective histories
        home_features = calculate_team_features(home_prev, home_team, 'home')
        away_features = calculate_team_features(away_prev, away_team, 'away')

        # --- Core match metadata ---
        match_data = {
            'date': current_match['date'],
            'season': current_match.get('season', np.nan),
            'league': league_name,
            'home_team': home_team,
            'away_team': away_team,
            'target_result': current_match['match_result'],
            'target_home_goals': current_match['home_team_goal_count'],
            'target_away_goals': current_match['away_team_goal_count'],
        }

        # --- ELO ratings (B5) — pre-match, then updated after ---
        if elo_calculator is not None:
            elo_home, elo_away, elo_diff = elo_calculator.get_prematch_elos(
                home_team, away_team
            )
        else:
            elo_home = elo_away = elo_diff = np.nan
        match_data['home_elo'] = elo_home
        match_data['away_elo'] = elo_away
        match_data['elo_diff'] = elo_diff  # positive → home team stronger

        # --- Bookmaker odds → implied probabilities (B1) ---
        odds_home = current_match.get('odds_ft_home_team_win', np.nan)
        odds_draw = current_match.get('odds_ft_draw', np.nan)
        odds_away = current_match.get('odds_ft_away_team_win', np.nan)

        if (pd.notna(odds_home) and pd.notna(odds_draw) and pd.notna(odds_away)
                and odds_home > 0 and odds_draw > 0 and odds_away > 0):
            raw_h = 1.0 / odds_home
            raw_d = 1.0 / odds_draw
            raw_a = 1.0 / odds_away
            total = raw_h + raw_d + raw_a
            match_data['implied_prob_home'] = raw_h / total
            match_data['implied_prob_draw'] = raw_d / total
            match_data['implied_prob_away'] = raw_a / total
            match_data['odds_ratio'] = odds_away / odds_home  # > 1 → home favourite
        else:
            match_data['implied_prob_home'] = np.nan
            match_data['implied_prob_draw'] = np.nan
            match_data['implied_prob_away'] = np.nan
            match_data['odds_ratio'] = np.nan

        # --- Team features with home_ / away_ prefix ---
        for key, value in home_features.items():
            match_data[f'home_{key}'] = value
        for key, value in away_features.items():
            match_data[f'away_{key}'] = value

        # --- Differential features (B3): home - away ---
        diff_pairs = [
            ('points_per_game',       'ppg'),
            ('recent_ppg_last5',      'recent_ppg'),
            ('avg_goals_scored',      'goals_scored'),
            ('avg_goals_conceded',    'goals_conceded'),
            ('avg_xg_scored',         'xg_scored'),
            ('avg_shots_per_game',    'shots'),
            ('avg_possession',        'possession'),
            ('form_trend',            'form_trend'),
        ]
        for feat_key, diff_name in diff_pairs:
            h_val = home_features.get(feat_key, np.nan)
            a_val = away_features.get(feat_key, np.nan)
            match_data[f'diff_{diff_name}'] = (
                h_val - a_val if pd.notna(h_val) and pd.notna(a_val) else np.nan
            )

        # --- Draw propensity (B7): combined tendency ---
        h_draw5 = home_features.get('draw_ratio_last5', np.nan)
        a_draw5 = away_features.get('draw_ratio_last5', np.nan)
        match_data['combined_draw_tendency'] = (
            (h_draw5 + a_draw5) / 2
            if pd.notna(h_draw5) and pd.notna(a_draw5) else np.nan
        )
        # Proxy: small ppg gap → more evenly matched → draws more likely
        h_ppg = home_features.get('points_per_game', np.nan)
        a_ppg = away_features.get('points_per_game', np.nan)
        match_data['match_competitiveness'] = (
            abs(h_ppg - a_ppg)
            if pd.notna(h_ppg) and pd.notna(a_ppg) else np.nan
        )

        # --- Head-to-Head features (B8) — PRE-match, from h2h_history ---
        h2h_key = frozenset({home_team, away_team})
        past_h2h = h2h_history.get(h2h_key, [])
        recent_h2h = past_h2h[-h2h_window:]  # last N encounters

        if recent_h2h:
            h2h_home_wins = sum(
                1 for e in recent_h2h
                if (e['home_team'] == home_team and e['result'] == 'HomeWin') or
                   (e['home_team'] == away_team and e['result'] == 'AwayWin')
            )
            h2h_away_wins = sum(
                1 for e in recent_h2h
                if (e['home_team'] == away_team and e['result'] == 'HomeWin') or
                   (e['home_team'] == home_team and e['result'] == 'AwayWin')
            )
            h2h_draws = sum(1 for e in recent_h2h if e['result'] == 'Draw')
            # Goals from perspective of current home_team
            home_goals_in_h2h = [
                e['home_goals'] if e['home_team'] == home_team else e['away_goals']
                for e in recent_h2h
            ]
            away_goals_in_h2h = [
                e['home_goals'] if e['home_team'] == away_team else e['away_goals']
                for e in recent_h2h
            ]
            match_data['h2h_home_wins']       = h2h_home_wins
            match_data['h2h_away_wins']       = h2h_away_wins
            match_data['h2h_draws']           = h2h_draws
            match_data['h2h_home_goals_avg']  = np.mean(home_goals_in_h2h)
            match_data['h2h_away_goals_avg']  = np.mean(away_goals_in_h2h)
            match_data['h2h_matches_count']   = len(recent_h2h)
        else:
            match_data['h2h_home_wins']      = 0
            match_data['h2h_away_wins']      = 0
            match_data['h2h_draws']          = 0
            match_data['h2h_home_goals_avg'] = np.nan
            match_data['h2h_away_goals_avg'] = np.nan
            match_data['h2h_matches_count']  = 0

        processed_matches.append(match_data)

        # --- Update ELO AFTER features are recorded (no leakage) ---
        if elo_calculator is not None:
            elo_calculator.update(
                home_team, away_team,
                int(current_match['home_team_goal_count']),
                int(current_match['away_team_goal_count']),
            )

        # --- Update team history AFTER building features (no leakage) ---
        match_row_df = df.iloc[[idx]].copy()
        for team in [home_team, away_team]:
            if team not in team_history:
                team_history[team] = match_row_df
            else:
                team_history[team] = pd.concat(
                    [team_history[team], match_row_df], ignore_index=True
                )

        # --- Update H2H history AFTER building features (no leakage) ---
        h2h_entry = {
            'home_team':  home_team,
            'away_team':  away_team,
            'result':     match_data['target_result'],
            'home_goals': int(current_match['home_team_goal_count']),
            'away_goals': int(current_match['away_team_goal_count']),
        }
        if h2h_key not in h2h_history:
            h2h_history[h2h_key] = [h2h_entry]
        else:
            h2h_history[h2h_key].append(h2h_entry)

    prediction_df = pd.DataFrame(processed_matches)
    prediction_df = prediction_df.replace([np.inf, -np.inf], np.nan)
    # H2H count is 0 for first meetings — don't drop these rows
    h2h_cols = ['h2h_home_goals_avg', 'h2h_away_goals_avg']
    non_h2h_cols = [c for c in prediction_df.columns if c not in h2h_cols]
    prediction_df = prediction_df.dropna(subset=non_h2h_cols)
    # Fill NaN H2H averages with 0 (no prior history)
    prediction_df[h2h_cols] = prediction_df[h2h_cols].fillna(0)

    return prediction_df, team_history, h2h_history


def calculate_team_features(previous_matches, team_name, venue_type):
    """
    Calculate comprehensive team statistics based on previous matches.

    Computes multi-window rolling stats (3, 5, 10 matches), xG rolling
    averages, draw propensity, and a form trend indicator.

    Args:
        previous_matches: DataFrame of all past raw match rows for this team
        team_name: Name of the team
        venue_type: 'home' or 'away' (used for venue-specific stats)

    Returns:
        dict of feature_name -> value
    """
    if len(previous_matches) == 0:
        return get_empty_features()

    # Filter to team's matches (both as home and away)
    team_matches = previous_matches[
        (previous_matches['home_team_name'] == team_name) |
        (previous_matches['away_team_name'] == team_name)
    ].copy()

    if len(team_matches) == 0:
        return get_empty_features()

    # --- Unified team perspective columns ---
    team_matches['is_home'] = team_matches['home_team_name'] == team_name

    # Goals
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

    # xG (B2) — columns may be missing or NaN in older data
    if 'team_a_xg' in team_matches.columns and 'team_b_xg' in team_matches.columns:
        team_matches['team_xg'] = np.where(
            team_matches['is_home'],
            team_matches['team_a_xg'],
            team_matches['team_b_xg']
        )
        team_matches['opp_xg'] = np.where(
            team_matches['is_home'],
            team_matches['team_b_xg'],
            team_matches['team_a_xg']
        )
    else:
        team_matches['team_xg'] = np.nan
        team_matches['opp_xg'] = np.nan

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

    # Match result from the team's perspective (for draw ratio)
    team_matches['team_result'] = np.where(
        team_matches['team_goals'] > team_matches['opponent_goals'], 'Win',
        np.where(team_matches['team_goals'] == team_matches['opponent_goals'], 'Draw', 'Loss')
    )

    # Venue-specific matches
    venue_matches = previous_matches[
        previous_matches[f'{venue_type}_team_name'] == team_name
    ]

    n = len(team_matches)
    total_shots = team_matches['team_shots'].sum()
    total_corners = team_matches['team_corners'].sum()
    total_possession = team_matches['team_possession'].sum()
    total_cards = (
        team_matches['team_first_half_cards'].sum() +
        team_matches['team_second_half_cards'].sum()
    )
    wins_all  = (team_matches['team_goals'] > team_matches['opponent_goals']).sum()
    draws_all = (team_matches['team_goals'] == team_matches['opponent_goals']).sum()
    losses_all = (team_matches['team_goals'] < team_matches['opponent_goals']).sum()

    avg_goals_scored = team_matches['team_goals'].mean()
    avg_xg_scored = team_matches['team_xg'].mean()

    # --- Multi-window rolling stats (B6) ---
    def window_stats(window):
        recent = team_matches.tail(window)
        k = len(recent)
        if k == 0:
            return {
                f'recent_ppg_last{window}': np.nan,
                f'recent_goals_scored_last{window}': np.nan,
                f'recent_goals_conceded_last{window}': np.nan,
                f'recent_clean_sheets_last{window}': np.nan,
                f'draw_ratio_last{window}': np.nan,
            }
        wins_r  = (recent['team_goals'] > recent['opponent_goals']).sum()
        draws_r = (recent['team_goals'] == recent['opponent_goals']).sum()
        return {
            f'recent_ppg_last{window}': (wins_r * 3 + draws_r) / k,
            f'recent_goals_scored_last{window}': recent['team_goals'].mean(),
            f'recent_goals_conceded_last{window}': recent['opponent_goals'].mean(),
            f'recent_clean_sheets_last{window}': (recent['opponent_goals'] == 0).mean(),
            f'draw_ratio_last{window}': (recent['team_result'] == 'Draw').mean(),
        }

    multi_window_feats = {}
    for w in [3, 5, 10]:
        multi_window_feats.update(window_stats(w))

    # Convenience aliases expected by differential and draw propensity logic
    multi_window_feats['recent_ppg_last5'] = multi_window_feats.get('recent_ppg_last5', np.nan)
    multi_window_feats['draw_ratio_last5'] = multi_window_feats.get('draw_ratio_last5', np.nan)

    # Form trend: positive = improving, negative = declining
    ppg3  = multi_window_feats.get('recent_ppg_last3', np.nan)
    ppg10 = multi_window_feats.get('recent_ppg_last10', np.nan)
    form_trend = (ppg3 - ppg10) if pd.notna(ppg3) and pd.notna(ppg10) else np.nan

    features = {
        # Overall Performance
        'games_played': n,
        'wins': wins_all,
        'draws': draws_all,
        'losses': losses_all,
        'points_per_game': (wins_all * 3 + draws_all) / n,

        # Goal Scoring Patterns
        'avg_goals_scored': avg_goals_scored,
        'avg_goals_conceded': team_matches['opponent_goals'].mean(),
        'avg_goal_diff': (team_matches['team_goals'] - team_matches['opponent_goals']).mean(),
        'goals_scored_first_half_ratio': (
            team_matches['team_goals_ht'] / team_matches['team_goals']
        ).mean(),
        'goals_conceded_first_half_ratio': (
            team_matches['opponent_goals_ht'] / team_matches['opponent_goals']
        ).mean(),

        # xG Rolling (B2)
        'avg_xg_scored': avg_xg_scored,
        'avg_xg_conceded': team_matches['opp_xg'].mean(),
        'xg_vs_goals_diff': avg_goals_scored - avg_xg_scored,  # +: lucky, -: unlucky

        # Shooting Efficiency
        'shot_conversion_rate': (
            team_matches['team_goals'].sum() / total_shots if total_shots > 0 else 0
        ),
        'shots_on_target_ratio': (
            team_matches['team_shots_on_target'].sum() / total_shots if total_shots > 0 else 0
        ),
        'avg_shots_per_game': team_matches['team_shots'].mean(),
        'avg_shots_on_target': team_matches['team_shots_on_target'].mean(),

        # Game Control
        'avg_possession': team_matches['team_possession'].mean(),
        'possession_efficiency': (
            team_matches['team_goals'].sum() / total_possession * 100
            if total_possession > 0 else 0
        ),
        'avg_corners_for': team_matches['team_corners'].mean(),
        'corner_efficiency': (
            team_matches['team_goals'].sum() / total_corners if total_corners > 0 else 0
        ),

        # Discipline
        'avg_fouls_committed': team_matches['team_fouls'].mean(),
        'avg_yellows': team_matches['team_yellows'].mean(),
        'avg_reds': team_matches['team_reds'].mean(),
        'cards_first_half_ratio': (
            team_matches['team_first_half_cards'].sum() / total_cards
            if total_cards > 0 else 0
        ),

        # Venue-specific Performance
        'venue_games': len(venue_matches),
        'venue_win_ratio': (
            (venue_matches[f'{venue_type}_team_goal_count'] >
             venue_matches[f'{"away" if venue_type == "home" else "home"}_team_goal_count']).sum()
            / len(venue_matches) if len(venue_matches) > 0 else 0
        ),
        'venue_goals_avg': (
            venue_matches[f'{venue_type}_team_goal_count'].mean()
            if len(venue_matches) > 0 else 0
        ),
        'venue_conceded_avg': (
            venue_matches[f'{"away" if venue_type == "home" else "home"}_team_goal_count'].mean()
            if len(venue_matches) > 0 else 0
        ),

        # Form Indicators
        'clean_sheets_ratio': (team_matches['opponent_goals'] == 0).mean(),
        'scoring_ratio': (team_matches['team_goals'] > 0).mean(),
        'comeback_ratio': (
            (team_matches['opponent_goals_ht'] > team_matches['team_goals_ht']) &
            (team_matches['team_goals'] > team_matches['opponent_goals'])
        ).sum() / n,
        'lead_loss_ratio': (
            (team_matches['team_goals_ht'] > team_matches['opponent_goals_ht']) &
            (team_matches['opponent_goals'] > team_matches['team_goals'])
        ).sum() / n,

        # Form Trend (B6)
        'form_trend': form_trend,
    }

    # Merge multi-window stats
    features.update(multi_window_feats)

    return features


def get_empty_features():
    """Return dictionary of features with null values for teams with no history."""
    base = {
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
        # xG
        'avg_xg_scored': np.nan,
        'avg_xg_conceded': np.nan,
        'xg_vs_goals_diff': np.nan,
        # Shooting
        'shot_conversion_rate': np.nan,
        'shots_on_target_ratio': np.nan,
        'avg_shots_per_game': np.nan,
        'avg_shots_on_target': np.nan,
        # Control
        'avg_possession': np.nan,
        'possession_efficiency': np.nan,
        'avg_corners_for': np.nan,
        'corner_efficiency': np.nan,
        # Discipline
        'avg_fouls_committed': np.nan,
        'avg_yellows': np.nan,
        'avg_reds': np.nan,
        'cards_first_half_ratio': np.nan,
        # Venue
        'venue_games': 0,
        'venue_win_ratio': np.nan,
        'venue_goals_avg': np.nan,
        'venue_conceded_avg': np.nan,
        # Form indicators
        'clean_sheets_ratio': np.nan,
        'scoring_ratio': np.nan,
        'comeback_ratio': np.nan,
        'lead_loss_ratio': np.nan,
        # Form trend
        'form_trend': np.nan,
    }
    # Multi-window features
    for w in [3, 5, 10]:
        base[f'recent_ppg_last{w}'] = np.nan
        base[f'recent_goals_scored_last{w}'] = np.nan
        base[f'recent_goals_conceded_last{w}'] = np.nan
        base[f'recent_clean_sheets_last{w}'] = np.nan
        base[f'draw_ratio_last{w}'] = np.nan
    return base


def process_season(
    input_file: Path,
    output_folder: Path,
    league: str,
    team_history: dict = None,
    elo_calculator: 'ELOCalculator' = None,
    h2h_history: dict = None,
    h2h_window: int = 5,
) -> tuple:
    """
    Process a single season's data file and save the processed version.

    Args:
        input_file: Path to the input CSV file
        output_folder: Path to save the processed data
        league: Name of the league being processed
        team_history: Cross-season team history dict (modified in-place and returned)
        elo_calculator: ELOCalculator shared across seasons for this league
        h2h_history: Cross-season head-to-head history dict
        h2h_window: Number of H2H encounters to use (default 5)

    Returns:
        (DataFrame with processed data, updated team_history, updated h2h_history)
        Returns (None, team_history, h2h_history) on error.
    """
    if team_history is None:
        team_history = {}
    if h2h_history is None:
        h2h_history = {}
    try:
        filename = input_file.name
        season_match = filename.split('matches-')[1].split('-stats')[0]

        print(f"Processing {league} season {season_match}...")
        df = pd.read_csv(input_file)
        df['season'] = season_match
        df['league'] = league

        prediction_data, team_history, h2h_history = prepare_match_data_for_prediction(
            df, league_name=league, team_history=team_history,
            elo_calculator=elo_calculator,
            h2h_history=h2h_history, h2h_window=h2h_window,
        )

        os.makedirs(output_folder, exist_ok=True)
        season_clean = season_match.replace('-to-', '_to_')
        output_file = output_folder / f'processed_matches_{season_clean}.csv'
        prediction_data.to_csv(output_file, index=False)

        print(f"Processed {league} {season_match} season: {len(prediction_data)} matches")
        return prediction_data, team_history, h2h_history

    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        return None, team_history, h2h_history


def process_league_data(league: str, config_manager: ConfigManager) -> pd.DataFrame:
    """
    Process all seasons for a specific league.
    Maintains cross-season rolling stat continuity via a shared team_history dict.

    Args:
        league: Name of the league to process
        config_manager: ConfigManager instance

    Returns:
        Combined DataFrame for all processed seasons, or None on failure.
    """
    league_config = config_manager.get_config_value('data_paths', 'leagues', league)
    if not league_config:
        print(f"No configuration found for {league}")
        return None

    input_path = Path(league_config['raw_data'])
    output_path = Path(league_config['processed_data'])
    start_year = league_config['start_year']
    end_year = league_config['end_year']

    pattern = config_manager.get_config_value('league_patterns', league)

    print(f"\nProcessing {league} data...")
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Processing seasons from {start_year} to {end_year}")

    all_seasons_data = []
    # Shared history and ELO persist across seasons (cross-season continuity, A2)
    global_team_history = {}
    global_h2h_history  = {}
    elo_calc = ELOCalculator(initial_elo=1500, k_factor=20, home_advantage=100)

    # H2H window from config (default 5)
    proc_params = config_manager.get_config_value('processing_params', default={})
    h2h_window = proc_params.get('h2h_window', 5)

    for input_file in sorted(input_path.glob(pattern)):
        try:
            season_start = int(input_file.name.split('-')[-4])
            if start_year <= season_start < end_year:
                season_data, global_team_history, global_h2h_history = process_season(
                    input_file, output_path, league,
                    global_team_history, elo_calc,
                    h2h_history=global_h2h_history, h2h_window=h2h_window,
                )
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


def _process_league_worker(args: tuple) -> tuple:
    """
    Top-level worker for ProcessPoolExecutor — must be defined at module level
    so it can be pickled and sent to child processes.

    Each worker creates its own ConfigManager to avoid sharing state.
    """
    league, config_path = args
    config_manager = ConfigManager(config_path)
    league_data = process_league_data(league, config_manager)
    return league, league_data


def main():
    CONFIG_PATH = "src/Config/data_processing_config.yaml"
    config_manager = ConfigManager(CONFIG_PATH)
    leagues = list(config_manager.get_config_value('data_paths', 'leagues').keys())

    n_workers = min(len(leagues), os.cpu_count() or 4)
    print(f"Lancement du traitement : {len(leagues)} ligues sur {n_workers} workers\n")

    all_leagues_data = {}

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_process_league_worker, (league, CONFIG_PATH)): league
            for league in leagues
        }
        for future in as_completed(futures):
            league_name = futures[future]
            try:
                league, league_data = future.result()
                if league_data is not None:
                    all_leagues_data[league] = league_data
                    print(f"✓ {league} terminé ({len(league_data)} matchs)")
                else:
                    print(f"✗ {league_name} : aucune donnée retournée")
            except Exception as exc:
                print(f"✗ {league_name} : erreur — {exc}")

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
        print(f"Total columns: {len(combined_all_leagues.columns)}")
        print(f"\nNew feature columns added:")
        new_cols = [c for c in combined_all_leagues.columns
                    if any(c.startswith(p) for p in
                           ['implied_prob', 'odds_ratio', 'diff_', 'combined_draw',
                            'match_comp', 'home_avg_xg', 'away_avg_xg',
                            'home_form_trend', 'away_form_trend',
                            'home_recent_ppg_last', 'away_recent_ppg_last'])]
        for col in new_cols:
            print(f"  {col}")


if __name__ == '__main__':
    main()
