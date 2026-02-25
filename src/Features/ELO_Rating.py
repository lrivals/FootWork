"""
ELO Rating system for football teams.

ELO is updated after every match:
    ELO_new = ELO_old + K × (score_actual - score_expected)

where:
    score_expected = 1 / (1 + 10^((ELO_opponent - ELO_team + home_advantage) / 400))
    score_actual   = 1 (win), 0.5 (draw), 0 (loss)

One ELOCalculator instance per league to avoid cross-league comparisons.
The instance persists across seasons to maintain cross-season continuity.
"""


class ELOCalculator:
    """
    Tracks and updates ELO ratings for all teams in a league.

    Usage per match:
        elo_home, elo_away = calc.get_prematch_elos(home_team, away_team)
        # → use these as features BEFORE updating
        calc.update(home_team, away_team, home_goals, away_goals)
    """

    def __init__(self, initial_elo: float = 1500, k_factor: float = 20,
                 home_advantage: float = 100):
        """
        Args:
            initial_elo:     Starting ELO for any team with no history.
            k_factor:        Sensitivity of ELO updates (higher = faster changes).
                             20 is standard for club football.
            home_advantage:  ELO bonus added to the home team's expected score
                             calculation. Typically 60–100 for football.
        """
        self.ratings: dict[str, float] = {}
        self.initial_elo = initial_elo
        self.k_factor = k_factor
        self.home_advantage = home_advantage

    def get_elo(self, team: str) -> float:
        """Return current ELO for a team (initial_elo if unseen)."""
        return self.ratings.get(team, self.initial_elo)

    def expected_score(self, elo_team: float, elo_opponent: float,
                       team_is_home: bool = False) -> float:
        """
        Compute the expected score (win probability proxy) for a team.

        The home team gets a bonus of `home_advantage` ELO points added
        before computing the expected score.
        """
        bonus = self.home_advantage if team_is_home else -self.home_advantage
        return 1.0 / (1.0 + 10.0 ** ((elo_opponent - elo_team - bonus) / 400.0))

    def get_prematch_elos(self, home_team: str, away_team: str) -> tuple:
        """
        Return (elo_home, elo_away, elo_diff) BEFORE the match is processed.
        Call this first to get the features, then call update().
        """
        elo_home = self.get_elo(home_team)
        elo_away = self.get_elo(away_team)
        return elo_home, elo_away, elo_home - elo_away

    def update(self, home_team: str, away_team: str,
               home_goals: int, away_goals: int) -> None:
        """
        Update ELO ratings after a match result.

        Args:
            home_team:  Name of the home team.
            away_team:  Name of the away team.
            home_goals: Goals scored by the home team.
            away_goals: Goals scored by the away team.
        """
        elo_home = self.get_elo(home_team)
        elo_away = self.get_elo(away_team)

        exp_home = self.expected_score(elo_home, elo_away, team_is_home=True)
        exp_away = 1.0 - exp_home

        if home_goals > away_goals:
            score_home, score_away = 1.0, 0.0
        elif home_goals < away_goals:
            score_home, score_away = 0.0, 1.0
        else:
            score_home, score_away = 0.5, 0.5

        self.ratings[home_team] = elo_home + self.k_factor * (score_home - exp_home)
        self.ratings[away_team] = elo_away + self.k_factor * (score_away - exp_away)

    def snapshot(self) -> dict:
        """Return a copy of all current ratings (useful for debugging)."""
        return dict(self.ratings)
