"""
src/Analysis/Betting_Backtest.py

Backtesting module for football match betting strategies.
Loads per-match prediction CSVs produced by the multiclass pipeline and
simulates 7 betting strategies on the test set (2022-2024).

Usage:
    python src/Analysis/Betting_Backtest.py \\
        --predictions results/All_Leagues/Multiclass_Target/<run>/predictions_CatBoost.csv \\
        --bankroll 1000 \\
        --output results/Backtest/ \\
        [--league Spain]
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


class BettingBacktester:
    """
    Simulates 7 betting strategies on historical football match predictions.

    Input DataFrame expected columns (from save_prediction_csv in pipeline):
        date, home_team, away_team, target_result, league (optional)
        prob_homewin, prob_draw, prob_awaywin  (calibrated model probabilities)
        pred_default, pred_opt                (argmax / threshold-optimized prediction)
        raw_odds_home, raw_odds_draw, raw_odds_away  (bookmaker decimal odds)
        ev_home, ev_draw, ev_away             (expected value per outcome)
        kelly_home, kelly_draw, kelly_away    (quarter Kelly fraction)
    """

    STRATEGIES = [
        'always_bet_prediction',
        'value_bets_ev5',
        'value_bets_ev10',
        'kelly_ev5',
        'high_confidence',
        'draw_only_ev5',
        'draw_only_ev10',
    ]

    # Flat stake as % of bankroll (used by flat strategies)
    FLAT_STAKE_PCT = 0.01  # 1% per bet

    def __init__(self, predictions_df: pd.DataFrame):
        self.df = predictions_df.copy()
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df = self.df.sort_values('date').reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_ev(self, row, outcome):
        mapping = {
            'HomeWin': row.get('ev_home', np.nan),
            'Draw':    row.get('ev_draw', np.nan),
            'AwayWin': row.get('ev_away', np.nan),
        }
        return mapping.get(outcome, np.nan)

    def _get_kelly(self, row, outcome):
        mapping = {
            'HomeWin': row.get('kelly_home', 0.0),
            'Draw':    row.get('kelly_draw', 0.0),
            'AwayWin': row.get('kelly_away', 0.0),
        }
        return mapping.get(outcome, 0.0)

    def _get_odds(self, row, outcome):
        mapping = {
            'HomeWin': row.get('raw_odds_home', np.nan),
            'Draw':    row.get('raw_odds_draw',  np.nan),
            'AwayWin': row.get('raw_odds_away',  np.nan),
        }
        return mapping.get(outcome, np.nan)

    def _best_ev_outcome(self, row):
        """Return (outcome, ev) with highest EV among the three outcomes."""
        candidates = []
        for outcome in ['HomeWin', 'Draw', 'AwayWin']:
            ev = self._get_ev(row, outcome)
            if pd.notna(ev):
                candidates.append((outcome, ev))
        if not candidates:
            return None, np.nan
        return max(candidates, key=lambda x: x[1])

    def _select_bet(self, row, strategy):
        """
        Determine which outcome to bet on and the stake fraction.

        Returns:
            (outcome, stake_fraction) or (None, 0) if no bet.
            stake_fraction is a fraction of current bankroll.
        """
        if strategy == 'always_bet_prediction':
            pred = row.get('pred_opt') or row.get('pred_default')
            if pd.isna(pred) or pred not in ('HomeWin', 'Draw', 'AwayWin'):
                return None, 0.0
            return pred, self.FLAT_STAKE_PCT

        elif strategy in ('value_bets_ev5', 'value_bets_ev10'):
            ev_thr = 0.05 if strategy == 'value_bets_ev5' else 0.10
            outcome, best_ev = self._best_ev_outcome(row)
            if outcome is None or best_ev < ev_thr:
                return None, 0.0
            return outcome, self.FLAT_STAKE_PCT

        elif strategy == 'kelly_ev5':
            # Bet on the outcome with highest positive EV >= 5%, using Kelly fraction
            best_outcome, best_ev = self._best_ev_outcome(row)
            if best_outcome is None or best_ev < 0.05:
                return None, 0.0
            kelly = self._get_kelly(row, best_outcome)
            if kelly <= 0:
                return None, 0.0
            return best_outcome, kelly

        elif strategy == 'high_confidence':
            probs = {
                'HomeWin': row.get('prob_homewin', 0.0) or 0.0,
                'Draw':    row.get('prob_draw',    0.0) or 0.0,
                'AwayWin': row.get('prob_awaywin', 0.0) or 0.0,
            }
            best_outcome = max(probs, key=probs.get)
            best_prob    = probs[best_outcome]
            pred_def = row.get('pred_default')
            pred_opt = row.get('pred_opt')
            if best_prob >= 0.55 and pred_def == pred_opt == best_outcome:
                return best_outcome, self.FLAT_STAKE_PCT
            return None, 0.0

        elif strategy in ('draw_only_ev5', 'draw_only_ev10'):
            ev_thr = 0.05 if strategy == 'draw_only_ev5' else 0.10
            ev_draw = row.get('ev_draw', np.nan)
            if pd.isna(ev_draw) or ev_draw < ev_thr:
                return None, 0.0
            return 'Draw', self.FLAT_STAKE_PCT

        return None, 0.0

    # ------------------------------------------------------------------
    # Run a single strategy
    # ------------------------------------------------------------------

    def run_strategy(self, strategy, bankroll=1000):
        """
        Simulate `strategy` chronologically and return a bets DataFrame.

        Columns: date, home_team, away_team, league, bet_on, odds,
                 stake, actual, won, profit, bankroll_after
        """
        bets = []
        current_bankroll = float(bankroll)

        for _, row in self.df.iterrows():
            outcome, stake_fraction = self._select_bet(row, strategy)
            if outcome is None or stake_fraction <= 0:
                continue

            odds = self._get_odds(row, outcome)
            if pd.isna(odds) or odds <= 1.0:
                continue

            stake   = current_bankroll * stake_fraction
            actual  = row.get('target_result')
            won     = (actual == outcome)
            profit  = stake * (odds - 1) if won else -stake
            current_bankroll += profit

            bets.append({
                'date':           row.get('date'),
                'home_team':      row.get('home_team', ''),
                'away_team':      row.get('away_team', ''),
                'league':         row.get('league', 'Unknown'),
                'bet_on':         outcome,
                'odds':           round(odds, 3),
                'stake':          round(stake, 4),
                'actual':         actual,
                'won':            won,
                'profit':         round(profit, 4),
                'bankroll_after': round(current_bankroll, 4),
            })

        return pd.DataFrame(bets)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_metrics(self, bets_df, initial_bankroll=1000):
        """Compute ROI, win rate, max drawdown, Sharpe, P&L breakdowns."""
        if len(bets_df) == 0:
            return {
                'n_bets': 0, 'win_rate': 0.0, 'roi': 0.0,
                'total_staked': 0.0, 'total_profit': 0.0,
                'final_bankroll': float(initial_bankroll),
                'max_drawdown_pct': 0.0, 'sharpe': np.nan,
                'pnl_by_outcome': None, 'pnl_by_league': None,
            }

        n_bets        = len(bets_df)
        wins          = int(bets_df['won'].sum())
        total_staked  = float(bets_df['stake'].sum())
        total_profit  = float(bets_df['profit'].sum())
        roi           = (total_profit / total_staked * 100) if total_staked > 0 else 0.0

        # Max drawdown from peak bankroll
        bk = bets_df['bankroll_after']
        rolling_max   = bk.cummax()
        drawdowns     = (bk - rolling_max) / rolling_max * 100
        max_drawdown  = float(drawdowns.min())

        # Monthly Sharpe (annualised)
        sharpe = np.nan
        if 'date' in bets_df.columns and pd.api.types.is_datetime64_any_dtype(bets_df['date']):
            monthly_pnl = (
                bets_df.set_index('date')['profit']
                .resample('ME').sum()
            )
            if len(monthly_pnl) >= 2 and monthly_pnl.std() > 0:
                sharpe = float(monthly_pnl.mean() / monthly_pnl.std() * np.sqrt(12))

        pnl_by_outcome = bets_df.groupby('bet_on')['profit'].agg(['sum', 'count', 'mean'])
        pnl_by_league  = None
        if 'league' in bets_df.columns:
            pnl_by_league = bets_df.groupby('league')['profit'].agg(['sum', 'count', 'mean'])

        return {
            'n_bets':           n_bets,
            'win_rate':         wins / n_bets,
            'roi':              roi,
            'total_staked':     total_staked,
            'total_profit':     total_profit,
            'final_bankroll':   float(bk.iloc[-1]),
            'max_drawdown_pct': max_drawdown,
            'sharpe':           sharpe,
            'pnl_by_outcome':   pnl_by_outcome,
            'pnl_by_league':    pnl_by_league,
        }

    # ------------------------------------------------------------------
    # Run all strategies
    # ------------------------------------------------------------------

    def run_all_strategies(self, bankroll=1000):
        """Run all 5 strategies and return dict strategy → {bets_df, metrics}."""
        all_results = {}
        for strategy in self.STRATEGIES:
            bets_df = self.run_strategy(strategy, bankroll=bankroll)
            metrics = self.compute_metrics(bets_df, initial_bankroll=bankroll)
            all_results[strategy] = {'bets_df': bets_df, 'metrics': metrics}
            sharpe_s = f"{metrics['sharpe']:.2f}" if pd.notna(metrics.get('sharpe')) else "N/A"
            print(
                f"  {strategy:<28}: {metrics['n_bets']:>5} bets  "
                f"win={metrics['win_rate']:>5.1%}  "
                f"ROI={metrics['roi']:>+6.1f}%  "
                f"P&L={metrics['total_profit']:>+8.0f}  "
                f"Sharpe={sharpe_s}"
            )
        return all_results

    # ------------------------------------------------------------------
    # Report + plots
    # ------------------------------------------------------------------

    def save_report(self, all_results, output_dir, initial_bankroll=1000, league_filter='All'):
        """Save text report + bankroll evolution plot + P&L breakdown chart."""
        os.makedirs(output_dir, exist_ok=True)

        # ── Text report ────────────────────────────────────────────────────
        report_path = os.path.join(output_dir, 'backtest_report.txt')
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("BETTING BACKTEST REPORT — FootWork\n")
            f.write(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"Test set  : 2022-2024  |  Initial bankroll : {initial_bankroll}\n")
            f.write(f"Flat stake: {self.FLAT_STAKE_PCT:.0%} per bet\n")
            f.write(f"League    : {league_filter}\n")
            f.write("=" * 80 + "\n\n")

            # Summary table
            f.write("=== STRATEGY SUMMARY ===\n")
            f.write(f"{'Strategy':<28} {'Bets':>6} {'WinRate':>8} {'ROI%':>7} "
                    f"{'P&L':>9} {'MaxDD%':>8} {'Sharpe':>8}\n")
            f.write("-" * 80 + "\n")
            for strat, res in all_results.items():
                m = res['metrics']
                sharpe_s = f"{m['sharpe']:.2f}" if pd.notna(m.get('sharpe')) else "   N/A"
                f.write(
                    f"{strat:<28} {m['n_bets']:>6} "
                    f"{m['win_rate']:>8.1%} "
                    f"{m['roi']:>7.1f}% "
                    f"{m['total_profit']:>+9.0f} "
                    f"{m['max_drawdown_pct']:>8.1f}% "
                    f"{sharpe_s:>8}\n"
                )
            f.write("\n")

            # P&L by outcome type per strategy
            f.write("=== P&L BY OUTCOME TYPE ===\n")
            for strat, res in all_results.items():
                m = res['metrics']
                if m['n_bets'] == 0:
                    continue
                f.write(f"\n{strat}:\n")
                pnl = m.get('pnl_by_outcome')
                if pnl is not None and len(pnl) > 0:
                    f.write(f"  {'Outcome':<12} {'P&L':>10} {'N bets':>8} {'Avg/bet':>10}\n")
                    for outcome, row in pnl.iterrows():
                        f.write(f"  {outcome:<12} {row['sum']:>+10.0f} "
                                f"{int(row['count']):>8} {row['mean']:>+10.2f}\n")
            f.write("\n")

            # P&L by league for all strategies
            f.write("=== P&L BY LEAGUE (all strategies) ===\n")
            for strat, res in all_results.items():
                m = res['metrics']
                if m['n_bets'] == 0:
                    continue
                pnl_league = m.get('pnl_by_league')
                if pnl_league is None or len(pnl_league) == 0:
                    continue
                f.write(f"\n{strat}:\n")
                f.write(f"  {'League':<22} {'P&L':>10} {'N bets':>8} {'Avg/bet':>10}\n")
                f.write("  " + "-" * 54 + "\n")
                for league, row in pnl_league.sort_values('sum', ascending=False).iterrows():
                    f.write(f"  {str(league):<22} {row['sum']:>+10.0f} "
                            f"{int(row['count']):>8} {row['mean']:>+10.2f}\n")

        print(f"Report → {report_path}")

        # ── Bankroll evolution ─────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(13, 6))
        colors = plt.cm.Set1(np.linspace(0, 0.9, len(all_results)))
        for (strat, res), color in zip(all_results.items(), colors):
            bdf = res['bets_df']
            if len(bdf) == 0:
                continue
            ax.plot(
                bdf['date'], bdf['bankroll_after'],
                label=f"{strat} (ROI={res['metrics']['roi']:+.1f}%)",
                linewidth=1.5, color=color
            )
        ax.axhline(initial_bankroll, color='gray', linestyle='--',
                   alpha=0.5, label='Initial bankroll')
        ax.set_xlabel('Date')
        ax.set_ylabel('Bankroll')
        ax.set_title('Bankroll Evolution by Strategy — Test Set 2022-2024')
        ax.legend(loc='upper left', fontsize=8)
        fig.tight_layout()
        plot_path = os.path.join(output_dir, 'bankroll_evolution.png')
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Plot   → {plot_path}")

        # ── P&L by outcome bar chart ───────────────────────────────────────
        outcomes = ['HomeWin', 'Draw', 'AwayWin']
        fig, ax = plt.subplots(figsize=(10, 6))
        x     = np.arange(len(outcomes))
        width = 0.15
        for i, (strat, res) in enumerate(all_results.items()):
            pnl = res['metrics'].get('pnl_by_outcome')
            if pnl is None or len(pnl) == 0:
                continue
            vals = [pnl.loc[o, 'sum'] if o in pnl.index else 0 for o in outcomes]
            ax.bar(x + i * width, vals, width, label=strat, alpha=0.85)
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(outcomes)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_ylabel('Total P&L')
        ax.set_title('P&L by Outcome Type per Strategy')
        ax.legend(fontsize=7)
        fig.tight_layout()
        pnl_path = os.path.join(output_dir, 'pnl_by_outcome.png')
        fig.savefig(pnl_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        # ── Save best-strategy bets CSV ────────────────────────────────────
        best_bets = best_res['bets_df']
        if len(best_bets) > 0:
            csv_path = os.path.join(output_dir, f'bets_{best_strat}.csv')
            best_bets.to_csv(csv_path, index=False)
            print(f"Bets   → {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Football Betting Backtester — Phase 4'
    )
    parser.add_argument(
        '--predictions', required=True,
        help='Path to predictions CSV generated by the multiclass pipeline '
             '(e.g. results/.../predictions_CatBoost.csv)'
    )
    parser.add_argument(
        '--bankroll', type=float, default=1000,
        help='Initial bankroll in units (default: 1000)'
    )
    parser.add_argument(
        '--output', default='results/Backtest/',
        help='Output directory for report and plots (default: results/Backtest/)'
    )
    parser.add_argument(
        '--league', default=None,
        help='Filter predictions to a single league before backtesting. '
             'Valid values: Germany, Italy, France, Spain, Premier_League, Brazil, All '
             '(default: All — no filter applied)'
    )
    args = parser.parse_args()

    print(f"Loading predictions: {args.predictions}")
    df = pd.read_csv(args.predictions)
    print(f"  {len(df)} matches loaded")

    if args.league and args.league.lower() != 'all':
        if 'league' not in df.columns:
            print("  WARNING: 'league' column not found — cannot filter. Running on all matches.")
        else:
            before = len(df)
            df = df[df['league'] == args.league].reset_index(drop=True)
            print(f"  League filter '{args.league}': {before} → {len(df)} matches")
            if len(df) == 0:
                print(f"  ERROR: No matches found for league='{args.league}'. "
                      f"Available: {pd.read_csv(args.predictions)['league'].unique().tolist()}")
                sys.exit(1)
    else:
        print("  No league filter applied (all leagues)")

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        test_dates = df['date'].dt.year.value_counts().sort_index()
        print(f"  Match distribution by year:\n{test_dates.to_string()}")

    backtester = BettingBacktester(df)

    league_label = args.league if (args.league and args.league.lower() != 'all') else 'All'
    print(f"\nRunning {len(BettingBacktester.STRATEGIES)} strategies "
          f"(bankroll={args.bankroll}, league={league_label})...\n")
    all_results = backtester.run_all_strategies(bankroll=args.bankroll)

    print(f"\nSaving report to: {args.output}\n")
    backtester.save_report(
        all_results, args.output,
        initial_bankroll=args.bankroll,
        league_filter=league_label,
    )
    print("\nBacktest complete.")


if __name__ == '__main__':
    main()
