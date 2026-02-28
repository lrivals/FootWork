"""
src/Models/Hierarchical/Football_Match_Hierarchical_Pipeline.py

Two-stage hierarchical classifier to improve Draw recall.

Architecture:
    Stage 1: HomeWin vs Not-HomeWin  (binary CatBoost + calibration)
    Stage 2: Draw vs AwayWin         (binary CatBoost + calibration, trained on
                                      the subset where target != HomeWin)

Cascade prediction:
    P(HomeWin)  = Stage 1 P(HomeWin)
    P(Draw)     = (1 − Stage 1 P(HomeWin)) × Stage 2 P(Draw)
    P(AwayWin)  = (1 − Stage 1 P(HomeWin)) × Stage 2 P(AwayWin)
    Final label = argmax of the three adjusted probabilities

Usage:
    python src/Models/Hierarchical/Football_Match_Hierarchical_Pipeline.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    matthews_corrcoef, cohen_kappa_score,
    classification_report, confusion_matrix,
    roc_auc_score,
)
from catboost import CatBoostClassifier

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.Config.Config_Manager import ConfigManager
from src.Models.threshold_optimizer import find_optimal_threshold

# ---------------------------------------------------------------------------
# Temporal split constants (same as multiclass pipeline)
# ---------------------------------------------------------------------------
CAL_START_YEAR = 2020
CAL_END_YEAR   = 2022


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(config):
    """
    Load dataset and apply temporal split.
    Returns (train_df, cal_df, test_df, exclude_columns, test_metadata).
    """
    input_path = config.get_paths()['full_dataset']
    exclude_columns = config.get_config_value('excluded_columns', default=[])

    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'])

    train_df = df[df['date'].dt.year < CAL_START_YEAR].copy()
    cal_df   = df[(df['date'].dt.year >= CAL_START_YEAR) &
                  (df['date'].dt.year <  CAL_END_YEAR)].copy()
    test_df  = df[df['date'].dt.year >= CAL_END_YEAR].copy()

    print(f"Temporal split: train={len(train_df)}  cal={len(cal_df)}  test={len(test_df)}")

    meta_cols = [c for c in [
        'date', 'home_team', 'away_team', 'target_result',
        'target_home_goals', 'target_away_goals', 'league',
        'implied_prob_home', 'implied_prob_draw', 'implied_prob_away',
        'raw_odds_home', 'raw_odds_draw', 'raw_odds_away',
    ] if c in test_df.columns]
    test_metadata = test_df[meta_cols].reset_index(drop=True)

    return train_df, cal_df, test_df, exclude_columns, test_metadata


def prepare_features(df, exclude_columns, scaler=None, fit_scaler=False):
    """
    Drop excluded columns, scale features.
    Returns (X_scaled, scaler).
    """
    all_drop = [c for c in exclude_columns if c in df.columns and c != 'target_result']
    X = df.drop(all_drop + ['target_result'], axis=1, errors='ignore')
    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, scaler


# ---------------------------------------------------------------------------
# Stage training helpers
# ---------------------------------------------------------------------------

def _train_catboost_stage(X_train, y_train, stage_name, output_dir):
    """Train a CatBoost binary classifier with calibration."""
    model = CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        random_seed=42,
        verbose=False,
        train_dir=os.path.join(output_dir, f'catboost_{stage_name}'),
    )
    model.fit(X_train, y_train)
    return model


def _calibrate(model, X_cal, y_cal):
    """Fit isotonic calibration on the calibration set."""
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method='isotonic')
    calibrated.fit(X_cal, y_cal)
    return calibrated


# ---------------------------------------------------------------------------
# Hierarchical prediction
# ---------------------------------------------------------------------------

def hierarchical_predict(proba_s1, proba_s2, thresh1=0.5, thresh2=0.5):
    """
    Combine Stage 1 and Stage 2 probabilities into 3-class predictions.

    Args:
        proba_s1: array (N,) — P(HomeWin) from Stage 1
        proba_s2: array (N,) — P(Draw) from Stage 2 (conditioned on Not-HomeWin)
        thresh1:  Stage 1 decision threshold for HomeWin
        thresh2:  Stage 2 decision threshold for Draw (among not-HomeWin)

    Returns:
        predictions: array of str ('HomeWin', 'Draw', 'AwayWin')
        combined_proba: array (N, 3) — [P(HomeWin), P(Draw), P(AwayWin)]
    """
    p_home = proba_s1                        # P(HomeWin)
    p_not_home = 1.0 - proba_s1             # P(Not-HomeWin)
    p_draw    = p_not_home * proba_s2       # P(Draw) = P(Not-HW) × P(Draw | Not-HW)
    p_away    = p_not_home * (1 - proba_s2) # P(AwayWin)

    combined = np.column_stack([p_home, p_draw, p_away])  # [HomeWin, Draw, AwayWin]

    # Apply thresholds (adjusted probability score)
    scores = combined.copy()
    scores[:, 0] /= thresh1 if thresh1 > 0 else 0.5
    scores[:, 1] /= thresh2 if thresh2 > 0 else 0.5

    idx_map = {0: 'HomeWin', 1: 'Draw', 2: 'AwayWin'}
    predictions = np.array([idx_map[i] for i in np.argmax(scores, axis=1)])

    return predictions, combined


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_hierarchical_pipeline(config, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    train_df, cal_df, test_df, exclude_columns, test_metadata = load_data(config)

    X_train_raw, scaler = prepare_features(train_df, exclude_columns, fit_scaler=True)
    X_cal_raw, _        = prepare_features(cal_df,   exclude_columns, scaler=scaler)
    X_test_raw, _       = prepare_features(test_df,  exclude_columns, scaler=scaler)

    y_train_full = train_df['target_result'].values
    y_cal_full   = cal_df['target_result'].values
    y_test_full  = test_df['target_result'].values

    # ── Stage 1: HomeWin vs Not-HomeWin ────────────────────────────────────
    print("\n--- Stage 1: HomeWin vs Not-HomeWin ---")
    y_s1_train = (y_train_full == 'HomeWin').astype(int)
    y_s1_cal   = (y_cal_full   == 'HomeWin').astype(int)
    y_s1_test  = (y_test_full  == 'HomeWin').astype(int)

    model_s1 = _train_catboost_stage(X_train_raw, y_s1_train, 'stage1', output_dir)
    cal_s1   = _calibrate(model_s1, X_cal_raw, y_s1_cal)

    proba_s1_cal  = cal_s1.predict_proba(X_cal_raw)[:, 1]   # P(HomeWin) on cal set
    proba_s1_test = cal_s1.predict_proba(X_test_raw)[:, 1]  # P(HomeWin) on test set

    thresh1, f1_s1 = find_optimal_threshold(proba_s1_cal, y_s1_cal)
    print(f"  Stage 1 optimal threshold: {thresh1:.3f}  (F1={f1_s1:.3f})")
    print(f"  Stage 1 ROC-AUC (test): {roc_auc_score(y_s1_test, proba_s1_test):.4f}")

    # ── Stage 2: Draw vs AwayWin (trained only on Not-HomeWin matches) ──────
    print("\n--- Stage 2: Draw vs AwayWin (Not-HomeWin subset) ---")
    mask_train_s2 = (y_train_full != 'HomeWin')
    mask_cal_s2   = (y_cal_full   != 'HomeWin')
    mask_test_s2  = (y_test_full  != 'HomeWin')

    X_s2_train = X_train_raw[mask_train_s2]
    X_s2_cal   = X_cal_raw[mask_cal_s2]
    X_s2_test  = X_test_raw[mask_test_s2]

    y_s2_train = (y_train_full[mask_train_s2] == 'Draw').astype(int)
    y_s2_cal   = (y_cal_full[mask_cal_s2]     == 'Draw').astype(int)
    y_s2_test  = (y_test_full[mask_test_s2]   == 'Draw').astype(int)

    print(f"  Stage 2 training size: {len(X_s2_train)} "
          f"(Draw={y_s2_train.sum()}, Away={len(y_s2_train)-y_s2_train.sum()})")

    model_s2 = _train_catboost_stage(X_s2_train, y_s2_train, 'stage2', output_dir)
    cal_s2   = _calibrate(model_s2, X_s2_cal, y_s2_cal)

    proba_s2_cal  = cal_s2.predict_proba(X_s2_cal)[:, 1]   # P(Draw | Not-HW) on cal
    proba_s2_full = np.full(len(X_test_raw), 0.5)          # default for HomeWin rows
    proba_s2_full[mask_test_s2] = cal_s2.predict_proba(X_test_raw[mask_test_s2])[:, 1]

    thresh2, f1_s2 = find_optimal_threshold(proba_s2_cal, y_s2_cal)
    print(f"  Stage 2 optimal threshold: {thresh2:.3f}  (F1={f1_s2:.3f})")
    print(f"  Stage 2 ROC-AUC (test, Not-HW subset): "
          f"{roc_auc_score(y_s2_test, proba_s2_full[mask_test_s2]):.4f}")

    # ── Cascaded final predictions ─────────────────────────────────────────
    print("\n--- Final cascaded predictions ---")
    # pred_default: argmax of combined proba (no threshold adjustment, thresh=0.5)
    _, combined_proba = hierarchical_predict(
        proba_s1_test, proba_s2_full, thresh1=0.5, thresh2=0.5
    )
    idx_map = {0: 'HomeWin', 1: 'Draw', 2: 'AwayWin'}
    y_pred_default = np.array([idx_map[i] for i in np.argmax(combined_proba, axis=1)])

    # pred_opt: with optimal thresholds from calibration set
    y_pred, _ = hierarchical_predict(
        proba_s1_test, proba_s2_full, thresh1=thresh1, thresh2=thresh2
    )

    acc    = accuracy_score(y_test_full, y_pred)
    bal    = balanced_accuracy_score(y_test_full, y_pred)
    mcc    = matthews_corrcoef(y_test_full, y_pred)
    kappa  = cohen_kappa_score(y_test_full, y_pred)
    report = classification_report(
        y_test_full, y_pred,
        target_names=['AwayWin', 'Draw', 'HomeWin']
    )

    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Balanced Acc    : {bal:.4f}")
    print(f"  MCC             : {mcc:.4f}")
    print(f"  Cohen's Kappa   : {kappa:.4f}")
    print("\nClassification Report:")
    print(report)

    # ── Confusion matrix plot ──────────────────────────────────────────────
    cm = confusion_matrix(y_test_full, y_pred, labels=['HomeWin', 'Draw', 'AwayWin'])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['HomeWin', 'Draw', 'AwayWin'],
                yticklabels=['HomeWin', 'Draw', 'AwayWin'], ax=ax)
    ax.set_title('Confusion Matrix — Hierarchical Classifier (Stage1+Stage2)')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'confusion_matrix_hierarchical.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Save metrics + predictions CSV ────────────────────────────────────
    _save_results(
        output_dir, acc, bal, mcc, kappa, report,
        test_metadata, combined_proba, y_pred_default, y_pred, y_test_full
    )

    return {
        'accuracy': acc,
        'balanced_accuracy': bal,
        'mcc': mcc,
        'kappa': kappa,
        'report': report,
        'combined_proba': combined_proba,
        'predictions': y_pred,
    }


def _save_results(output_dir, acc, bal, mcc, kappa, report,
                  test_metadata, combined_proba, y_pred_default, y_pred_opt, y_true):
    """Save metrics text file and predictions CSV (with EV and Kelly fractions)."""
    # Metrics
    metrics_path = os.path.join(output_dir, 'metrics_hierarchical.txt')
    with open(metrics_path, 'w') as f:
        f.write("Hierarchical Cascade Classifier — Results\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Accuracy         : {acc:.4f}\n")
        f.write(f"Balanced Accuracy: {bal:.4f}\n")
        f.write(f"MCC              : {mcc:.4f}\n")
        f.write(f"Cohen's Kappa    : {kappa:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"\nMetrics → {metrics_path}")

    # Predictions CSV — schema compatible with Betting_Backtest.py
    if test_metadata is not None and len(test_metadata) > 0:
        df = test_metadata.copy().reset_index(drop=True)

        # Calibrated probabilities (Stage1+Stage2 cascade)
        df['prob_homewin'] = combined_proba[:, 0]
        df['prob_draw']    = combined_proba[:, 1]
        df['prob_awaywin'] = combined_proba[:, 2]

        df['pred_default'] = y_pred_default   # argmax of combined proba
        df['pred_opt']     = y_pred_opt       # threshold-adjusted prediction

        # EV: ev = (model_prob × decimal_odds) − 1
        has_raw_odds = all(c in df.columns
                           for c in ['raw_odds_home', 'raw_odds_draw', 'raw_odds_away'])
        if has_raw_odds:
            df['ev_home'] = df['prob_homewin'] * df['raw_odds_home'] - 1
            df['ev_draw'] = df['prob_draw']    * df['raw_odds_draw'] - 1
            df['ev_away'] = df['prob_awaywin'] * df['raw_odds_away'] - 1

            # Quarter Kelly (positive EV only, capped at 10% of bankroll)
            for ev_col, kelly_col, odds_col in [
                ('ev_home', 'kelly_home', 'raw_odds_home'),
                ('ev_draw', 'kelly_draw', 'raw_odds_draw'),
                ('ev_away', 'kelly_away', 'raw_odds_away'),
            ]:
                divisor = np.where(df[odds_col] > 1, df[odds_col] - 1, np.nan)
                kelly = np.where(
                    pd.notna(divisor) & (df[ev_col] > 0),
                    (df[ev_col] / divisor) * 0.25,
                    0.0,
                )
                df[kelly_col] = np.clip(kelly, 0.0, 0.10)

        csv_path = os.path.join(output_dir, 'predictions_Hierarchical.csv')
        df.to_csv(csv_path, index=False)
        print(f"Predictions → {csv_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    config = ConfigManager('src/Config/configMC_1.yaml')

    timestamp  = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = Path('results/All_Leagues/Hierarchical') / f'hierarchical_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  FootWork — Hierarchical Cascade Classifier")
    print("=" * 60)

    run_hierarchical_pipeline(config, str(output_dir))
    print(f"\nAll outputs saved in: {output_dir}")


if __name__ == '__main__':
    main()
