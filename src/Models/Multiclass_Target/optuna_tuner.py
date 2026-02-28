"""
src/Models/Multiclass_Target/optuna_tuner.py

Optuna hyperparameter search for CatBoost (multiclass football prediction).

Objective: weighted macro-F1 on the calibration set, with extra emphasis on
           Draw and AwayWin (minority classes).
Search space:
    - CatBoost hyperparams: depth, learning_rate, l2_leaf_reg,
      bagging_temperature, min_data_in_leaf
    - class_weight_draw, class_weight_awaywin  (HomeWin fixed at 1.0)
    - calibration_method: isotonic | sigmoid

Usage (standalone):
    python src/Models/Multiclass_Target/optuna_tuner.py

Typical usage via pipeline (when optuna.enabled: true in configMC_1.yaml):
    run_optuna_study(X_train, y_train, X_cal, y_cal, n_trials=50, timeout=600)
    → returns dict of best CatBoost kwargs + calibration_method
"""

import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

from sklearn.metrics import f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from catboost import CatBoostClassifier

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def _build_objective(X_train, y_train, X_cal, y_cal, output_dir):
    """
    Return an Optuna objective function that trains CatBoost on X_train,
    calibrates on X_cal, and returns a weighted F1 score that emphasises
    Draw and AwayWin.

    class label order (LabelEncoder sorts alphabetically):
        0 = AwayWin, 1 = Draw, 2 = HomeWin
    """
    def objective(trial):
        # ── CatBoost hyperparameters ───────────────────────────────────────
        depth               = trial.suggest_int('depth', 4, 10)
        learning_rate       = trial.suggest_float('learning_rate', 0.01, 0.2, log=True)
        l2_leaf_reg         = trial.suggest_float('l2_leaf_reg', 1.0, 15.0, log=True)
        bagging_temperature = trial.suggest_float('bagging_temperature', 0.0, 2.0)
        min_data_in_leaf    = trial.suggest_int('min_data_in_leaf', 10, 100)

        # ── Class weights (AwayWin=0, Draw=1, HomeWin=2) ──────────────────
        cw_awaywin = trial.suggest_float('class_weight_awaywin', 1.0, 2.5)
        cw_draw    = trial.suggest_float('class_weight_draw',    1.0, 3.5)
        class_weights = [cw_awaywin, cw_draw, 1.0]

        # ── Calibration method ─────────────────────────────────────────────
        cal_method = trial.suggest_categorical('calibration_method',
                                               ['isotonic', 'sigmoid'])

        model = CatBoostClassifier(
            iterations=100,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            bagging_temperature=bagging_temperature,
            min_data_in_leaf=min_data_in_leaf,
            class_weights=class_weights,
            random_seed=42,
            verbose=False,
            train_dir=os.path.join(output_dir, f'catboost_optuna_trial_{trial.number}'),
        )
        model.fit(X_train, y_train)

        calibrated = CalibratedClassifierCV(
            FrozenEstimator(model), method=cal_method
        )
        calibrated.fit(X_cal, y_cal)

        y_pred = calibrated.predict(X_cal)

        # Per-class F1
        f1_per_class = f1_score(y_cal, y_pred, average=None, labels=[0, 1, 2])
        f1_away = f1_per_class[0] if len(f1_per_class) > 0 else 0.0
        f1_draw = f1_per_class[1] if len(f1_per_class) > 1 else 0.0
        f1_home = f1_per_class[2] if len(f1_per_class) > 2 else 0.0

        # Weighted objective: emphasise Draw + AwayWin
        score = 0.4 * f1_home + 0.3 * f1_draw + 0.3 * f1_away
        return score

    return objective


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_optuna_study(X_train, y_train, X_cal, y_cal,
                     n_trials=50, timeout=600, output_dir='/tmp/optuna_catboost'):
    """
    Run an Optuna study and return the best CatBoost params + calibration method.

    Args:
        X_train, y_train : training data (scaled numpy arrays, labels encoded)
        X_cal, y_cal     : calibration data
        n_trials         : number of Optuna trials
        timeout          : max seconds (0 = no limit)
        output_dir       : where to write catboost train logs per trial

    Returns:
        dict with keys matching CatBoostClassifier kwargs + 'calibration_method':
        {
            'depth': int,
            'learning_rate': float,
            'l2_leaf_reg': float,
            'bagging_temperature': float,
            'min_data_in_leaf': int,
            'class_weights': [float, float, float],  # [AwayWin, Draw, HomeWin]
            'calibration_method': str,
        }
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError(
            "optuna is required for hyperparameter search. "
            "Install it with: pip install optuna"
        )

    os.makedirs(output_dir, exist_ok=True)

    study = optuna.create_study(direction='maximize',
                                study_name='catboost_football')
    objective = _build_objective(X_train, y_train, X_cal, y_cal, output_dir)

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout if timeout > 0 else None,
        show_progress_bar=True,
    )

    best = study.best_params
    print(f"\n[Optuna] Best trial: #{study.best_trial.number}  "
          f"score={study.best_value:.4f}")
    print(f"[Optuna] Best params: {best}")

    # Reshape into CatBoostClassifier-compatible kwargs
    best_catboost_kwargs = {
        'depth':               best['depth'],
        'learning_rate':       best['learning_rate'],
        'l2_leaf_reg':         best['l2_leaf_reg'],
        'bagging_temperature': best['bagging_temperature'],
        'min_data_in_leaf':    best['min_data_in_leaf'],
        'class_weights':       [best['class_weight_awaywin'],
                                 best['class_weight_draw'],
                                 1.0],
        'calibration_method':  best['calibration_method'],
    }
    return best_catboost_kwargs


def save_best_params(best_params, output_dir):
    """Persist best Optuna params to YAML for reproducibility."""
    import yaml
    path = os.path.join(output_dir, 'best_params_optuna.yaml')
    with open(path, 'w') as f:
        yaml.dump(best_params, f, default_flow_style=False)
    print(f"[Optuna] Best params saved → {path}")
