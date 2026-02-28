import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score,
                             balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score,
                             roc_curve, auc)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.frozen import FrozenEstimator
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # non-interactive, thread-safe backend
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.Config.Config_Manager import ConfigManager
from src.Models.threshold_optimizer import (
    find_optimal_thresholds_multiclass, predict_with_thresholds
)
from src.Models.Multiclass_Target.optuna_tuner import run_optuna_study, save_best_params

# ---------------------------------------------------------------------------
# Calibration split year constants
# ---------------------------------------------------------------------------
CAL_START_YEAR = 2020   # calibration set: 2020 <= year < 2022
CAL_END_YEAR   = 2022   # test set: year >= 2022


def get_models(config):
    model_params = config.get_config_value('model_parameters', default={})

    return {
        'Random Forest': RandomForestClassifier(**model_params.get('random_forest', {})),
        'Logistic Regression': LogisticRegression(**model_params.get('logistic_regression', {})),
        'SVM': SVC(**model_params.get('svm', {})),
        'Gradient Boosting': GradientBoostingClassifier(**model_params.get('gradient_boosting', {})),
        'XGBoost': XGBClassifier(**model_params.get('xgboost', {})),
        'LightGBM': LGBMClassifier(**model_params.get('lightgbm', {})),
        'CatBoost': CatBoostClassifier(**model_params.get('catboost', {}),
                              train_dir='src/Models/Multiclass_Target/catboost_info'),
        'Neural Network': MLPClassifier(**model_params.get('neural_network', {})),
        'KNN': KNeighborsClassifier(**model_params.get('knn', {})),
        'AdaBoost': AdaBoostClassifier(**model_params.get('adaboost', {})),
        'Extra Trees': ExtraTreesClassifier(**model_params.get('extra_trees', {}))
    }


def load_and_prepare_data(config):
    """
    Load data and create three temporal splits:
        train : year < CAL_START_YEAR  (< 2020)
        cal   : CAL_START_YEAR <= year < CAL_END_YEAR  (2020-2021)
        test  : year >= CAL_END_YEAR  (>= 2022)

    The calibration set is used for CalibratedClassifierCV and threshold optimisation.

    Returns:
        X_train_scaled, X_test_scaled, X_cal_scaled,
        y_train, y_test, y_cal,
        class_names, le, test_metadata
    """
    input_path = config.get_paths()['full_dataset']
    exclude_columns = config.get_config_value('excluded_columns', default=[])
    split_config = config.get_config_value('data_split', default={'test_size': 0.2, 'random_state': 42})

    df = pd.read_csv(input_path)

    print("\nTarget distribution:")
    print(df['target_result'].value_counts())

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        train_df = df[df['date'].dt.year < CAL_START_YEAR].copy()
        cal_df   = df[(df['date'].dt.year >= CAL_START_YEAR) &
                      (df['date'].dt.year <  CAL_END_YEAR)].copy()
        test_df  = df[df['date'].dt.year >= CAL_END_YEAR].copy()
        print(f"\nTemporal split: "
              f"train < {CAL_START_YEAR} ({len(train_df)} rows), "
              f"cal {CAL_START_YEAR}–{CAL_END_YEAR-1} ({len(cal_df)} rows), "
              f"test >= {CAL_END_YEAR} ({len(test_df)} rows)")
        all_cols_to_drop = [c for c in exclude_columns if c in df.columns and c != 'target_result']
        X_train = train_df.drop(all_cols_to_drop + ['target_result'], axis=1, errors='ignore')
        X_cal   = cal_df.drop(all_cols_to_drop + ['target_result'],   axis=1, errors='ignore')
        X_test  = test_df.drop(all_cols_to_drop + ['target_result'],  axis=1, errors='ignore')
        y_train_raw = train_df['target_result']
        y_cal_raw   = cal_df['target_result']
        y_test_raw  = test_df['target_result']
        # --- Preserve test set metadata for prediction CSV export ---
        meta_cols = [c for c in [
            'date', 'home_team', 'away_team', 'target_result',
            'target_home_goals', 'target_away_goals', 'league',
            'implied_prob_home', 'implied_prob_draw', 'implied_prob_away',
            'raw_odds_home', 'raw_odds_draw', 'raw_odds_away',
        ] if c in test_df.columns]
        test_metadata = test_df[meta_cols].reset_index(drop=True)
    else:
        # Fallback: random split (no calibration set)
        random_params = {k: v for k, v in split_config.items() if k != 'temporal_split_year'}
        all_cols_to_drop = [c for c in exclude_columns if c in df.columns and c != 'target_result']
        X = df.drop(all_cols_to_drop + ['target_result'], axis=1, errors='ignore')
        y_raw = df['target_result']
        X_train, X_test, y_train_raw, y_test_raw = train_test_split(X, y_raw, **random_params)
        X_cal, y_cal_raw = X_test, y_test_raw  # reuse test as cal in fallback
        test_metadata = pd.DataFrame()

    # Encode target labels
    le = LabelEncoder()
    le.fit(df['target_result'])  # fit on full distribution for consistent class mapping
    y_train = le.transform(y_train_raw)
    y_cal   = le.transform(y_cal_raw)
    y_test  = le.transform(y_test_raw)
    class_names = le.classes_

    feature_names = list(X_train.columns)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_cal_scaled   = scaler.transform(X_cal)
    X_test_scaled  = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, X_cal_scaled, y_train, y_test, y_cal, class_names, le, test_metadata, feature_names


# ---------------------------------------------------------------------------
# Loss / iteration tracking helpers
# ---------------------------------------------------------------------------

def _fit_with_tracking(model, X_train, X_test, y_train, y_test, sample_weight=None):
    """
    Fit model and capture per-iteration loss/accuracy where supported.

    Args:
        sample_weight: Optional per-sample weights for class rebalancing.
                       Used by XGBoost, GradientBoosting, AdaBoost (which don't
                       accept class_weight in their constructor).

    Returns:
        (fitted_model, loss_history | None)
        loss_history: {'train': list, 'val': list, 'metric': str}
    """
    model_class = model.__class__.__name__

    if model_class == 'XGBClassifier':
        model.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False,
        )
        evals = model.evals_result()
        metric_key = list(evals['validation_0'].keys())[0]
        return model, {
            'train':  evals['validation_0'][metric_key],
            'val':    evals['validation_1'][metric_key],
            'metric': metric_key,
        }

    elif model_class == 'LGBMClassifier':
        evals_result = {}
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_names=['train', 'val'],
            callbacks=[lgb.record_evaluation(evals_result)],
        )
        metric_key = list(evals_result['train'].keys())[0]
        return model, {
            'train':  evals_result['train'][metric_key],
            'val':    evals_result['val'][metric_key],
            'metric': metric_key,
        }

    elif model_class == 'CatBoostClassifier':
        base_dir = model.get_param('train_dir') or 'catboost_info'
        safe_dir = os.path.join(base_dir, f'run_{id(model)}')
        os.makedirs(safe_dir, exist_ok=True)
        model.set_params(train_dir=safe_dir)
        model.fit(X_train, y_train, eval_set=(X_test, y_test))
        evals = model.get_evals_result()
        learn_key = 'learn' if 'learn' in evals else list(evals.keys())[0]
        val_key   = 'validation' if 'validation' in evals else list(evals.keys())[-1]
        metric_key = list(evals[learn_key].keys())[0]
        return model, {
            'train':  evals[learn_key][metric_key],
            'val':    evals[val_key][metric_key],
            'metric': metric_key,
        }

    elif model_class == 'GradientBoostingClassifier':
        model.fit(X_train, y_train, sample_weight=sample_weight)
        train_acc = [accuracy_score(y_train, p) for p in model.staged_predict(X_train)]
        val_acc   = [accuracy_score(y_test,  p) for p in model.staged_predict(X_test)]
        return model, {
            'train':  train_acc,
            'val':    val_acc,
            'metric': 'accuracy',
        }

    elif model_class == 'AdaBoostClassifier':
        model.fit(X_train, y_train, sample_weight=sample_weight)
        return model, None

    else:
        model.fit(X_train, y_train)
        return model, None


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_loss_curve(loss_history, model_name, output_dir):
    """Save train vs. validation loss/accuracy curve."""
    train_vals = loss_history['train']
    val_vals   = loss_history['val']
    metric     = loss_history['metric']

    _LOSS_METRICS = {'loss', 'logloss', 'multiclass', 'merror', 'mae', 'mse', 'rmse'}
    is_loss = any(tok in metric.lower() for tok in _LOSS_METRICS)
    best_iter = int(np.argmin(val_vals) if is_loss else np.argmax(val_vals))

    fig, ax = plt.subplots(figsize=(8, 5))
    iters = range(1, len(train_vals) + 1)
    ax.plot(iters, train_vals, label='Train',      linewidth=1.5)
    ax.plot(iters, val_vals,   label='Validation', linewidth=1.5)
    ax.axvline(best_iter + 1, color='red', linestyle='--', alpha=0.7,
               label=f'Best iter = {best_iter + 1}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel(metric)
    ax.set_title(f'Training Curve — {model_name}')
    ax.legend()
    fig.tight_layout()
    safe = model_name.replace(' ', '_')
    fig.savefig(os.path.join(output_dir, f'loss_curve_{safe}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_roc_multiclass(probs, y_test, class_names, model_name, output_dir):
    """OvR ROC curves for each class + macro average. Returns (per_class_auc, macro_auc)."""
    n_classes = len(class_names)
    y_bin = label_binarize(y_test, classes=list(range(n_classes)))

    fpr_dict, tpr_dict, auc_dict = {}, {}, {}
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        fpr_dict[i] = fpr
        tpr_dict[i] = tpr
        auc_dict[i] = auc(fpr, tpr)

    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.Set1(np.linspace(0, 0.8, n_classes))
    for i, (cls, color) in enumerate(zip(class_names, colors)):
        ax.plot(fpr_dict[i], tpr_dict[i], color=color, lw=2,
                label=f'{cls} (AUC = {auc_dict[i]:.3f})')
    ax.plot(all_fpr, mean_tpr, 'k--', lw=2,
            label=f'Macro avg (AUC = {macro_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'gray', linestyle=':', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves (OvR) — {model_name}')
    ax.legend(loc='lower right')
    fig.tight_layout()
    safe = model_name.replace(' ', '_')
    fig.savefig(os.path.join(output_dir, f'roc_{safe}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    return auc_dict, macro_auc


def _plot_calibration_curve(model, calibrated_model, X_cal, y_cal, class_names,
                            model_name, output_dir):
    """
    Reliability diagrams (one per class, OvR) comparing raw vs. calibrated probabilities.
    """
    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 5))
    if n_classes == 1:
        axes = [axes]

    for i, (cls, ax) in enumerate(zip(class_names, axes)):
        y_bin = (y_cal == i).astype(int)
        if hasattr(model, 'predict_proba'):
            try:
                raw_proba = model.predict_proba(X_cal)[:, i]
                CalibrationDisplay.from_predictions(
                    y_bin, raw_proba, n_bins=10, ax=ax,
                    name='Uncalibrated', color='steelblue'
                )
            except Exception:
                pass
        if calibrated_model is not None:
            try:
                cal_proba = calibrated_model.predict_proba(X_cal)[:, i]
                CalibrationDisplay.from_predictions(
                    y_bin, cal_proba, n_bins=10, ax=ax,
                    name='Calibrated (isotonic)', color='coral'
                )
            except Exception:
                pass
        ax.set_title(f'{model_name} — {cls}')
        ax.legend()

    fig.suptitle(f'Calibration Curves — {model_name}', fontsize=13)
    fig.tight_layout()
    safe = model_name.replace(' ', '_')
    fig.savefig(os.path.join(output_dir, f'calibration_{safe}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_summary_charts(all_results, output_dir):
    """Bar charts comparing accuracy, balanced accuracy, and macro AUC across models."""
    model_names = list(all_results.keys())
    accuracies  = [r['accuracy']          for r in all_results.values()]
    bal_accs    = [r['balanced_accuracy'] for r in all_results.values()]
    macro_aucs  = [r['macro_roc_auc'] if r['macro_roc_auc'] is not None else 0.0
                   for r in all_results.values()]

    x     = np.arange(len(model_names))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.bar(x - width / 2, accuracies, width, label='Accuracy',     color='steelblue')
    ax.bar(x + width / 2, bal_accs,   width, label='Balanced Acc', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha='right')
    ax.set_ylim([0, 1])
    ax.set_ylabel('Score')
    ax.set_title('Accuracy vs Balanced Accuracy')
    ax.legend()

    ax = axes[1]
    ax.bar(x, macro_aucs, color='mediumseagreen')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha='right')
    ax.set_ylim([0, 1])
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random baseline')
    ax.set_ylabel('Macro ROC-AUC (OvR)')
    ax.set_title('Macro AUC by Model')
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'summary_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Core train / evaluate
# ---------------------------------------------------------------------------

def train_and_evaluate_model(model, X_train, X_test, X_cal, y_train, y_test, y_cal,
                             class_names, dataset_name, output_dir):
    # --- Axe 1: sample_weight for models that don't accept class_weight in constructor ---
    sample_weight = compute_sample_weight('balanced', y_train)

    model, loss_history = _fit_with_tracking(
        model, X_train, X_test, y_train, y_test, sample_weight=sample_weight
    )

    y_pred = model.predict(X_test)

    # Probabilities → ROC
    per_class_auc = {}
    macro_roc_auc = None
    probs = None
    if hasattr(model, 'predict_proba'):
        try:
            probs = model.predict_proba(X_test)
            per_class_auc, macro_roc_auc = _plot_roc_multiclass(
                probs, y_test, class_names, model.__class__.__name__, output_dir
            )
        except Exception as e:
            print(f"    [ROC skipped for {model.__class__.__name__}]: {e}")

    results = {
        'model':              model,
        'accuracy':           accuracy_score(y_test, y_pred),
        'balanced_accuracy':  balanced_accuracy_score(y_test, y_pred),
        'mcc':                matthews_corrcoef(y_test, y_pred),
        'kappa':              cohen_kappa_score(y_test, y_pred),
        'macro_roc_auc':      macro_roc_auc,
        'per_class_auc':      per_class_auc,
        'loss_history':       loss_history,
        'confusion_matrix':   confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, target_names=class_names),
    }

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f'Confusion Matrix — {model.__class__.__name__} ({dataset_name})')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir,
        f'confusion_matrix_{dataset_name}_{model.__class__.__name__}.png'), dpi=150)
    plt.close(fig)

    if loss_history is not None:
        _plot_loss_curve(loss_history, model.__class__.__name__, output_dir)

    # Calibration and threshold optimisation are run serially in train_all_models()
    # after the ThreadPoolExecutor block, so that exceptions are visible and not
    # silently swallowed by tqdm in worker threads.
    results['calibrated_model']      = None
    results['calibrated_macro_auc']  = None
    results['optimal_thresholds']    = None
    results['accuracy_opt']          = None
    results['balanced_accuracy_opt'] = None
    results['report_opt']            = None

    return results


# ---------------------------------------------------------------------------
# SHAP analysis (CatBoost native — Phase 5)
# ---------------------------------------------------------------------------

def compute_shap_analysis(catboost_model, X_test, class_names, feature_names,
                          output_dir, n_samples=500):
    """
    Compute per-class SHAP values using CatBoost's native ShapValues.

    Generates:
      - shap_summary_<Class>.png  : horizontal bar chart of top-20 mean |SHAP|
      - shap_top_features_by_class.csv : mean |SHAP| per feature per class
    """
    try:
        from catboost import Pool
    except ImportError:
        print("  [SHAP] catboost not available — skipping SHAP analysis")
        return

    n = min(n_samples, len(X_test))
    X_sub = X_test[:n]

    pool = Pool(X_sub)
    try:
        # Returns (n_samples, n_classes, n_features + 1); last column is bias
        raw_shap = catboost_model.get_feature_importance(type='ShapValues', data=pool)
    except Exception as e:
        print(f"  [SHAP] get_feature_importance failed: {e}")
        return

    if raw_shap.ndim != 3:
        print(f"  [SHAP] Unexpected ShapValues shape {raw_shap.shape} — expected (N, C, F+1)")
        return

    n_classes = raw_shap.shape[1]
    if n_classes != len(class_names):
        print(f"  [SHAP] n_classes mismatch ({n_classes} vs {len(class_names)}) — skipping")
        return

    shap_per_class = raw_shap[:, :, :-1]  # drop bias column → (N, C, F)

    top_n = 20
    summary_data = {}
    for i, cls in enumerate(class_names):
        sv = shap_per_class[:, i, :]               # (N, F)
        mean_abs = np.abs(sv).mean(axis=0)          # (F,)
        summary_data[cls] = mean_abs

        # Sort by importance
        order = np.argsort(mean_abs)[::-1][:top_n]
        feats = [feature_names[j] for j in order]
        vals  = mean_abs[order]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(feats)), vals[::-1], color='steelblue', alpha=0.85)
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(feats[::-1], fontsize=9)
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title(f'SHAP Feature Importance — {cls} (top {top_n})')
        fig.tight_layout()
        safe_cls = cls.replace(' ', '_')
        plot_path = os.path.join(output_dir, f'shap_summary_{safe_cls}.png')
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  [SHAP] {cls} → {plot_path}")

    # CSV: feature × class → mean |SHAP|
    shap_df = pd.DataFrame(summary_data, index=feature_names)
    shap_df.index.name = 'feature'
    shap_df = shap_df.assign(max_importance=shap_df.max(axis=1)).sort_values(
        'max_importance', ascending=False
    ).drop(columns='max_importance')
    csv_path = os.path.join(output_dir, 'shap_top_features_by_class.csv')
    shap_df.to_csv(csv_path)
    print(f"  [SHAP] Summary CSV → {csv_path}")


def train_all_models(config):
    X_train, X_test, X_cal, y_train, y_test, y_cal, class_names, le, test_metadata, feature_names = load_and_prepare_data(config)
    output_dir = config.get_paths()['output_dir']
    models = get_models(config)

    # ── Phase 5: Optuna tuning for CatBoost (optional) ─────────────────────
    optuna_cfg = config.get_config_value('optuna', default={})
    if optuna_cfg.get('enabled', False):
        print("\n[Optuna] Tuning CatBoost hyperparameters + calibration method...")
        n_trials = int(optuna_cfg.get('n_trials', 50))
        timeout  = int(optuna_cfg.get('timeout_seconds', 600))
        optuna_dir = os.path.join(output_dir, 'optuna_trials')
        best_params = run_optuna_study(
            X_train, y_train, X_cal, y_cal,
            n_trials=n_trials, timeout=timeout, output_dir=optuna_dir,
        )
        save_best_params(best_params, output_dir)

        # Extract calibration method (not a CatBoost kwarg)
        cal_method_optuna = best_params.pop('calibration_method', 'isotonic')

        # Replace CatBoost in models dict with Optuna-tuned version
        cb_base_params = config.get_config_value('model_parameters', {}).get('catboost', {})
        merged = {**cb_base_params, **best_params}  # Optuna params override defaults
        merged.pop('calibration_method', None)       # safety — not a CatBoost kwarg
        models['CatBoost'] = CatBoostClassifier(
            **merged,
            random_seed=42,
            verbose=False,
            train_dir='src/Models/Multiclass_Target/catboost_info',
        )
        # Store chosen calibration method for later use
        config._optuna_cal_method = cal_method_optuna
        print(f"[Optuna] CatBoost replaced with tuned params. cal_method={cal_method_optuna}")

    max_workers = min(len(models), 5)
    print(f"\nLaunching {max_workers} parallel training workers for {len(models)} models...")

    def _train_one(name, model):
        return name, train_and_evaluate_model(
            model, X_train, X_test, X_cal, y_train, y_test, y_cal,
            class_names, "Full_Dataset", output_dir
        )

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_train_one, name, model): name
            for name, model in models.items()
        }
        with tqdm(as_completed(futures), total=len(futures),
                  desc="Training", unit="model") as pbar:
            for future in pbar:
                name = futures[future]
                try:
                    name, result = future.result()
                    results[name] = result
                    pbar.set_postfix_str(f"{name}  acc={result['accuracy']:.4f}")
                except Exception as e:
                    pbar.set_postfix_str(f"{name}  ERROR")
                    print(f"  [error] {name}: {e}")

    # ── Serial calibration + threshold optimisation ────────────────────────
    # Run after ThreadPoolExecutor so exceptions are fully visible (not swallowed
    # by tqdm in worker threads).
    print("\nCalibration + threshold optimisation (serial)...")
    for name, result in results.items():
        model = result['model']
        if not hasattr(model, 'predict_proba'):
            continue
        try:
            # Use Optuna-chosen calibration method for CatBoost if available
            cal_method = 'isotonic'
            if name == 'CatBoost' and hasattr(config, '_optuna_cal_method'):
                cal_method = config._optuna_cal_method
            calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=cal_method)
            calibrated.fit(X_cal, y_cal)
            cal_probs_test = calibrated.predict_proba(X_test)
            _, cal_auc = _plot_roc_multiclass(
                cal_probs_test, y_test, class_names,
                f'{name}_calibrated', output_dir
            )
            _plot_calibration_curve(
                model, calibrated, X_cal, y_cal, class_names, name, output_dir
            )
            result['calibrated_model']     = calibrated
            result['calibrated_macro_auc'] = cal_auc

            optimal_thresholds = find_optimal_thresholds_multiclass(calibrated, X_cal, y_cal)
            y_pred_opt = predict_with_thresholds(cal_probs_test, optimal_thresholds)
            y_pred_default_cal = np.argmax(cal_probs_test, axis=1)
            result['optimal_thresholds']    = optimal_thresholds
            result['accuracy_opt']          = accuracy_score(y_test, y_pred_opt)
            result['balanced_accuracy_opt'] = balanced_accuracy_score(y_test, y_pred_opt)
            result['report_opt']            = classification_report(
                y_test, y_pred_opt, target_names=class_names
            )
            print(f"  {name}: cal_auc={cal_auc:.4f}  acc_opt={result['accuracy_opt']:.4f}")

            # --- Phase 4 D1/D2: save per-match predictions with EV ---
            save_prediction_csv(
                name, test_metadata, cal_probs_test,
                y_pred_default_cal, y_pred_opt,
                class_names, le, output_dir
            )

            # --- Phase 5: SHAP analysis for CatBoost only ---
            shap_cfg = config.get_config_value('shap', default={})
            if name == 'CatBoost' and shap_cfg.get('enabled', False):
                print(f"\n  [SHAP] Running SHAP analysis for {name}...")
                compute_shap_analysis(
                    model, X_test, class_names, feature_names, output_dir,
                    n_samples=int(shap_cfg.get('n_samples', 500)),
                )
        except Exception:
            import traceback
            print(f"\n[Calibration FAILED for {name}]:")
            traceback.print_exc()

    _plot_summary_charts(results, output_dir)
    return results, class_names


# ---------------------------------------------------------------------------
# Per-match prediction CSV export (Phase 4 — D1/D2)
# ---------------------------------------------------------------------------

def save_prediction_csv(model_name, test_metadata, cal_probs_test,
                        y_pred_default, y_pred_opt, class_names, le, output_dir):
    """
    Save per-match calibrated predictions with EV and Kelly fractions to CSV.

    Columns: date, teams, actual result, prob_*, pred_default, pred_opt,
             ev_home/draw/away (requires raw_odds_* in test_metadata),
             kelly_home/draw/away (quarter Kelly, capped at 10%).
    """
    if test_metadata is None or len(test_metadata) == 0:
        return

    df = test_metadata.copy().reset_index(drop=True)

    # Calibrated probabilities per class (LabelEncoder sorts alphabetically)
    for i, cls in enumerate(class_names):
        df[f'prob_{cls.lower()}'] = cal_probs_test[:, i]

    df['pred_default'] = le.inverse_transform(y_pred_default)
    df['pred_opt']     = le.inverse_transform(y_pred_opt)

    # EV: ev = (model_prob × decimal_odds) − 1
    has_raw_odds = all(c in df.columns for c in ['raw_odds_home', 'raw_odds_draw', 'raw_odds_away'])
    if has_raw_odds:
        class_list = list(class_names)
        idx_home = class_list.index('HomeWin') if 'HomeWin' in class_list else None
        idx_draw = class_list.index('Draw')    if 'Draw'    in class_list else None
        idx_away = class_list.index('AwayWin') if 'AwayWin' in class_list else None

        for idx, ev_col, odds_col in [
            (idx_home, 'ev_home', 'raw_odds_home'),
            (idx_draw, 'ev_draw', 'raw_odds_draw'),
            (idx_away, 'ev_away', 'raw_odds_away'),
        ]:
            if idx is not None:
                df[ev_col] = cal_probs_test[:, idx] * df[odds_col] - 1

        # Quarter Kelly (positive EV only, capped at 10% of bankroll)
        for ev_col, kelly_col, odds_col in [
            ('ev_home', 'kelly_home', 'raw_odds_home'),
            ('ev_draw', 'kelly_draw', 'raw_odds_draw'),
            ('ev_away', 'kelly_away', 'raw_odds_away'),
        ]:
            if ev_col in df.columns:
                divisor = np.where(df[odds_col] > 1, df[odds_col] - 1, np.nan)
                kelly = np.where(
                    pd.notna(divisor) & (df[ev_col] > 0),
                    (df[ev_col] / divisor) * 0.25,
                    0.0
                )
                df[kelly_col] = np.clip(kelly, 0.0, 0.10)

    safe_name = model_name.replace(' ', '_')
    output_path = os.path.join(output_dir, f'predictions_{safe_name}.csv')
    df.to_csv(output_path, index=False)
    print(f"  Predictions CSV → {output_path}")


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(results, class_names, config):
    output_dir = config.get_paths()['output_dir']
    filename = os.path.join(output_dir, "metrics_results_Full_Dataset_Multiclass.txt")

    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

    with open(filename, 'w') as f:
        f.write("Results for Full Dataset - Multiclass Prediction\n")
        f.write("=" * 70 + "\n\n")

        # ── Ranking table ──────────────────────────────────────────────────
        f.write("=== RANKING TABLE ===\n")
        header = (f"{'Model':<22} {'Accuracy':>9} {'Bal.Acc':>9} "
                  f"{'MCC':>8} {'Kappa':>8} {'MacroAUC':>10}\n")
        f.write(header)
        f.write("-" * 72 + "\n")
        for model_name, result in sorted_results:
            auc_str = (f"{result['macro_roc_auc']:.4f}"
                       if result['macro_roc_auc'] is not None else "  N/A  ")
            f.write(f"{model_name:<22} {result['accuracy']:>9.4f} "
                    f"{result['balanced_accuracy']:>9.4f} "
                    f"{result['mcc']:>8.4f} {result['kappa']:>8.4f} {auc_str:>10}\n")
        f.write("\n")

        # ── Per-class AUC table ────────────────────────────────────────────
        f.write("=== PER-CLASS AUC (OvR) ===\n")
        f.write(f"{'Model':<22}" + "".join(f"{c:>12}" for c in class_names) + "\n")
        f.write("-" * (22 + 12 * len(class_names)) + "\n")
        for model_name, result in sorted_results:
            row = f"{model_name:<22}"
            for i in range(len(class_names)):
                v = result['per_class_auc'].get(i)
                row += f"{v:>12.4f}" if v is not None else f"{'N/A':>12}"
            f.write(row + "\n")
        f.write("\n")

        # ── Calibrated vs Uncalibrated AUC ────────────────────────────────
        f.write("=== CALIBRATED vs UNCALIBRATED MACRO AUC ===\n")
        f.write(f"{'Model':<22} {'Raw AUC':>10} {'Cal AUC':>10} {'Delta':>8}\n")
        f.write("-" * 54 + "\n")
        for model_name, result in sorted_results:
            raw = result.get('macro_roc_auc')
            cal = result.get('calibrated_macro_auc')
            raw_s   = f"{raw:.4f}" if raw is not None else "   N/A"
            cal_s   = f"{cal:.4f}" if cal is not None else "   N/A"
            delta_s = (f"{cal - raw:+.4f}" if raw is not None and cal is not None else "   N/A")
            f.write(f"{model_name:<22} {raw_s:>10} {cal_s:>10} {delta_s:>8}\n")
        f.write("\n")

        # ── Optimised threshold metrics ────────────────────────────────────
        f.write("=== OPTIMISED THRESHOLD METRICS ===\n")
        f.write(f"{'Model':<22} {'Acc(def)':>9} {'Acc(opt)':>9} "
                f"{'BalAcc(def)':>12} {'BalAcc(opt)':>12}\n")
        f.write("-" * 70 + "\n")
        for model_name, result in sorted_results:
            acc_def  = result['accuracy']
            acc_opt  = result.get('accuracy_opt')
            ba_def   = result['balanced_accuracy']
            ba_opt   = result.get('balanced_accuracy_opt')
            acc_opt_s = f"{acc_opt:.4f}" if acc_opt is not None else "   N/A"
            ba_opt_s  = f"{ba_opt:.4f}"  if ba_opt  is not None else "      N/A"
            f.write(f"{model_name:<22} {acc_def:>9.4f} {acc_opt_s:>9} "
                    f"{ba_def:>12.4f} {ba_opt_s:>12}\n")
        f.write("\n")

        # ── Optimal threshold values per class ─────────────────────────────
        f.write("=== OPTIMAL THRESHOLDS (OvR, on calibration set) ===\n")
        col_w = 18
        f.write(f"{'Model':<22}" + "".join(f"{c:>{col_w}}" for c in class_names) + "\n")
        f.write("-" * (22 + col_w * len(class_names)) + "\n")
        for model_name, result in sorted_results:
            thrs = result.get('optimal_thresholds')
            row = f"{model_name:<22}"
            for i in range(len(class_names)):
                if thrs and i in thrs:
                    thr_val, f1_val = thrs[i]
                    entry = f"{thr_val:.3f} (f1={f1_val:.2f})"
                else:
                    entry = "N/A"
                row += f"{entry:>{col_w}}"
            f.write(row + "\n")
        f.write("\n")

        # ── Detailed per-model results ─────────────────────────────────────
        f.write("=== DETAILED RESULTS ===\n\n")
        for model_name, result in sorted_results:
            f.write(f"Model: {model_name}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Accuracy:          {result['accuracy']:.4f}\n")
            f.write(f"Balanced Accuracy: {result['balanced_accuracy']:.4f}\n")
            f.write(f"MCC:               {result['mcc']:.4f}\n")
            f.write(f"Cohen's Kappa:     {result['kappa']:.4f}\n")
            if result['macro_roc_auc'] is not None:
                f.write(f"Macro ROC-AUC:     {result['macro_roc_auc']:.4f}\n")
            if result.get('calibrated_macro_auc') is not None:
                f.write(f"Calibrated AUC:    {result['calibrated_macro_auc']:.4f}\n")
            if result.get('accuracy_opt') is not None:
                f.write(f"Accuracy (opt thr): {result['accuracy_opt']:.4f}\n")
                f.write(f"Bal.Acc (opt thr):  {result['balanced_accuracy_opt']:.4f}\n")
            f.write("\nClassification Report (default threshold):\n")
            f.write(result['classification_report'])
            if result.get('report_opt'):
                f.write("\nClassification Report (optimised threshold):\n")
                f.write(result['report_opt'])
            f.write("\nConfusion Matrix:\n")
            f.write(str(result['confusion_matrix']))
            lh = result.get('loss_history')
            if lh is not None:
                _LOSS_METRICS = {'loss', 'logloss', 'multiclass', 'merror', 'mae', 'mse', 'rmse'}
                is_loss = any(tok in lh['metric'].lower() for tok in _LOSS_METRICS)
                best = int(np.argmin(lh['val']) if is_loss else np.argmax(lh['val']))
                f.write(f"\nTraining curve: metric={lh['metric']}, "
                        f"best_iter={best + 1}, "
                        f"final_train={lh['train'][-1]:.4f}, "
                        f"final_val={lh['val'][-1]:.4f}\n")
            f.write("\n" + "=" * 70 + "\n")


def main():
    try:
        config = ConfigManager('src/Config/configMC_1.yaml')
        catboost_dir = Path('src/Models/Multiclass_Target/catboost_info')
        catboost_dir.mkdir(parents=True, exist_ok=True)
        results, class_names = train_all_models(config)
        save_results(results, class_names, config)
        print("\nTraining completed. Results saved in output directory.")
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
