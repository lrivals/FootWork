import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score,
                             balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score,
                             roc_curve, auc)
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
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


def get_models(config):
    model_params = config.get_config_value('model_parameters', default={})

    catboost_params = dict(model_params.get('catboost', {}))
    catboost_params['train_dir'] = 'src/Models/Binary_Target/catboost_info'

    return {
        'Random Forest':       RandomForestClassifier(**model_params.get('random_forest', {})),
        'Logistic Regression': LogisticRegression(**model_params.get('logistic_regression', {})),
        'SVM':                 SVC(**model_params.get('svm', {})),
        'Gradient Boosting':   GradientBoostingClassifier(**model_params.get('gradient_boosting', {})),
        'XGBoost':             XGBClassifier(**model_params.get('xgboost', {})),
        'LightGBM':            LGBMClassifier(**model_params.get('lightgbm', {})),
        'CatBoost':            CatBoostClassifier(**catboost_params),
        'Neural Network':     MLPClassifier(**model_params.get('neural_network', {})),
        'KNN':                 KNeighborsClassifier(**model_params.get('knn', {})),
        'AdaBoost':            AdaBoostClassifier(**model_params.get('adaboost', {})),
        'Extra Trees':         ExtraTreesClassifier(**model_params.get('extra_trees', {}))
    }


def load_and_prepare_binary_data(config):
    input_path = config.get_paths()['full_dataset']
    exclude_columns = config.get_config_value('excluded_columns', default=[])
    split_config = config.get_config_value('data_split', default={'test_size': 0.2, 'random_state': 42})

    df = pd.read_csv(input_path)

    df['home_win'] = (df['target_result'] == 'HomeWin').astype(int)
    df['away_win'] = (df['target_result'] == 'AwayWin').astype(int)

    cols_to_drop = [c for c in exclude_columns + ['target_result', 'home_win', 'away_win']
                    if c in df.columns]

    temporal_split_year = split_config.get('temporal_split_year', None)
    cal_split_year      = split_config.get('cal_split_year', 2020)
    if temporal_split_year and 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        train_df = df[df['date'].dt.year < cal_split_year].copy()
        cal_df   = df[(df['date'].dt.year >= cal_split_year) &
                      (df['date'].dt.year < temporal_split_year)].copy()
        test_df  = df[df['date'].dt.year >= temporal_split_year].copy()
        print(f"\nTemporal split: train < {cal_split_year} ({len(train_df)} rows), "
              f"cal {cal_split_year}–{temporal_split_year-1} ({len(cal_df)} rows), "
              f"test >= {temporal_split_year} ({len(test_df)} rows)")
        X_train = train_df.drop(cols_to_drop, axis=1, errors='ignore')
        X_cal   = cal_df.drop(cols_to_drop, axis=1, errors='ignore')
        X_test  = test_df.drop(cols_to_drop, axis=1, errors='ignore')
        y_home_train, y_home_cal, y_home_test = (
            train_df['home_win'], cal_df['home_win'], test_df['home_win']
        )
        y_away_train, y_away_cal, y_away_test = (
            train_df['away_win'], cal_df['away_win'], test_df['away_win']
        )
    else:
        random_params = {k: v for k, v in split_config.items() if k != 'temporal_split_year'}
        X = df.drop(cols_to_drop, axis=1, errors='ignore')
        X_train, X_test, y_home_train, y_home_test, y_away_train, y_away_test = train_test_split(
            X, df['home_win'], df['away_win'], **random_params
        )
        X_cal = None
        y_home_cal, y_away_cal = None, None

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_cal_scaled   = scaler.transform(X_cal) if X_cal is not None else None
    X_test_scaled  = scaler.transform(X_test)

    return (X_train_scaled, X_test_scaled, X_cal_scaled,
            y_home_train, y_home_test, y_home_cal,
            y_away_train, y_away_test, y_away_cal)


# ---------------------------------------------------------------------------
# Loss tracking helpers
# ---------------------------------------------------------------------------

def _fit_with_tracking(model, X_train, X_test, y_train, y_test, sample_weight=None):
    """
    Fit model and capture per-iteration loss/accuracy where supported.
    Returns (fitted_model, loss_history | None)
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
        # Isolate CatBoost log dir per model instance to avoid concurrent write conflicts
        base_dir = model.get_param('train_dir') or 'catboost_info'
        safe_dir = os.path.join(base_dir, f'run_{id(model)}')
        os.makedirs(safe_dir, exist_ok=True)
        model.set_params(train_dir=safe_dir)
        model.fit(X_train, y_train, eval_set=(X_test, y_test))
        evals = model.get_evals_result()
        learn_key  = 'learn'      if 'learn'      in evals else list(evals.keys())[0]
        val_key    = 'validation' if 'validation' in evals else list(evals.keys())[-1]
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

def _plot_loss_curve(loss_history, model_name, target_type, output_dir):
    """Save train vs. validation loss/accuracy curve for one model."""
    train_vals = loss_history['train']
    val_vals   = loss_history['val']
    metric     = loss_history['metric']

    _LOSS_METRICS = {'loss', 'logloss', 'multiclass', 'merror', 'mae', 'mse', 'rmse'}
    is_loss   = any(tok in metric.lower() for tok in _LOSS_METRICS)
    best_iter = int(np.argmin(val_vals) if is_loss else np.argmax(val_vals))

    fig, ax = plt.subplots(figsize=(8, 5))
    iters = range(1, len(train_vals) + 1)
    ax.plot(iters, train_vals, label='Train',      linewidth=1.5)
    ax.plot(iters, val_vals,   label='Validation', linewidth=1.5)
    ax.axvline(best_iter + 1, color='red', linestyle='--', alpha=0.7,
               label=f'Best iter = {best_iter + 1}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel(metric)
    ax.set_title(f'Training Curve — {model_name} ({target_type})')
    ax.legend()
    fig.tight_layout()
    safe_model  = model_name.replace(' ', '_')
    safe_target = target_type.replace(' ', '_')
    fig.savefig(os.path.join(output_dir, f'loss_curve_{safe_target}_{safe_model}.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_roc_curves(models_dict, X_test, y_test, dataset_name, target_type, output_dir):
    """All-models ROC overlay (binary classification)."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for model_name, result in models_dict.items():
        try:
            model = result['model']
            if hasattr(model, 'predict_proba'):
                y_score = model.predict_proba(X_test)[:, 1]
            else:
                y_score = model.decision_function(X_test)

            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc_val = auc(fpr, tpr)
            result['roc_auc'] = roc_auc_val

            ax.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc_val:.2f})')
        except Exception as e:
            print(f"Warning: Could not generate ROC curve for {model_name}: {e}")
            continue

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves — {dataset_name} ({target_type})')
    ax.legend(loc='lower right')
    fig.tight_layout()
    safe_target = target_type.replace(' ', '_')
    fig.savefig(os.path.join(output_dir, f'roc_curves_{dataset_name}_{safe_target}.png'),
                bbox_inches='tight', dpi=150)
    plt.close(fig)


def _plot_summary_charts(home_results, away_results, output_dir):
    """Bar charts comparing all models for Home Win and Away Win."""
    for label, results in [("Home Win", home_results), ("Away Win", away_results)]:
        model_names = list(results.keys())
        accuracies  = [r['accuracy']          for r in results.values()]
        bal_accs    = [r['balanced_accuracy'] for r in results.values()]
        roc_aucs    = [r.get('roc_auc') or 0.0 for r in results.values()]

        x     = np.arange(len(model_names))
        width = 0.28

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Model Comparison — {label}', fontsize=14)

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
        ax.bar(x, roc_aucs, color='mediumseagreen')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=30, ha='right')
        ax.set_ylim([0, 1])
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random baseline')
        ax.set_ylabel('ROC-AUC')
        ax.set_title('ROC-AUC by Model')
        ax.legend()

        fig.tight_layout()
        safe = label.replace(' ', '_')
        fig.savefig(os.path.join(output_dir, f'summary_comparison_{safe}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)


def _plot_calibration_curve_binary(model, calibrated_model, X_cal, y_cal,
                                   model_name, target_type, output_dir):
    """Save binary calibration curve (uncalibrated vs calibrated)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    if hasattr(model, 'predict_proba'):
        prob_raw = model.predict_proba(X_cal)[:, 1]
        CalibrationDisplay.from_predictions(
            y_cal, prob_raw, n_bins=10, ax=ax, label='Uncalibrated', color='steelblue'
        )
    prob_cal = calibrated_model.predict_proba(X_cal)[:, 1]
    CalibrationDisplay.from_predictions(
        y_cal, prob_cal, n_bins=10, ax=ax, label='Calibrated', color='darkorange'
    )
    ax.set_title(f'Calibration — {model_name} ({target_type})')
    ax.legend(fontsize=9)
    fig.tight_layout()
    safe_m = model_name.replace(' ', '_')
    safe_t = target_type.replace(' ', '_')
    fig.savefig(os.path.join(output_dir, f'calibration_{safe_t}_{safe_m}.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Core train / evaluate
# ---------------------------------------------------------------------------

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test,
                             target_type, dataset_name, output_dir,
                             X_cal=None, y_cal=None):
    # Compute sample weights for models that don't support class_weight in constructor
    sample_weight = compute_sample_weight('balanced', y_train)

    # CatBoost: set class_weights dynamically based on target imbalance
    if model.__class__.__name__ == 'CatBoostClassifier':
        cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=np.array(y_train))
        model.set_params(class_weights=list(cw))

    model, loss_history = _fit_with_tracking(
        model, X_train, X_test, y_train, y_test, sample_weight=sample_weight
    )

    y_pred = model.predict(X_test)
    class_labels = [f'Not {target_type}', target_type]

    results = {
        'model':              model,
        'accuracy':           accuracy_score(y_test, y_pred),
        'balanced_accuracy':  balanced_accuracy_score(y_test, y_pred),
        'mcc':                matthews_corrcoef(y_test, y_pred),
        'kappa':              cohen_kappa_score(y_test, y_pred),
        'loss_history':       loss_history,
        'confusion_matrix':   confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(
            y_test, y_pred, target_names=class_labels),
        # Calibration / threshold fields
        'calibrated_roc_auc': None,
        'optimal_threshold':  None,
        'accuracy_opt':       None,
        'balanced_accuracy_opt': None,
        'report_opt':         None,
    }

    # Calibration and threshold optimisation are run serially in train_all_models()
    # after the ThreadPoolExecutor block, so exceptions are fully visible.

    # Confusion matrix plot (OO API — thread-safe)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    ax.set_title(f'Confusion Matrix — {model.__class__.__name__} ({dataset_name} — {target_type})')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    fig.tight_layout()
    safe_target = target_type.replace(' ', '_')
    fig.savefig(os.path.join(output_dir,
        f'confusion_matrix_{dataset_name}_{safe_target}_{model.__class__.__name__}.png'), dpi=150)
    plt.close(fig)

    if loss_history is not None:
        _plot_loss_curve(loss_history, model.__class__.__name__, target_type, output_dir)

    return results


def train_all_models(config):
    (X_train, X_test, X_cal,
     y_home_train, y_home_test, y_home_cal,
     y_away_train, y_away_test, y_away_cal) = load_and_prepare_binary_data(config)
    output_dir = config.get_paths()['output_dir']
    models = get_models(config)

    max_workers = min(len(models), 5)
    print(f"\nLaunching {max_workers} parallel training workers for {len(models)} models "
          f"(home + away per worker)...")

    def _train_one(name, model):
        h = train_and_evaluate_model(
            model, X_train, X_test, y_home_train, y_home_test,
            'Home Win', 'Full_Dataset', output_dir,
            X_cal=X_cal, y_cal=y_home_cal
        )
        model_away = type(model)(**model.get_params())
        a = train_and_evaluate_model(
            model_away, X_train, X_test, y_away_train, y_away_test,
            'Away Win', 'Full_Dataset', output_dir,
            X_cal=X_cal, y_cal=y_away_cal
        )
        return name, h, a

    home_results = {}
    away_results = {}
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
                    name, h, a = future.result()
                    home_results[name] = h
                    away_results[name] = a
                    pbar.set_postfix_str(
                        f"{name}  home={h['accuracy']:.4f}  away={a['accuracy']:.4f}"
                    )
                except Exception as e:
                    pbar.set_postfix_str(f"{name}  ERROR")
                    print(f"  [error] {name}: {e}")

    # ── Serial calibration + threshold optimisation ────────────────────────
    # Run after ThreadPoolExecutor so exceptions are fully visible (not swallowed
    # by tqdm in worker threads).
    from src.Models.threshold_optimizer import find_optimal_threshold, predict_binary_with_threshold

    print("\nCalibration + threshold optimisation (serial)...")
    for name in list(home_results.keys()):
        for target_label, result, y_cal_t, y_test_t in [
            ('Home Win', home_results[name], y_home_cal, y_home_test),
            ('Away Win', away_results[name], y_away_cal, y_away_test),
        ]:
            model = result['model']
            if not hasattr(model, 'predict_proba') or X_cal is None or y_cal_t is None:
                continue
            try:
                calibrated = CalibratedClassifierCV(FrozenEstimator(model), method='isotonic')
                calibrated.fit(X_cal, y_cal_t)
                cal_probs_test = calibrated.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test_t, cal_probs_test)
                result['calibrated_roc_auc'] = auc(fpr, tpr)
                result['calibrated_model']   = calibrated

                _plot_calibration_curve_binary(
                    model, calibrated, X_cal, y_cal_t,
                    name, target_label, output_dir
                )

                cal_probs_cal = calibrated.predict_proba(X_cal)[:, 1]
                opt_thr, opt_f1 = find_optimal_threshold(cal_probs_cal, np.array(y_cal_t))
                y_pred_opt = predict_binary_with_threshold(cal_probs_test, opt_thr)
                result['optimal_threshold']       = (opt_thr, opt_f1)
                result['accuracy_opt']            = accuracy_score(y_test_t, y_pred_opt)
                result['balanced_accuracy_opt']   = balanced_accuracy_score(y_test_t, y_pred_opt)
                result['report_opt']              = classification_report(
                    y_test_t, y_pred_opt,
                    target_names=[f'Not {target_label}', target_label]
                )
                print(f"  {name} / {target_label}: "
                      f"cal_auc={result['calibrated_roc_auc']:.4f}  "
                      f"acc_opt={result['accuracy_opt']:.4f}")
            except Exception:
                import traceback
                print(f"\n[Calibration FAILED for {name} / {target_label}]:")
                traceback.print_exc()

    plot_roc_curves(home_results, X_test, y_home_test, "Full_Dataset", "Home Win", output_dir)
    plot_roc_curves(away_results, X_test, y_away_test, "Full_Dataset", "Away Win", output_dir)
    _plot_summary_charts(home_results, away_results, output_dir)

    return home_results, away_results


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_all_results(home_results, away_results, config):
    output_dir = config.get_paths()['output_dir']

    for target_type, results in [("Home_Win", home_results), ("Away_Win", away_results)]:
        filename = os.path.join(output_dir, f"metrics_results_Full_Dataset_{target_type}.txt")
        sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

        with open(filename, 'w') as f:
            f.write(f"Results for Full Dataset - {target_type}\n")
            f.write("=" * 70 + "\n\n")

            # Ranking table
            f.write("=== RANKING TABLE ===\n")
            header = (f"{'Model':<22} {'Accuracy':>9} {'Bal.Acc':>9} "
                      f"{'MCC':>8} {'Kappa':>8} {'ROC-AUC':>9}\n")
            f.write(header)
            f.write("-" * 70 + "\n")
            for model_name, result in sorted_results:
                roc_str = (f"{result['roc_auc']:.4f}"
                           if result.get('roc_auc') is not None else "  N/A  ")
                f.write(f"{model_name:<22} {result['accuracy']:>9.4f} "
                        f"{result['balanced_accuracy']:>9.4f} "
                        f"{result['mcc']:>8.4f} {result['kappa']:>8.4f} {roc_str:>9}\n")
            f.write("\n")

            # Calibrated vs Uncalibrated AUC
            f.write("=== CALIBRATED vs UNCALIBRATED ROC-AUC ===\n")
            f.write(f"{'Model':<22} {'Uncalibrated':>14} {'Calibrated':>12} {'Delta':>8}\n")
            f.write("-" * 60 + "\n")
            for model_name, result in sorted_results:
                raw = result.get('roc_auc')
                cal = result.get('calibrated_roc_auc')
                raw_s = f"{raw:.4f}" if raw is not None else "   N/A"
                cal_s = f"{cal:.4f}" if cal is not None else "   N/A"
                delta_s = f"{cal - raw:+.4f}" if (raw is not None and cal is not None) else "   N/A"
                f.write(f"{model_name:<22} {raw_s:>14} {cal_s:>12} {delta_s:>8}\n")
            f.write("\n")

            # Optimised threshold metrics
            f.write("=== OPTIMISED THRESHOLD METRICS ===\n")
            f.write(f"{'Model':<22} {'Acc(default)':>13} {'Acc(opt)':>10} "
                    f"{'BalAcc(default)':>16} {'BalAcc(opt)':>12}\n")
            f.write("-" * 77 + "\n")
            for model_name, result in sorted_results:
                acc_d   = result.get('accuracy', float('nan'))
                acc_o   = result.get('accuracy_opt')
                bal_d   = result.get('balanced_accuracy', float('nan'))
                bal_o   = result.get('balanced_accuracy_opt')
                acc_o_s = f"{acc_o:.4f}" if acc_o is not None else "   N/A"
                bal_o_s = f"{bal_o:.4f}" if bal_o is not None else "   N/A"
                f.write(f"{model_name:<22} {acc_d:>13.4f} {acc_o_s:>10} "
                        f"{bal_d:>16.4f} {bal_o_s:>12}\n")
            f.write("\n")

            # Detailed per-model results
            f.write("=== DETAILED RESULTS ===\n\n")
            for model_name, result in sorted_results:
                f.write(f"Model: {model_name}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Accuracy:          {result['accuracy']:.4f}\n")
                f.write(f"Balanced Accuracy: {result['balanced_accuracy']:.4f}\n")
                f.write(f"MCC:               {result['mcc']:.4f}\n")
                f.write(f"Cohen's Kappa:     {result['kappa']:.4f}\n")
                if result.get('roc_auc') is not None:
                    f.write(f"ROC-AUC:           {result['roc_auc']:.4f}\n")
                if result.get('calibrated_roc_auc') is not None:
                    f.write(f"Calibrated ROC-AUC:{result['calibrated_roc_auc']:.4f}\n")
                if result.get('accuracy_opt') is not None:
                    f.write(f"Accuracy (opt thr):         {result['accuracy_opt']:.4f}\n")
                    f.write(f"Balanced Acc (opt thr):     {result['balanced_accuracy_opt']:.4f}\n")
                f.write("\nClassification Report:\n")
                f.write(result['classification_report'])
                if result.get('report_opt'):
                    f.write("\nClassification Report (Optimised Threshold):\n")
                    f.write(result['report_opt'])
                if result.get('optimal_threshold') is not None:
                    thr, f1 = result['optimal_threshold']
                    f.write(f"\nOptimal Threshold: {thr:.4f}  (best F1={f1:.4f})\n")
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


def ensure_directories(config):
    output_dir = config.get_paths()['output_dir']
    for dir_path in [Path(output_dir), Path('src/Models/Binary_Target/catboost_info')]:
        dir_path.mkdir(parents=True, exist_ok=True)


def main():
    try:
        config = ConfigManager('src/Config/configBT_1.yaml')
        ensure_directories(config)
        home_results, away_results = train_all_models(config)
        save_all_results(home_results, away_results, config)
        print("\nTraining completed. Results saved in output directory.")
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
