"""
DNN PyTorch pipeline for football match outcome prediction.

Architecture: Input(n_features) → Dense(256)+BN+ReLU+Dropout → Dense(128)+BN+ReLU+Dropout
              → Dense(64)+BN+ReLU+Dropout → Dense(3)+Softmax

Follows the same conventions as the multiclass sklearn pipeline:
  - Temporal 3-way split (train<2020, cal 2020-2021, test>=2022)
  - Isotonic calibration (per-class IsotonicRegression) on calibration set
  - Threshold optimization via threshold_optimizer.py
  - Same metrics and output format
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
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, matthews_corrcoef,
    cohen_kappa_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.calibration import CalibrationDisplay
from sklearn.isotonic import IsotonicRegression

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.Config.Config_Manager import ConfigManager
from src.Models.threshold_optimizer import (
    find_optimal_thresholds_multiclass, predict_with_thresholds
)

CAL_START_YEAR = 2020
CAL_END_YEAR   = 2022


# ---------------------------------------------------------------------------
# Neural Network architecture
# ---------------------------------------------------------------------------

class FootballMatchNet(nn.Module):
    """
    MLP with BatchNorm and Dropout for 3-class football match prediction.

    Args:
        input_dim   : number of input features
        hidden_layers : list of hidden layer sizes, e.g. [256, 128, 64]
        dropout_rates : dropout probability per hidden layer
        n_classes   : number of output classes (default 3)
    """

    def __init__(self, input_dim, hidden_layers, dropout_rates, n_classes=3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for size, drop in zip(hidden_layers, dropout_rates):
            layers += [
                nn.Linear(in_dim, size),
                nn.BatchNorm1d(size),
                nn.ReLU(),
                nn.Dropout(p=drop),
            ]
            in_dim = size
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Sklearn-compatible wrapper (for CalibratedClassifierCV)
# ---------------------------------------------------------------------------

class FootballNNWrapper(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper around FootballMatchNet.

    Exposes fit / predict_proba / predict so that CalibratedClassifierCV
    and threshold_optimizer can be applied without modification.
    """

    def __init__(self, input_dim, hidden_layers, dropout_rates, n_classes,
                 class_weights_tensor, epochs, batch_size, learning_rate,
                 weight_decay, early_stopping_patience, scheduler_patience,
                 scheduler_factor, device):
        self.input_dim              = input_dim
        self.hidden_layers          = hidden_layers
        self.dropout_rates          = dropout_rates
        self.n_classes              = n_classes
        self.class_weights_tensor   = class_weights_tensor
        self.epochs                 = epochs
        self.batch_size             = batch_size
        self.learning_rate          = learning_rate
        self.weight_decay           = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.scheduler_patience     = scheduler_patience
        self.scheduler_factor       = scheduler_factor
        self.device                 = device

        self.model_         = None
        self.classes_       = None
        self.loss_history_  = None   # {'train': [...], 'val': [...]}

    # ------------------------------------------------------------------
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Train the DNN.

        Args:
            X, y         : training data (numpy arrays)
            X_val, y_val : optional validation data for early stopping.
                           If None, 10% of training data is held out.
        """
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Validation split if not provided
        if X_val is None:
            n_val = max(1, int(0.1 * len(X)))
            idx = np.random.default_rng(42).permutation(len(X))
            val_idx, tr_idx = idx[:n_val], idx[n_val:]
            X_val, y_val = X[val_idx], y[val_idx]
            X, y = X[tr_idx], y[tr_idx]

        def to_tensors(Xa, ya):
            return (
                torch.tensor(Xa, dtype=torch.float32).to(self.device),
                torch.tensor(ya, dtype=torch.long).to(self.device),
            )

        Xt, yt = to_tensors(X, y)
        Xv, yv = to_tensors(X_val, y_val)

        dataset = TensorDataset(Xt, yt)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_ = FootballMatchNet(
            self.input_dim, self.hidden_layers, self.dropout_rates, n_classes
        ).to(self.device)

        cw = self.class_weights_tensor.to(self.device)
        criterion = nn.CrossEntropyLoss(weight=cw)
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.scheduler_patience,
            factor=self.scheduler_factor,
            min_lr=1e-5,
        )

        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            self.model_.train()
            epoch_loss = 0.0
            for Xb, yb in loader:
                optimizer.zero_grad()
                logits = self.model_(Xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(Xb)
            epoch_loss /= len(Xt)

            self.model_.eval()
            with torch.no_grad():
                val_logits = self.model_(Xv)
                val_loss = criterion(val_logits, yv).item()

            train_losses.append(epoch_loss)
            val_losses.append(val_loss)
            scheduler.step(val_loss)

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print(f"    Early stopping at epoch {epoch + 1} (best val_loss={best_val_loss:.4f})")
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.loss_history_ = {'train': train_losses, 'val': val_losses}
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X):
        self.model_.eval()
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model_(Xt)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# ---------------------------------------------------------------------------
# Isotonic calibration wrapper (avoids FrozenEstimator _estimator_type bug)
# ---------------------------------------------------------------------------

class _IsotonicCalibratedNN:
    """
    Per-class isotonic regression calibration for FootballNNWrapper.

    Replaces CalibratedClassifierCV(FrozenEstimator(...)) which fails because
    FrozenEstimator does not forward _estimator_type to sklearn's is_classifier().

    Implements predict_proba(X) compatible with threshold_optimizer.py.
    """

    def __init__(self, nn_wrapper):
        self.nn_wrapper = nn_wrapper
        self.calibrators_ = []
        self.classes_ = nn_wrapper.classes_

    def fit(self, X_cal, y_cal):
        raw_probs = self.nn_wrapper.predict_proba(X_cal)
        n_classes = raw_probs.shape[1]
        self.calibrators_ = []
        for i in range(n_classes):
            y_bin = (y_cal == i).astype(float)
            ir = IsotonicRegression(out_of_bounds='clip')
            ir.fit(raw_probs[:, i], y_bin)
            self.calibrators_.append(ir)
        return self

    def predict_proba(self, X):
        raw_probs = self.nn_wrapper.predict_proba(X)
        cal = np.column_stack([
            self.calibrators_[i].predict(raw_probs[:, i])
            for i in range(len(self.calibrators_))
        ])
        # Normalize rows to sum to 1
        row_sums = cal.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return cal / row_sums

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# ---------------------------------------------------------------------------
# Data loading (same 3-way temporal split as multiclass pipeline)
# ---------------------------------------------------------------------------

def load_and_prepare_data(config):
    input_path      = config.get_paths()['full_dataset']
    exclude_columns = config.get_config_value('excluded_columns', default=[])

    df = pd.read_csv(input_path)
    print("\nTarget distribution:")
    print(df['target_result'].value_counts())

    df['date'] = pd.to_datetime(df['date'])
    train_df = df[df['date'].dt.year <  CAL_START_YEAR].copy()
    cal_df   = df[(df['date'].dt.year >= CAL_START_YEAR) &
                  (df['date'].dt.year <  CAL_END_YEAR)].copy()
    test_df  = df[df['date'].dt.year >= CAL_END_YEAR].copy()
    print(f"\nTemporal split: "
          f"train<{CAL_START_YEAR} ({len(train_df)} rows), "
          f"cal {CAL_START_YEAR}-{CAL_END_YEAR-1} ({len(cal_df)} rows), "
          f"test>={CAL_END_YEAR} ({len(test_df)} rows)")

    cols_to_drop = [c for c in exclude_columns if c in df.columns and c != 'target_result']
    X_train = train_df.drop(cols_to_drop + ['target_result'], axis=1, errors='ignore')
    X_cal   = cal_df.drop(cols_to_drop   + ['target_result'], axis=1, errors='ignore')
    X_test  = test_df.drop(cols_to_drop  + ['target_result'], axis=1, errors='ignore')

    le = LabelEncoder()
    le.fit(df['target_result'])
    y_train = le.transform(train_df['target_result'])
    y_cal   = le.transform(cal_df['target_result'])
    y_test  = le.transform(test_df['target_result'])
    class_names = le.classes_  # alphabetical: ['AwayWin', 'Draw', 'HomeWin']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_cal_scaled   = scaler.transform(X_cal)
    X_test_scaled  = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, X_cal_scaled, y_train, y_test, y_cal, class_names


# ---------------------------------------------------------------------------
# Plot helpers (mirrors multiclass pipeline)
# ---------------------------------------------------------------------------

def _plot_loss_curve(loss_history, output_dir):
    train_vals = loss_history['train']
    val_vals   = loss_history['val']
    best_iter  = int(np.argmin(val_vals))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(train_vals) + 1), train_vals, label='Train',      linewidth=1.5)
    ax.plot(range(1, len(val_vals)   + 1), val_vals,   label='Validation', linewidth=1.5)
    ax.axvline(best_iter + 1, color='red', linestyle='--', alpha=0.7,
               label=f'Best epoch = {best_iter + 1}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('Training Curve — DNN PyTorch')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'loss_curve_DNN_PyTorch.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_roc_multiclass(probs, y_test, class_names, tag, output_dir):
    n_classes = len(class_names)
    y_bin = label_binarize(y_test, classes=list(range(n_classes)))

    fpr_dict, tpr_dict, auc_dict = {}, {}, {}
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        fpr_dict[i], tpr_dict[i] = fpr, tpr
        auc_dict[i] = auc(fpr, tpr)

    all_fpr  = np.unique(np.concatenate([fpr_dict[i] for i in range(n_classes)]))
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
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves (OvR) — {tag}')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'roc_{tag}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    return auc_dict, macro_auc


def _plot_calibration_curve(wrapper, calibrated_model, X_cal, y_cal, class_names, output_dir):
    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 5))

    for i, (cls, ax) in enumerate(zip(class_names, axes)):
        y_bin = (y_cal == i).astype(int)
        raw_proba = wrapper.predict_proba(X_cal)[:, i]
        CalibrationDisplay.from_predictions(y_bin, raw_proba, n_bins=10, ax=ax,
                                            name='Uncalibrated', color='steelblue')
        cal_proba = calibrated_model.predict_proba(X_cal)[:, i]
        CalibrationDisplay.from_predictions(y_bin, cal_proba, n_bins=10, ax=ax,
                                            name='Calibrated (isotonic)', color='coral')
        ax.set_title(f'DNN PyTorch — {cls}')
        ax.legend()

    fig.suptitle('Calibration Curves — DNN PyTorch', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'calibration_DNN_PyTorch.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_confusion_matrix(y_true, y_pred, class_names, tag, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f'Confusion Matrix — {tag}')
    ax.set_ylabel('True Label'); ax.set_xlabel('Predicted Label')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'confusion_matrix_{tag}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def train_nn_pipeline(config_path='src/Config/configNN_1.yaml'):
    config     = ConfigManager(config_path)
    output_dir = config.get_paths()['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    arch_cfg     = config.get_config_value('architecture', default={})
    train_cfg    = config.get_config_value('training', default={})
    cw_cfg       = config.get_config_value('class_weights', default={})

    hidden_layers  = arch_cfg.get('hidden_layers',  [256, 128, 64])
    dropout_rates  = arch_cfg.get('dropout_rates',  [0.3, 0.3, 0.2])

    epochs                   = train_cfg.get('epochs',                   200)
    batch_size               = train_cfg.get('batch_size',               256)
    learning_rate            = train_cfg.get('learning_rate',            0.001)
    weight_decay             = train_cfg.get('weight_decay',             0.0001)
    early_stopping_patience  = train_cfg.get('early_stopping_patience',  30)
    scheduler_patience       = train_cfg.get('scheduler_patience',       10)
    scheduler_factor         = train_cfg.get('scheduler_factor',         0.5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # ── Load data ──────────────────────────────────────────────────────────
    X_train, X_test, X_cal, y_train, y_test, y_cal, class_names = \
        load_and_prepare_data(config)

    n_features = X_train.shape[1]
    n_classes  = len(class_names)
    print(f"Features: {n_features}  |  Classes: {list(class_names)}")

    # Class weights tensor — aligned with LabelEncoder order (alphabetical)
    # class_names: ['AwayWin', 'Draw', 'HomeWin']
    weights_list = [
        cw_cfg.get('AwayWin', 1.0),
        cw_cfg.get('Draw',    1.74),
        cw_cfg.get('HomeWin', 1.0),
    ]
    class_weights_tensor = torch.tensor(weights_list, dtype=torch.float32)

    # ── Train ──────────────────────────────────────────────────────────────
    print("\nTraining DNN PyTorch...")
    wrapper = FootballNNWrapper(
        input_dim=n_features,
        hidden_layers=hidden_layers,
        dropout_rates=dropout_rates,
        n_classes=n_classes,
        class_weights_tensor=class_weights_tensor,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
        scheduler_patience=scheduler_patience,
        scheduler_factor=scheduler_factor,
        device=device,
    )
    wrapper.fit(X_train, y_train, X_val=X_cal, y_val=y_cal)

    # ── Raw evaluation ─────────────────────────────────────────────────────
    probs_test = wrapper.predict_proba(X_test)
    y_pred     = np.argmax(probs_test, axis=1)

    acc     = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    mcc     = matthews_corrcoef(y_test, y_pred)
    kappa   = cohen_kappa_score(y_test, y_pred)
    print(f"\nRaw metrics — Accuracy: {acc:.4f}  Balanced: {bal_acc:.4f}  MCC: {mcc:.4f}")

    per_class_auc, macro_auc = _plot_roc_multiclass(
        probs_test, y_test, class_names, 'DNN_PyTorch', output_dir
    )
    print(f"Macro AUC: {macro_auc:.4f}  |  Per-class: "
          + "  ".join(f"{c}={per_class_auc[i]:.4f}" for i, c in enumerate(class_names)))

    _plot_loss_curve(wrapper.loss_history_, output_dir)
    _plot_confusion_matrix(y_test, y_pred, class_names, 'DNN_PyTorch_raw', output_dir)

    # ── Calibration ────────────────────────────────────────────────────────
    print("\nCalibrating with isotonic regression on calibration set...")
    calibrated = _IsotonicCalibratedNN(wrapper).fit(X_cal, y_cal)

    cal_probs_test = calibrated.predict_proba(X_test)
    _, cal_macro_auc = _plot_roc_multiclass(
        cal_probs_test, y_test, class_names, 'DNN_PyTorch_calibrated', output_dir
    )
    _plot_calibration_curve(wrapper, calibrated, X_cal, y_cal, class_names, output_dir)
    print(f"Calibrated Macro AUC: {cal_macro_auc:.4f}  (delta: {cal_macro_auc - macro_auc:+.4f})")

    # ── Threshold optimisation ─────────────────────────────────────────────
    print("\nOptimising decision thresholds on calibration set...")
    optimal_thresholds = find_optimal_thresholds_multiclass(calibrated, X_cal, y_cal)
    y_pred_opt         = predict_with_thresholds(cal_probs_test, optimal_thresholds)

    acc_opt     = accuracy_score(y_test, y_pred_opt)
    bal_acc_opt = balanced_accuracy_score(y_test, y_pred_opt)
    report_opt  = classification_report(y_test, y_pred_opt, target_names=class_names)
    print(f"Optimised thresholds — Accuracy: {acc_opt:.4f}  Balanced: {bal_acc_opt:.4f}")

    _plot_confusion_matrix(y_test, y_pred_opt, class_names, 'DNN_PyTorch_opt', output_dir)

    # ── Save results ───────────────────────────────────────────────────────
    _save_results(
        output_dir=output_dir,
        class_names=class_names,
        acc=acc, bal_acc=bal_acc, mcc=mcc, kappa=kappa,
        macro_auc=macro_auc, per_class_auc=per_class_auc,
        cal_macro_auc=cal_macro_auc,
        optimal_thresholds=optimal_thresholds,
        acc_opt=acc_opt, bal_acc_opt=bal_acc_opt,
        report_default=classification_report(y_test, y_pred, target_names=class_names),
        report_opt=report_opt,
        cm_default=confusion_matrix(y_test, y_pred),
        cm_opt=confusion_matrix(y_test, y_pred_opt),
        loss_history=wrapper.loss_history_,
    )
    print(f"\nResults saved to {output_dir}/metrics_results_DNN_PyTorch.txt")
    return wrapper, calibrated


def _save_results(output_dir, class_names, acc, bal_acc, mcc, kappa,
                  macro_auc, per_class_auc, cal_macro_auc,
                  optimal_thresholds, acc_opt, bal_acc_opt,
                  report_default, report_opt, cm_default, cm_opt, loss_history):
    filepath = os.path.join(output_dir, 'metrics_results_DNN_PyTorch.txt')
    with open(filepath, 'w') as f:
        f.write("Results — DNN PyTorch (Football Match Prediction)\n")
        f.write("=" * 70 + "\n\n")

        f.write("=== BASELINE METRICS (default threshold) ===\n")
        f.write(f"{'Accuracy':<22}: {acc:.4f}\n")
        f.write(f"{'Balanced Accuracy':<22}: {bal_acc:.4f}\n")
        f.write(f"{'MCC':<22}: {mcc:.4f}\n")
        f.write(f"{'Cohen Kappa':<22}: {kappa:.4f}\n")
        f.write(f"{'Macro ROC-AUC':<22}: {macro_auc:.4f}\n\n")

        f.write("=== PER-CLASS AUC (OvR) ===\n")
        for i, cls in enumerate(class_names):
            f.write(f"  {cls:<12}: {per_class_auc[i]:.4f}\n")
        f.write("\n")

        f.write("=== CALIBRATED vs UNCALIBRATED MACRO AUC ===\n")
        f.write(f"  Raw AUC : {macro_auc:.4f}\n")
        f.write(f"  Cal AUC : {cal_macro_auc:.4f}\n")
        f.write(f"  Delta   : {cal_macro_auc - macro_auc:+.4f}\n\n")

        f.write("=== OPTIMISED THRESHOLD METRICS ===\n")
        f.write(f"  Accuracy (default) : {acc:.4f}\n")
        f.write(f"  Accuracy (opt)     : {acc_opt:.4f}\n")
        f.write(f"  Bal.Acc (default)  : {bal_acc:.4f}\n")
        f.write(f"  Bal.Acc (opt)      : {bal_acc_opt:.4f}\n\n")

        f.write("=== OPTIMAL THRESHOLDS (OvR, on calibration set) ===\n")
        for i, cls in enumerate(class_names):
            if i in optimal_thresholds:
                thr, f1 = optimal_thresholds[i]
                f.write(f"  {cls:<12}: threshold={thr:.3f}  F1={f1:.3f}\n")
        f.write("\n")

        f.write("=== CLASSIFICATION REPORT (default threshold) ===\n")
        f.write(report_default + "\n")

        f.write("=== CLASSIFICATION REPORT (optimised threshold) ===\n")
        f.write(report_opt + "\n")

        f.write("=== CONFUSION MATRIX (default threshold) ===\n")
        f.write(str(cm_default) + "\n\n")

        f.write("=== CONFUSION MATRIX (optimised threshold) ===\n")
        f.write(str(cm_opt) + "\n\n")

        f.write("=== TRAINING CURVE SUMMARY ===\n")
        train_vals = loss_history['train']
        val_vals   = loss_history['val']
        best       = int(np.argmin(val_vals))
        f.write(f"  Total epochs     : {len(train_vals)}\n")
        f.write(f"  Best epoch       : {best + 1}\n")
        f.write(f"  Best val_loss    : {val_vals[best]:.4f}\n")
        f.write(f"  Final train_loss : {train_vals[-1]:.4f}\n")
        f.write(f"  Final val_loss   : {val_vals[-1]:.4f}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    config_path = 'src/Config/configNN_1.yaml'
    train_nn_pipeline(config_path)
    print("\nDNN PyTorch pipeline completed.")


if __name__ == '__main__':
    main()
