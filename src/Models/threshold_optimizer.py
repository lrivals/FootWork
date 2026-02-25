"""
Threshold optimisation utilities for binary and multiclass classifiers.

For binary models: find the decision threshold that maximises F1 on a
validation set (instead of the default 0.5).

For multiclass (OvR): find one threshold per class on the calibration set,
then predict by taking argmax of (proba / threshold), which boosts underrepresented
classes whose optimal threshold is lower than the default.
"""

import numpy as np
from sklearn.metrics import precision_recall_curve


def find_optimal_threshold(proba_pos, y_val):
    """
    Find the decision threshold that maximises F1-score for the positive class.

    Args:
        proba_pos: 1-D array of predicted probabilities for the positive class.
        y_val:     1-D binary array of true labels (0 / 1).

    Returns:
        (best_threshold, best_f1): the threshold and its F1 score.
    """
    precision, recall, thresholds = precision_recall_curve(y_val, proba_pos)
    # precision_recall_curve returns n+1 values for precision/recall but n for thresholds
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx]), float(f1[best_idx])


def find_optimal_thresholds_multiclass(model, X_val, y_val):
    """
    Find one optimal F1 threshold per class (OvR) on a validation set.

    Args:
        model: fitted classifier with predict_proba().
        X_val: feature matrix of validation set.
        y_val: 1-D integer array of true class labels.

    Returns:
        dict mapping class_index → (best_threshold, best_f1).
        e.g. {0: (0.28, 0.41), 1: (0.21, 0.35), 2: (0.39, 0.52)}
    """
    proba = model.predict_proba(X_val)
    n_classes = proba.shape[1]
    thresholds = {}
    for i in range(n_classes):
        y_bin = (y_val == i).astype(int)
        thr, f1 = find_optimal_threshold(proba[:, i], y_bin)
        thresholds[i] = (thr, f1)
    return thresholds


def predict_with_thresholds(proba, thresholds):
    """
    Predict class labels by rescaling probabilities by per-class thresholds,
    then taking the argmax of the rescaled scores.

    A class whose optimal threshold is lower than average gets a boost,
    making it more likely to be predicted — which helps minority classes
    like Draw or AwayWin.

    Args:
        proba:      2-D array (n_samples, n_classes) of predicted probabilities.
        thresholds: dict {class_idx: (threshold, f1)} from
                    find_optimal_thresholds_multiclass.

    Returns:
        1-D integer array of predicted class labels.
    """
    thr_array = np.array([thresholds[i][0] for i in range(proba.shape[1])])
    # Avoid division by zero
    thr_array = np.maximum(thr_array, 1e-6)
    adjusted = proba / thr_array
    return np.argmax(adjusted, axis=1)


def predict_binary_with_threshold(proba_pos, threshold):
    """
    Apply a custom decision threshold to binary probability predictions.

    Args:
        proba_pos: 1-D array of predicted probabilities for the positive class.
        threshold: scalar decision threshold.

    Returns:
        1-D binary integer array.
    """
    return (proba_pos >= threshold).astype(int)
