import re
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score
)

def evaluate_prob(y_true, y_prob, weights, verbose=True):
    """Evaluate probabilistic predictions with weighted metrics."""
    roc = roc_auc_score(y_true, y_prob, sample_weight=weights)
    pr = average_precision_score(y_true, y_prob, sample_weight=weights)
    if verbose:
        print(f"ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f}")
    return {"roc_auc": roc, "pr_auc": pr}

def threshold_report(y_true, y_prob, weights, target="f1", min_recall=None, min_precision=None):
    """
    Sweep thresholds and return the best threshold by chosen objective.
    Supports constraints like min_recall or min_precision.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best = {"thr": 0.5, "precision": 0, "recall": 0, "f1": 0}
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        prec = precision_score(y_true, y_pred, sample_weight=weights, zero_division=0)
        rec = recall_score(y_true, y_pred, sample_weight=weights, zero_division=0)
        f1 = f1_score(y_true, y_pred, sample_weight=weights, zero_division=0)
        if min_recall is not None and rec < min_recall: 
            continue
        if min_precision is not None and prec < min_precision:
            continue
        score = {"precision": prec, "recall": rec, "f1": f1}[target]
        if score > best[target]:
            best = {"thr": float(t), "precision": float(prec), "recall": float(rec), "f1": float(f1)}
    return best

def get_feature_names(preprocessor, num_cols, cat_cols):
    """Get feature names after preprocessing transformations."""
    names = []
    # numeric
    names += list(preprocessor.named_transformers_["num"].get_feature_names_out(num_cols))
    # categorical (ohe)
    ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
    cat_names = ohe.get_feature_names_out(cat_cols)
    names += list(cat_names)
    return np.array([re.sub(r"[\\[\\]<>]", "_", n) for n in names])

def _norm_key(s):
    """Normalize string keys for consistent mapping."""
    s = "" if pd.isna(s) else str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s