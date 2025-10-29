import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

try:
    from xgboost import XGBClassifier
    XGB_OK = True
except Exception:
    XGB_OK = False

from src.config import RANDOM_STATE
from src.utils.metrics import evaluate_prob, threshold_report

def create_models(preprocessor):
    """Create a list of models with preprocessing pipeline."""
    models = []
    
    # Logistic Regression
    logit = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", LogisticRegression(
            solver="liblinear",
            class_weight=None,
            random_state=RANDOM_STATE,
            max_iter=200
        ))
    ])
    models.append(("logit", logit))
    
    # Random Forest
    rf = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=100,  # Reduced from 500 to 100
            max_depth=10,      # Added max_depth to limit tree size
            min_samples_split=2,
            n_jobs=-1,
            random_state=RANDOM_STATE
        ))
    ])
    models.append(("rf", rf))
    
    # XGBoost (if available)
    if XGB_OK:
        xgb = Pipeline(steps=[
            ("prep", preprocessor),
            ("clf", XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                tree_method="hist",
                eval_metric="logloss",
                n_jobs=-1
            ))
        ])
        models.append(("xgb", xgb))
    
    return models

def train_and_evaluate_models(models, X_train, y_train, X_test, y_test, w_train, w_test):
    """Train models and evaluate their performance."""
    results = []
    for name, pipe in models:
        # Fit model
        pipe.fit(X_train, y_train, **({"clf__sample_weight": w_train} if name!="xgb" else {"clf__sample_weight": w_train}))
        
        # Predict probabilities
        y_prob = pipe.predict_proba(X_test)[:,1]
        
        # Evaluate
        print(f"\n[{name}]")
        metrics = evaluate_prob(y_test, y_prob, w_test, verbose=True)
        best = threshold_report(y_test, y_prob, w_test, target="f1")
        print("Best threshold by F1:", best)
        
        results.append((name, pipe, metrics, best))
    
    return sorted(results, key=lambda r: r[2]["pr_auc"], reverse=True)

def calibrate_models(results_sorted, X_train, y_train, X_test, y_test, w_train, w_test):
    """Calibrate top models using sigmoid calibration."""
    calibrated = []
    cal_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    
    for name, pipe, metrics, best in results_sorted[:2]:
        print(f"\n== Calibrating {name} with sigmoid + 3-fold StratifiedKFold ==")
        
        # Get pipeline components
        base_prep = pipe.named_steps["prep"]
        base_est = pipe.named_steps["clf"]
        
        # Create fresh copies
        prep_clone = clone(base_prep)
        est_clone = clone(base_est)
        
        # Create calibrated pipeline
        cal = CalibratedClassifierCV(
            estimator=est_clone,  # Changed from base_estimator to estimator
            method="sigmoid",
            cv=cal_cv,
        )
        cal_pipe = Pipeline([("prep", prep_clone), ("cal", cal)])
        
        # Fit and evaluate
        cal_pipe.fit(X_train, y_train, cal__sample_weight=w_train)
        y_prob = cal_pipe.predict_proba(X_test)[:, 1]
        print("prob range:", float(y_prob.min()), "→", float(y_prob.max()))
        metrics2 = evaluate_prob(y_test, y_prob, w_test, verbose=True)
        best2 = threshold_report(y_test, y_prob, w_test, target="f1")
        print("Best threshold by F1:", best2)
        
        calibrated.append((f"cal_{name}", cal_pipe, metrics2, best2))
    
    return sorted(calibrated, key=lambda r: r[2]["pr_auc"], reverse=True)

def cross_validate_best_model(best_pipe, X_train, y_train, w_train):
    """Perform cross-validation on the best model."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_roc, cv_pr = [], []
    
    for tr_idx, va_idx in skf.split(X_train, y_train):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        w_tr, w_va = w_train[tr_idx], w_train[va_idx]
        
        # Fit
        best_pipe.fit(X_tr, y_tr, cal__sample_weight=w_tr)
        
        # Predict
        y_prob = best_pipe.predict_proba(X_va)[:,1]
        cv_roc.append(evaluate_prob(y_va, y_prob, w_va)["roc_auc"])
        cv_pr.append(evaluate_prob(y_va, y_prob, w_va)["pr_auc"])
    
    print(f"5-fold weighted ROC-AUC: {np.mean(cv_roc):.4f} ± {np.std(cv_roc):.4f}")
    print(f"5-fold weighted PR-AUC : {np.mean(cv_pr):.4f} ± {np.std(cv_pr):.4f}")
    
    return {
        "cv_roc_mean": np.mean(cv_roc),
        "cv_roc_std": np.std(cv_roc),
        "cv_pr_mean": np.mean(cv_pr),
        "cv_pr_std": np.std(cv_pr)
    }