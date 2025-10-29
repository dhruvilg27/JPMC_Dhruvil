import re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.impute import SimpleImputer

from src.config import RARE_THRESH

def load_data(data_file, cols_file):
    """Load and prepare the census data."""
    # Load columns
    cols = pd.read_csv(cols_file, header=None, names=["col"])["col"].str.strip().tolist()
    
    # Load data
    df = pd.read_csv(data_file, header=None, names=cols)
    return df

def process_target(df, target_map):
    """Process target variable with given mapping."""
    # Normalize target map keys
    _target_map_norm = {_norm_key(k): v for k, v in target_map.items()}
    
    # Map target values
    df["label_bin"] = df["label"].map(lambda x: _target_map_norm.get(_norm_key(x), np.nan)).astype("float")
    
    # Handle weights
    weight_col_candidates = [c for c in df.columns if "weight" in c.lower() or "wt" in c.lower()]
    weight_col = weight_col_candidates[0] if weight_col_candidates else None
    if weight_col is None:
        df["weight"] = 1.0
        weight_col = "weight"
    
    # Filter valid rows
    df = df[df["label_bin"].isin([0.0, 1.0])].copy()
    df = df[df[weight_col].fillna(0) > 0].copy()
    
    return df, weight_col

def strip_categories(df):
    """Clean categorical values by stripping whitespace."""
    def strip_cat(val):
        if pd.isna(val): return val
        return re.sub(r"\s+", " ", str(val)).strip()
    
    for c in df.select_dtypes(include="object").columns:
        if c not in ["label"]:
            df[c] = df[c].map(strip_cat)
    return df

def get_feature_columns(df, label_col="label_bin", weight_col="weight"):
    """Extract feature columns and identify their types."""
    # Drop target and weight columns
    X = df.drop(columns=[label_col]).copy()
    
    # Drop target-like columns
    suspect_regex = re.compile(r"(?i)\b(label|income|earn|target|>50|<=50|50k)\b")
    leak_cols = [c for c in X.columns if suspect_regex.search(c)]
    if leak_cols:
        print("Dropping suspect columns (possible leakage):", leak_cols)
        X = X.drop(columns=leak_cols)
    
    # Drop weight column if present
    if weight_col in X.columns:
        X = X.drop(columns=[weight_col])
    
    # Identify column types
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    
    return X, cat_cols, num_cols

def handle_rare_categories(X_train, X_test, cat_cols, w_train, thresh=RARE_THRESH):
    """Handle rare categories in categorical features."""
    def rare_mapper(series, weights, thresh=thresh):
        s = series.fillna("MISSING").astype(str)
        # weighted frequencies
        wf = pd.Series(weights, index=s.index).groupby(s).sum()
        wf = wf / wf.sum()
        rare_vals = set(wf[wf < thresh].index.tolist())
        return s.where(~s.isin(rare_vals), other="__RARE__")
    
    X_train_cat = X_train[cat_cols].copy()
    X_test_cat = X_test[cat_cols].copy()
    
    for c in cat_cols:
        X_train_cat[c] = rare_mapper(X_train_cat[c], w_train)
        # map test: unseen -> __RARE__
        tr_vals = set(X_train_cat[c].dropna().unique())
        X_test_cat[c] = X_test_cat[c].fillna("MISSING").astype(str)
        X_test_cat[c] = X_test_cat[c].where(X_test_cat[c].isin(tr_vals), other="__RARE__")
    
    return X_train_cat, X_test_cat

def create_preprocessor(num_cols, cat_cols):
    """Create preprocessing pipeline for numerical and categorical features."""
    # Numeric pipeline
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("yeojohnson", PowerTransformer(method="yeo-johnson", standardize=True)),
    ])
    
    # Categorical pipeline
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])
    
    # Combined preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols)
        ],
        remainder="drop"
    )
    
    return preprocessor

def _norm_key(s):
    """Normalize string keys for consistent mapping."""
    s = "" if pd.isna(s) else str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s