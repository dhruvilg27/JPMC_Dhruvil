import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

from src.config import RANDOM_STATE, MIN_CLUSTERS, MAX_CLUSTERS

def try_kmeans_weighted(X_enc, sample_weight, k_list=None, random_state=RANDOM_STATE):
    """Try different numbers of clusters and return the best model."""
    if k_list is None:
        k_list = range(MIN_CLUSTERS, MAX_CLUSTERS + 1)
    
    best_k, best_inertia = None, float("inf")
    best_model = None
    
    for k in k_list:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        km.fit(X_enc, sample_weight=sample_weight)
        if km.inertia_ < best_inertia:
            best_inertia = km.inertia_
            best_k, best_model = k, km
    
    return best_k, best_model

def weighted_profile(X_raw, y, w, clusters, cat_cols, num_cols, top_cats=8):
    """Create a weighted profile of the clusters."""
    prof = []
    df_local = X_raw.copy()
    df_local["_y"] = y
    df_local["_w"] = w
    df_local["_cl"] = clusters

    for k in sorted(np.unique(clusters)):
        sub = df_local[df_local["_cl"] == k]
        wsum = sub["_w"].sum()
        pos_rate = np.average(sub["_y"], weights=sub["_w"])
        row = {
            "cluster": int(k),
            "rows": int(len(sub)),
            "weight_sum": float(wsum),
            "pos_rate": float(pos_rate)
        }
        
        # Top categorical shares
        for c in cat_cols[:]:
            sv = sub.groupby(c)["_w"].sum().sort_values(ascending=False)
            sv = (sv / sv.sum()).head(top_cats)
            row[f"top_{c}"] = "; ".join([f"{idx}:{pct:.1%}" for idx, pct in sv.items()])
        
        # Numeric weighted medians/means
        for c in num_cols[:]:
            vals = sub[c]
            try:
                wmean = np.average(vals.fillna(vals.median()), weights=sub["_w"])
            except Exception:
                wmean = float("nan")
            row[f"mean_{c}"] = float(wmean)
        
        prof.append(row)
    
    return pd.DataFrame(prof)

def analyze_cluster_predictions(train_clusters, X_train, best_pipe, y_train, w_train):
    """Analyze prediction probabilities by cluster."""
    df_prob = pd.DataFrame({
        "cluster": train_clusters,
        "prob": best_pipe.predict_proba(X_train)[:,1],
        "y": y_train,
        "w": w_train
    })
    
    agg = (df_prob
           .groupby("cluster")
           .apply(lambda g: pd.Series({
               "w_mean_prob": np.average(g["prob"], weights=g["w"]),
               "w_pos_rate": np.average(g["y"], weights=g["w"]),
               "w": g["w"].sum(),
               "n": len(g)
           }))
           .sort_values("w_mean_prob", ascending=False))
    
    return agg

def project_to_2d(X_enc, train_clusters, max_samples=12000):
    """Project high-dimensional data to 2D using TruncatedSVD."""
    # Ensure sparse format
    if not sparse.issparse(X_enc):
        X_enc = sparse.csr_matrix(X_enc)
    
    # 2D projection
    svd2 = TruncatedSVD(n_components=2, random_state=RANDOM_STATE)
    Z = svd2.fit_transform(X_enc)
    
    # Sample for visualization if needed
    if Z.shape[0] > max_samples:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(Z.shape[0], size=max_samples, replace=False)
        Z = Z[idx]
        clusters = np.array(train_clusters)[idx]
    else:
        clusters = train_clusters
    
    return Z, clusters, svd2

def get_cluster_lift(cluster_profile, y_train, w_train):
    """Calculate lift metrics for clusters."""
    overall_pos = np.average(y_train, weights=w_train)
    
    lift_tbl = (cluster_profile
                .assign(
                    lift=lambda d: d["pos_rate"] / overall_pos,
                    weight_share=lambda d: d["weight_sum"]/d["weight_sum"].sum()
                )
                .sort_values("lift", ascending=False))
    
    return overall_pos, lift_tbl[["cluster", "rows", "weight_sum", "weight_share", "pos_rate", "lift"]]