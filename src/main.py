import pickle
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import (
    RANDOM_STATE, RAW_DATA_FILE, RAW_COLS_FILE,
    TARGET_MIN_RECALL
)
from src.preprocessing.data_processor import (
    load_data, process_target, strip_categories,
    get_feature_columns, handle_rare_categories,
    create_preprocessor
)
from src.models.classification import (
    create_models, train_and_evaluate_models,
    calibrate_models, cross_validate_best_model
)
from src.models.segmentation import (
    try_kmeans_weighted, weighted_profile,
    analyze_cluster_predictions, project_to_2d,
    get_cluster_lift
)
from src.utils.metrics import get_feature_names, threshold_report

def main():
    # Set random seed
    np.random.seed(RANDOM_STATE)
    
    # 1. Load and prepare data
    print("Loading data...")
    df = load_data(RAW_DATA_FILE, RAW_COLS_FILE)
    
    # 2. Process target variable
    target_map = {'- 50000.': 0, ' - 50000.': 0, '50000+.': 1, ' 50000+.': 1}
    df, weight_col = process_target(df, target_map)
    
    # 3. Clean categorical variables
    df = strip_categories(df)
    
    # 4. Get feature columns
    X, cat_cols, num_cols = get_feature_columns(df, "label_bin", weight_col)
    y = df["label_bin"].astype(int).values
    w = df[weight_col].astype(float).values
    
    # 5. Split data
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    
    # 6. Handle rare categories
    X_train_cat, X_test_cat = handle_rare_categories(X_train, X_test, cat_cols, w_train)
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[cat_cols] = X_train_cat
    X_test[cat_cols] = X_test_cat
    
    # 7. Create preprocessor
    print("\nCreating preprocessor...")
    preprocessor = create_preprocessor(num_cols, cat_cols)
    
    # 8. Train and evaluate models
    print("\nTraining models...")
    models = create_models(preprocessor)
    
    # Start with just one model for faster execution
    results_sorted = train_and_evaluate_models([models[0]], X_train, y_train, X_test, y_test, w_train, w_test)  # Only run logistic regression first
    
    # 9. Calibrate the model
    print("\nCalibrating model...")
    calibrated_sorted = calibrate_models(results_sorted, X_train, y_train, X_test, y_test, w_train, w_test)
    
    # 10. Cross-validate best model
    print("\nCross-validating best model...")
    best_name, best_pipe = calibrated_sorted[0][:2]
    cv_metrics = cross_validate_best_model(best_pipe, X_train, y_train, w_train)
    
    # Create output directories if they don't exist
    output_dir = Path("outputs")
    models_dir = output_dir / "models"
    results_dir = output_dir / "results"
    output_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    # Save best model
    print("\nSaving outputs...")
    feat_names = get_feature_names(preprocessor, num_cols, cat_cols)
    y_prob_test = best_pipe.predict_proba(X_test)[:,1]
    best_thr = threshold_report(y_test, y_prob_test, w_test, target="f1", min_recall=TARGET_MIN_RECALL)
    
    # Save model and related info
    with open(models_dir / "final_model.pkl", "wb") as f:
        pickle.dump({
            "model": best_pipe,
            "threshold": best_thr["thr"],
            "feature_names": feat_names.tolist()
        }, f)
        
    # Save performance metrics
    performance_metrics = {
        "classification_metrics": {
            "roc_auc": cv_metrics["cv_roc_mean"],
            "roc_auc_std": cv_metrics["cv_roc_std"],
            "pr_auc": cv_metrics["cv_pr_mean"],
            "pr_auc_std": cv_metrics["cv_pr_std"],
            "best_threshold": best_thr
        }
    }
    
    with open(results_dir / "model_performance.json", "w") as f:
        json.dump(performance_metrics, f, indent=4)
    
    # 12. Segmentation
    print("\nPerforming segmentation analysis...")
    X_train_enc = preprocessor.fit_transform(X_train)
    best_k, km_model = try_kmeans_weighted(X_train_enc, w_train)
    train_clusters = km_model.predict(X_train_enc)
    
    # 13. Analyze clusters
    cluster_profile = weighted_profile(X_train, y_train, w_train, train_clusters, cat_cols, num_cols)
    cluster_predictions = analyze_cluster_predictions(train_clusters, X_train, best_pipe, y_train, w_train)
    overall_pos, lift_table = get_cluster_lift(cluster_profile, y_train, w_train)
    
    print(f"\nNumber of clusters: {best_k}")
    print(f"Overall positive rate (weighted): {overall_pos:.4f}")
    print("\nCluster lift table:")
    print(lift_table)
    
    # Save segmentation results
    cluster_results = {
        "n_clusters": best_k,
        "overall_positive_rate": float(overall_pos)
    }
    lift_table.to_csv(results_dir / "cluster_lift_table.csv", index=False)
    cluster_profile.to_csv(results_dir / "cluster_profiles.csv", index=False)
    
    with open(results_dir / "segmentation_results.json", "w") as f:
        json.dump(cluster_results, f, indent=4)
    
    print("\nOutputs saved to:")
    print(f"- Model: {models_dir / 'final_model.pkl'}")
    print(f"- Performance metrics: {results_dir / 'model_performance.json'}")
    print(f"- Cluster lift table: {results_dir / 'cluster_lift_table.csv'}")
    print(f"- Cluster profiles: {results_dir / 'cluster_profiles.csv'}")
    print(f"- Segmentation results: {results_dir / 'segmentation_results.json'}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()