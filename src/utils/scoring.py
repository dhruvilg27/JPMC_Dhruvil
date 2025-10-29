import pickle
import pandas as pd

def score_and_decide(df_new: pd.DataFrame, model_artifact_path="final_model.pkl"):
    """
    Score new data using the trained model.
    
    Parameters:
    -----------
    df_new : pd.DataFrame
        New data to score, with same columns as training data
    model_artifact_path : str
        Path to the saved model artifact
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with probability and binary predictions
    """
    # Load model artifacts
    with open(model_artifact_path, "rb") as f:
        obj = pickle.load(f)
    
    pipe = obj["model"]
    thr = obj["threshold"]
    
    # Generate predictions
    probs = pipe.predict_proba(df_new)[:,1]
    preds = (probs >= thr).astype(int)
    
    return pd.DataFrame({
        "probability": probs,
        "prediction": preds
    })