import pandas as pd
from sklearn.ensemble import IsolationForest


def run_anomaly_detection(df: pd.DataFrame, contamination: float = 0.03, random_state: int = 42) -> pd.DataFrame:
    """Flags anomalous transactions based on amount and frequency patterns."""
    data = df.copy()

    features = data[["amount"]].copy()

    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100
    )
    # -1 represents outlier, 1 represents inlier
    preds = iso_forest.fit_predict(features)
    scores = iso_forest.decision_function(features)

    data["is_anomaly"] = (preds == -1).astype(int)
    data["anomaly_score"] = -scores  # Higher score = more anomalous
    return data