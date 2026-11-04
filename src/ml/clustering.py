import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def run_customer_clustering(df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> pd.DataFrame:
    """Computes customer spend & frequency features and tags clusters."""
    data = df.copy()

    # Per-customer aggregation
    cust_summary = data.groupby("customer_id").agg(
        total_spend=("amount", "sum"),
        txn_count=("transaction_id", "count"),
        avg_spend=("amount", "mean"),
    ).reset_index()

    feature_cols = ["total_spend", "txn_count", "avg_spend"]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cust_summary[feature_cols])

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    cust_summary["cluster_id"] = kmeans.fit_predict(scaled_features)

    # Map back to main transaction dataframe
    merged = data.merge(cust_summary[["customer_id", "cluster_id"]], on="customer_id", how="left")
    return merged