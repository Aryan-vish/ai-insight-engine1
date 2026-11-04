from src.ml.clustering import run_customer_clustering
from src.ml.anomaly import run_anomaly_detection
from src.ml.trends import run_trend_analysis


def test_customer_clustering(mock_transaction_df):
    df_clustered = run_customer_clustering(mock_transaction_df, n_clusters=2)
    assert "cluster_id" in df_clustered.columns
    assert df_clustered["cluster_id"].nunique() <= 2


def test_anomaly_detection(mock_transaction_df):
    df_anom = run_anomaly_detection(mock_transaction_df, contamination=0.05)
    assert "is_anomaly" in df_anom.columns
    assert "anomaly_score" in df_anom.columns
    # Outliers inserted at idx 10 and 20 must be flagged
    assert df_anom.loc[10, "is_anomaly"] == 1
    assert df_anom.loc[20, "is_anomaly"] == 1


def test_trend_analysis(mock_transaction_df):
    trend_df = run_trend_analysis(mock_transaction_df)
    assert "rolling_mean" in trend_df.columns
    assert "z_score" in trend_df.columns
    assert "is_spike" in trend_df.columns