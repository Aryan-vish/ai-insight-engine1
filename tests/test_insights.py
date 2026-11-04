from src.ml.clustering import run_customer_clustering
from src.ml.anomaly import run_anomaly_detection
from src.ml.trends import run_trend_analysis
from src.insights.generator import generate_candidates
from src.insights.models import InsightCandidate


def test_generate_candidates(mock_transaction_df):
    clustered = run_customer_clustering(mock_transaction_df, n_clusters=2)
    anomalies = run_anomaly_detection(clustered, contamination=0.05)
    trends = run_trend_analysis(mock_transaction_df)

    candidates = generate_candidates(anomalies, trends)
    assert isinstance(candidates, list)
    assert len(candidates) > 0
    assert all(isinstance(c, InsightCandidate) for c in candidates)