import uuid
import pandas as pd
from typing import List
from src.insights.models import InsightCandidate, InsightCategory


def generate_candidates(
    enriched_df: pd.DataFrame,
    trend_df: pd.DataFrame
) -> List[InsightCandidate]:
    candidates: List[InsightCandidate] = []

    # 1. Anomaly Candidate Generation
    anomalies = enriched_df[enriched_df["is_anomaly"] == 1]
    anomaly_count = len(anomalies)
    if anomaly_count > 0:
        high_val_anomalies = anomalies[anomalies["amount"] > enriched_df["amount"].quantile(0.95)]
        candidates.append(
            InsightCandidate(
                id=str(uuid.uuid4())[:8],
                category=InsightCategory.ANOMALY,
                title="Elevated High-Value Anomalies Detected",
                description=f"Identified {anomaly_count} total anomalous transactions. {len(high_val_anomalies)} fall into the top 95th percentile of value.",
                magnitude=float(anomaly_count),
                raw_metrics={
                    "total_anomalies": int(anomaly_count),
                    "anomaly_rate": float(anomaly_count / len(enriched_df)),
                    "total_flagged_amount": float(anomalies["amount"].sum()),
                }
            )
        )

    # 2. Segment Candidate Generation
    if "cluster_id" in enriched_df.columns:
        cluster_stats = enriched_df.groupby("cluster_id")["amount"].mean()
        overall_mean = enriched_df["amount"].mean()

        for cluster_id, mean_val in cluster_stats.items():
            ratio = mean_val / overall_mean if overall_mean > 0 else 1.0
            if ratio >= 1.5 or ratio <= 0.5:
                direction = "higher" if ratio >= 1.5 else "lower"
                candidates.append(
                    InsightCandidate(
                        id=str(uuid.uuid4())[:8],
                        category=InsightCategory.SEGMENT,
                        title=f"Segment {cluster_id} Divergent Spend Pattern",
                        description=f"Cluster {cluster_id} average spend is {ratio:.2f}x {direction} than the global average.",
                        magnitude=float(ratio),
                        raw_metrics={
                            "cluster_id": int(cluster_id),
                            "avg_spend": float(mean_val),
                            "ratio_to_mean": float(ratio),
                        }
                    )
                )

    # 3. Trend Spike Candidate Generation
    spikes = trend_df[trend_df["is_spike"] == 1]
    for _, row in spikes.iterrows():
        candidates.append(
            InsightCandidate(
                id=str(uuid.uuid4())[:8],
                category=InsightCategory.TREND,
                title=f"Volume Surge on {row['date']}",
                description=f"Transaction volume surged past normal distribution with a Z-score of {row['z_score']:.2f}.",
                magnitude=float(row["z_score"]),
                raw_metrics={
                    "date": str(row["date"]),
                    "z_score": float(row["z_score"]),
                    "daily_amount": float(row["daily_amount"]),
                },
                detected_at=str(row["date"])
            )
        )

    return candidates