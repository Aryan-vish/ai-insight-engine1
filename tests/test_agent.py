from src.insights.models import InsightCandidate, InsightCategory, PriorityLevel
from src.agent.decision_engine import InsightAgent


def test_agent_priority_ranking():
    agent = InsightAgent()

    high_candidate = InsightCandidate(
        id="c1",
        category=InsightCategory.ANOMALY,
        title="High Anomaly",
        description="Massive spike",
        magnitude=15.0,
        raw_metrics={"total_anomalies": 25, "total_flagged_amount": 100000.0}
    )

    low_candidate = InsightCandidate(
        id="c2",
        category=InsightCategory.SEGMENT,
        title="Low Deviation",
        description="Slight shift",
        magnitude=1.05,
        raw_metrics={"cluster_id": 1, "avg_spend": 105.0}
    )

    results = agent.process([low_candidate, high_candidate])

    assert len(results) >= 1
    # Higher score anomaly must appear first
    assert results[0].candidate.id == "c1"
    assert results[0].priority == PriorityLevel.CRITICAL
    assert "Audit flagged accounts" in results[0].recommended_action