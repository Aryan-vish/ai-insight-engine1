from typing import List
from src.insights.models import InsightCandidate, PrioritizedInsight, PriorityLevel, InsightCategory
from src.agent.rules import (
    ANOMALY_CRITICAL_COUNT,
    ANOMALY_CRITICAL_AMOUNT,
    TREND_CRITICAL_Z,
    SEGMENT_HIGH_DEVIATION_RATIO,
    ACTION_MAP,
)


class InsightAgent:
    """Agentic decision layer to filter, score, prioritize, and suggest actions."""

    def evaluate_candidate(self, candidate: InsightCandidate) -> PrioritizedInsight:
        score = 0.0
        action = ACTION_MAP["default"]

        if candidate.category == InsightCategory.ANOMALY:
            metrics = candidate.raw_metrics
            count = metrics.get("total_anomalies", 0)
            amount = metrics.get("total_flagged_amount", 0.0)

            score += min(count * 5.0, 50.0)
            if amount > ANOMALY_CRITICAL_AMOUNT:
                score += 40.0

            if score >= 75.0 or count >= ANOMALY_CRITICAL_COUNT:
                priority = PriorityLevel.CRITICAL
                action = ACTION_MAP["anomaly_critical"]
            elif score >= 40.0:
                priority = PriorityLevel.MEDIUM
                action = ACTION_MAP["anomaly_medium"]
            else:
                priority = PriorityLevel.LOW

        elif candidate.category == InsightCategory.TREND:
            z = candidate.magnitude
            score = min(z * 25.0, 100.0)

            if z >= TREND_CRITICAL_Z:
                priority = PriorityLevel.CRITICAL
                action = ACTION_MAP["trend_spike"]
            elif z >= 2.0:
                priority = PriorityLevel.MEDIUM
                action = ACTION_MAP["trend_spike"]
            else:
                priority = PriorityLevel.LOW

        elif candidate.category == InsightCategory.SEGMENT:
            ratio = candidate.magnitude
            diff = abs(ratio - 1.0)
            score = min(diff * 40.0, 95.0)

            if ratio >= SEGMENT_HIGH_DEVIATION_RATIO:
                priority = PriorityLevel.CRITICAL
                action = ACTION_MAP["segment_high"]
            elif ratio >= 1.5:
                priority = PriorityLevel.MEDIUM
                action = ACTION_MAP["segment_high"]
            else:
                priority = PriorityLevel.LOW
                action = ACTION_MAP["segment_low"]
        else:
            priority = PriorityLevel.LOW

        # Noise filter: drop insights with very low scores
        should_report = score >= 30.0

        return PrioritizedInsight(
            candidate=candidate,
            score=round(score, 2),
            priority=priority,
            recommended_action=action,
            should_report=should_report
        )

    def process(self, candidates: List[InsightCandidate]) -> List[PrioritizedInsight]:
        prioritized = [self.evaluate_candidate(c) for c in candidates]
        filtered = [item for item in prioritized if item.should_report]
        # Sort descending by priority score
        filtered.sort(key=lambda x: x.score, reverse=True)
        return filtered