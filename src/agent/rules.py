ANOMALY_CRITICAL_COUNT = 10
ANOMALY_CRITICAL_AMOUNT = 50000.0
TREND_CRITICAL_Z = 3.0
SEGMENT_HIGH_DEVIATION_RATIO = 2.5

ACTION_MAP = {
    "anomaly_critical": "Audit flagged accounts immediately and inspect gateway fraud filters.",
    "anomaly_medium": "Queue flagged transactions for tier-2 risk review.",
    "segment_high": "Alert marketing team to design personalized retention campaigns for high-spenders.",
    "segment_low": "Review churn prevention incentives and product re-engagement flows.",
    "trend_spike": "Verify server uptime capacity and align inventory stock with marketing promotions.",
    "default": "Monitor metrics during next daily batch processing.",
}