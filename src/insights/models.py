from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class InsightCategory(str, Enum):
    ANOMALY = "anomaly"
    SEGMENT = "segment"
    TREND = "trend"


class PriorityLevel(str, Enum):
    CRITICAL = "CRITICAL"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class InsightCandidate(BaseModel):
    id: str
    category: InsightCategory
    title: str
    description: str
    magnitude: float
    raw_metrics: Dict[str, Any] = Field(default_factory=dict)
    detected_at: Optional[str] = None


class PrioritizedInsight(BaseModel):
    candidate: InsightCandidate
    score: float
    priority: PriorityLevel
    recommended_action: str
    should_report: bool = True