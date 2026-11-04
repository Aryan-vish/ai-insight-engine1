from dataclasses import dataclass
from typing import List
import pandas as pd

from src.ml.clustering import run_customer_clustering
from src.ml.anomaly import run_anomaly_detection
from src.ml.trends import run_trend_analysis
from src.insights.generator import generate_candidates
from src.agent.decision_engine import InsightAgent
from src.insights.models import PrioritizedInsight
from src.genai.summarizer import generate_executive_summary


@dataclass
class PipelineResult:
    enriched_df: pd.DataFrame
    trend_df: pd.DataFrame
    insights: List[PrioritizedInsight]
    ai_summary: str


def run_pipeline(df: pd.DataFrame, generate_llm: bool = True) -> PipelineResult:
    # Phase 1: ML Layers
    clustered_df = run_customer_clustering(df)
    anomaly_df = run_anomaly_detection(clustered_df)
    trend_df = run_trend_analysis(df)

    # Phase 2: Candidate Generation
    candidates = generate_candidates(anomaly_df, trend_df)

    # Phase 3: Agent Prioritization
    agent = InsightAgent()
    prioritized_insights = agent.process(candidates)

    # Phase 4: GenAI Narrative
    ai_summary = ""
    if generate_llm and prioritized_insights:
        ai_summary = generate_executive_summary(prioritized_insights)
    elif not prioritized_insights:
        ai_summary = "All transaction metrics fall within normal behavioral baselines."

    return PipelineResult(
        enriched_df=anomaly_df,
        trend_df=trend_df,
        insights=prioritized_insights,
        ai_summary=ai_summary
    )