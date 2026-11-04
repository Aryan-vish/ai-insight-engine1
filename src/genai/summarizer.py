import json
from typing import List
from src.insights.models import PrioritizedInsight
from src.genai.client import get_genai_client
from src.genai.prompts import EXECUTIVE_SYSTEM_PROMPT, SUMMARY_USER_TEMPLATE


def generate_executive_summary(prioritized_insights: List[PrioritizedInsight]) -> str:
    if not prioritized_insights:
        return "No high-priority insights detected during this cycle. Operations are within nominal parameters."

    # Convert dataclasses to structured payload for LLM
    payload = []
    for item in prioritized_insights:
        payload.append({
            "title": item.candidate.title,
            "category": item.candidate.category.value,
            "score": item.score,
            "priority": item.priority.value,
            "summary": item.candidate.description,
            "suggested_action": item.recommended_action
        })

    prompt_content = SUMMARY_USER_TEMPLATE.format(insights_json=json.dumps(payload, indent=2))

    try:
        client = get_genai_client()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_content,
            config={
                "system_instruction": EXECUTIVE_SYSTEM_PROMPT,
                "temperature": 0.3
            }
        )
        return response.text
    except Exception as e:
        return f"GenAI generation fallback (API Error: {str(e)}). Primary findings: " + "; ".join(
            [f"[{i.priority.value}] {i.candidate.title}" for i in prioritized_insights[:3]]
        )