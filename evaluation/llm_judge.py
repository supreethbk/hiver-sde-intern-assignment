"""LLM-based response judge using the shared evaluation rubric."""

import re
import requests

from judge_rubric import CRITERIA, format_rubric, calculate_total

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"


def judge_reply(customer_message, generated_reply):
    """Judge one generated customer-support reply."""

    prompt = f"""
Evaluate this customer-support reply.

CUSTOMER:
{customer_message}

REPLY:
{generated_reply}

Use the following rubric:

{format_rubric()}

Give an overall quality score from 1 to 5:

1 = Very poor
2 = Poor
3 = Acceptable
4 = Good
5 = Excellent

Return ONLY this format:

RELEVANCE: number
USEFULNESS: number
GROUNDING: number
FACTUAL_SAFETY: number
LEAKAGE_SAFETY: number
STYLE: number
OVERALL_SCORE: number
REASON: one short sentence
"""

    for attempt in range(2):

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 120,
                },
            },
            timeout=120,
        )

        response.raise_for_status()

        text = response.json()["response"].strip()

        scores = {}

        for criterion in CRITERIA:
            pattern = rf"{criterion.upper()}\s*:\s*([012])"
            match = re.search(pattern, text, re.IGNORECASE)
            scores[criterion] = int(match.group(1)) if match else None

        overall_match = re.search(
            r"OVERALL_SCORE\s*:\s*([1-5])",
            text,
            re.IGNORECASE,
        )

        overall_score = (
            int(overall_match.group(1))
            if overall_match
            else None
        )

        reason_match = re.search(
            r"REASON\s*:\s*(.+)",
            text,
            re.IGNORECASE,
        )

        reason = (
            reason_match.group(1).strip()
            if reason_match
            else "No reason returned."
        )

        total_score = calculate_total(scores)

        valid_judgment = (
            total_score is not None
            and overall_score is not None
        )

        if valid_judgment:
            return {
                "relevance": scores["relevance"],
                "usefulness": scores["usefulness"],
                "grounding": scores["grounding"],
                "factual_safety": scores["factual_safety"],
                "leakage_safety": scores["leakage_safety"],
                "style": scores["style"],
                "overall_score": overall_score,
                "total_score": total_score,
                "valid_judgment": True,
                "reason": reason,
            }

    return {
        "relevance": None,
        "usefulness": None,
        "grounding": None,
        "factual_safety": None,
        "leakage_safety": None,
        "style": None,
        "overall_score": None,
        "total_score": None,
        "valid_judgment": False,
        "reason": "Invalid LLM judgment after retry.",
    }