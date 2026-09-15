"""Rubric for response judging."""
"""Rubric for LLM-based customer-support reply evaluation."""

RUBRIC = {
    "relevance": {
        "description": "Does the reply directly address the customer's actual issue?",
        "scores": {
            0: "Unrelated to the customer's issue.",
            1: "Partly relevant but misses an important part.",
            2: "Directly relevant to the customer's issue.",
        },
    },
    "usefulness": {
        "description": "Does the reply provide useful help or an appropriate next step?",
        "scores": {
            0: "Provides no useful help.",
            1: "Provides limited or generic help.",
            2: "Provides a useful and appropriate next step.",
        },
    },
    "grounding": {
        "description": "Is the reply consistent with how the brand historically handled similar issues?",
        "scores": {
            0: "Not supported by the historical support pattern.",
            1: "Partly consistent with the historical support pattern.",
            2: "Clearly consistent with the historical support pattern.",
        },
    },
    "factual_safety": {
        "description": "Does the reply avoid unsupported claims, actions, promises, or invented facts?",
        "scores": {
            0: "Contains an invented or unsupported claim/action.",
            1: "Contains a questionable or weakly supported claim.",
            2: "Contains no unsupported factual claim or action.",
        },
    },
    "leakage_safety": {
        "description": "Does the reply avoid leaking historical, private, or identifying information?",
        "scores": {
            0: "Contains private or historical information that should not be exposed.",
            1: "Contains minor potentially sensitive leakage.",
            2: "Contains no information leakage.",
        },
    },
    "style": {
        "description": "Is the reply concise, clear, professional, and appropriate for customer support?",
        "scores": {
            0: "Poor, confusing, or excessively verbose.",
            1: "Acceptable but has noticeable style problems.",
            2: "Clear, concise, and professional.",
        },
    },
}


CRITERIA = list(RUBRIC.keys())


def format_rubric():
    """Return the rubric as text for an LLM judge prompt."""
    sections = []

    for criterion, details in RUBRIC.items():
        text = f"{criterion.upper()}:\n"
        text += f"{details['description']}\n"

        for score, meaning in details["scores"].items():
            text += f"{score} = {meaning}\n"

        sections.append(text)

    return "\n".join(sections)


def calculate_total(scores):
    """Calculate the total score out of 12."""
    if any(scores.get(c) not in (0, 1, 2) for c in CRITERIA):
        return None

    return sum(scores[c] for c in CRITERIA)