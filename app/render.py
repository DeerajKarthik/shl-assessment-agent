from __future__ import annotations

import re

from app.catalog import CatalogItem

_URL_RE = re.compile(r"https?://\S+")


def refusal_reply(kind: str) -> str:
    if kind == "legal":
        return (
            "I can explain what an SHL assessment measures, but I cannot determine "
            "legal obligations or whether a test satisfies a regulatory requirement. "
            "Please consult qualified legal or compliance counsel."
        )
    if kind == "injection":
        return (
            "I can only help select and compare grounded SHL catalog assessments, "
            "so I cannot follow instructions that bypass those constraints."
        )
    return (
        "I can help with SHL assessment recommendations, comparisons, and explanations, "
        "but I cannot help with unrelated questions in this conversation."
    )


def recommendation_reply(items: list[CatalogItem], confirmed: bool = False) -> str:
    type_codes: list[str] = []
    for item in items:
        for code in item.test_type.split(","):
            if code not in type_codes:
                type_codes.append(code)
    coverage = _coverage_phrase(type_codes)
    prefix = "Confirmed." if confirmed else "Here is a grounded shortlist."
    return sanitize_reply(f"{prefix} The selections cover {coverage} for the requirements provided.")


def _coverage_phrase(codes: list[str]) -> str:
    names = {
        "A": "ability and aptitude",
        "B": "situational judgment",
        "C": "competencies",
        "D": "development reporting",
        "E": "assessment exercises",
        "K": "knowledge and skills",
        "P": "personality and behavior",
        "S": "job simulations",
    }
    values = [names[code] for code in codes if code in names]
    if not values:
        return "the requested assessment needs"
    if len(values) == 1:
        return values[0]
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def comparison_reply(items: list[CatalogItem]) -> str:
    if not items:
        return (
            "I could not uniquely resolve the assessment name in the catalog. "
            "Please provide the exact catalog name."
        )
    sentences = []
    for item in items:
        levels = ", ".join(item.job_levels[:3]) or "not specified"
        languages = ", ".join(item.languages[:3]) or "not specified"
        duration = item.duration or "not specified"
        sentences.append(
            f"{item.name} is type {item.test_type}; its catalog duration is {duration}, "
            f"listed levels include {levels}, and listed languages include {languages}. "
            f"The catalog describes it as: {item.description}"
        )
    return sanitize_reply(" ".join(sentences))



def sanitize_reply(text: str) -> str:
    """Remove any http/https URLs from reply text.

    The assignment requires URLs appear only in recommendation objects,
    never in the reply string.
    """
    return _URL_RE.sub("", text).strip()
