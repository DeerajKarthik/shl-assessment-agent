from __future__ import annotations

from dataclasses import dataclass, field
import re

from app.catalog import Catalog, normalize_text
from app.schemas import Message


TYPE_TERMS = {
    "A": ("cognitive", "ability", "aptitude", "reasoning", "numerical"),
    "B": ("situational judgement", "situational judgment", "sjt", "biodata"),
    "C": ("competency", "competencies"),
    "D": ("development report", "development and 360", "360"),
    "E": ("assessment exercise", "assessment centre", "assessment center"),
    "K": ("knowledge", "technical test", "skills test"),
    "P": ("personality", "behaviour", "behavior", "work style"),
    "S": ("simulation", "hands on", "hands-on"),
}

TYPE_CODE_TO_NAME = {
    "A": "cognitive/ability",
    "B": "situational judgement",
    "C": "competency",
    "D": "development",
    "E": "assessment exercise",
    "K": "knowledge/skills",
    "P": "personality/behavior",
    "S": "simulation",
}

CONFIRMATION_RE = re.compile(
    r"\b(thanks|thank you|that works|looks good|that(?:'s| is) good|perfect|clear|"
    r"confirmed|lock(?:ing)? it in|that covers it|keeping the|we(?:'ll| will) use|"
    r"keep (?:it|the shortlist|the list) as[- ]is|final list)\b",
    re.I,
)
LEGAL_RE = re.compile(
    r"\b(legally|required by law|legal requirement|legal advice|lawful|law 144|"
    r"satisf(?:y|ies) (?:hipaa|gdpr|a legal)|regulatory obligation|compliance counsel)\b",
    re.I,
)
OFF_TOPIC_RE = re.compile(
    r"\b(write an? offer letter|salary negotiation|compensation package|visa advice|"
    r"immigration advice|how (?:do|should) i fire|terminate an employee|onboarding plan|"
    r"medical advice|diagnos(?:e|is)|prescri(?:be|ption)|headache|treatment plan|"
    r"weather|news|sports|politics|how to cook|recipe|where to travel|buy a car|date|time|joke)\b",
    re.I,
)

# Patterns that indicate prompt injection — but only when NOT inside quotes
_RAW_INJECTION_PATTERNS = [
    r"ignore (?:all |your |the )?(?:previous|prior|system) (?:instructions|limitations)",
    r"reveal (?:your |the )?(?:system prompt|instructions)",
    r"developer message",
    r"pretend you are unrestricted",
    r"bypass (?:the )?catalog",
    r"jailbreak",
    r"what are your (?:prior )?instructions",
    r"output (?:your |the )?(?:system|internal) (?:prompt|instructions)",
]
INJECTION_RE = re.compile(
    r"\b(" + "|".join(_RAW_INJECTION_PATTERNS) + r")\b",
    re.I,
)
EXPLAIN_RE = re.compile(r"\b(compare|difference between|different from|versus|\bvs\.?\b|why|what does|explain)\b", re.I)

# Language normalization
LANGUAGE_MAP = {
    "english": "English",
    "english usa": "English (USA)",
    "english us": "English (USA)",
    "american english": "English (USA)",
    "english international": "English International",
    "british english": "English International",
    "spanish": "Spanish",
    "latin american spanish": "Latin American Spanish",
    "french": "French",
    "german": "German",
    "dutch": "Dutch",
    "portuguese": "Portuguese",
    "italian": "Italian",
    "japanese": "Japanese",
    "chinese": "Chinese Simplified",
    "mandarin": "Chinese Simplified",
    "korean": "Korean",
    "arabic": "Arabic",
    "turkish": "Turkish",
    "russian": "Russian",
    "swedish": "Swedish",
    "norwegian": "Norwegian",
    "danish": "Danish",
    "finnish": "Finnish",
    "polish": "Polish",
    "czech": "Czech",
    "hungarian": "Hungarian",
    "thai": "Thai",
    "vietnamese": "Vietnamese",
    "hindi": "Hindi",
    "indonesian": "Indonesian",
    "malay": "Malay",
}

# Duration parsing
DURATION_RE = re.compile(
    r"\b(?:under|below|less than|max(?:imum)?|at most|no (?:more|longer) than)\s+(\d+)\s*(?:min(?:ute)?s?|m)\b",
    re.I,
)


@dataclass(slots=True)
class ConversationState:
    existing_messages: int
    user_texts: list[str]
    assistant_texts: list[str]
    last_user_text: str
    combined_user_text: str
    normalized_text: str
    requested_types: set[str] = field(default_factory=set)
    excluded_types: set[str] = field(default_factory=set)
    included_ids: list[str] = field(default_factory=list)
    excluded_ids: set[str] = field(default_factory=set)
    excluded_themes: set[str] = field(default_factory=set)
    required_languages: list[str] = field(default_factory=list)
    max_duration_minutes: int | None = None
    confirmation: bool = False
    legal: bool = False
    off_topic: bool = False
    injection: bool = False
    comparison: bool = False
    terminal_response: bool = False
    clarification_count: int = 0

    def current_constraints(self) -> str:
        parts = []
        if self.requested_types:
            parts.append(f"Requested types: {', '.join(sorted(self.requested_types))}")
        if self.excluded_types:
            parts.append(f"Excluded types: {', '.join(sorted(self.excluded_types))}")
        if self.required_languages:
            parts.append(f"Required languages: {', '.join(self.required_languages)}")
        if self.max_duration_minutes:
            parts.append(f"Max duration: {self.max_duration_minutes}m")
        return " | ".join(parts) if parts else "No explicit constraints."


def _strip_quoted_text(text: str) -> str:
    """Remove text inside quotation marks to avoid false-positive injection detection
    on legitimate JDs that quote suspicious phrases."""
    # Remove single-quoted, double-quoted, and smart-quoted text
    text = re.sub(r"'[^']*'", " ", text)
    text = re.sub(r'"[^"]*"', " ", text)
    text = re.sub(r"\u2018[^\u2019]*\u2019", " ", text)
    text = re.sub(r"\u201c[^\u201d]*\u201d", " ", text)
    return text


def _detect_injection(text: str) -> bool:
    """Detect injection attempts, ignoring text inside quotes."""
    unquoted = _strip_quoted_text(text)
    return bool(INJECTION_RE.search(unquoted))


def build_state(
    messages: list[Message], catalog: Catalog, aliases: dict[str, str]
) -> ConversationState:
    user_texts = [message.content for message in messages if message.role == "user"]
    
    # Process explicit replacements to fix retrieval contamination
    for i in range(len(user_texts)):
        replacements = re.findall(r"\breplace\s+([\w\s+.-]+?)\s+with\s+([\w\s+.-]+)", user_texts[i], flags=re.I)
        for old_str, new_str in replacements:
            old_str, new_str = old_str.strip(), new_str.strip()
            if old_str and new_str:
                for j in range(i):
                    pattern = r"(?i)\b" + re.escape(old_str) + r"\b"
                    user_texts[j] = re.sub(pattern, new_str, user_texts[j])
                    
    assistant_texts = [message.content for message in messages if message.role == "assistant"]
    last = user_texts[-1]
    is_confirmation = bool(CONFIRMATION_RE.search(last))
    
    # If the user is just confirming, strip their confirmation text from the semantic query 
    # so we get the exact same embedding, candidate pool, and LLM reranking result.
    if is_confirmation and len(user_texts) > 1:
        combined = "\n".join(user_texts[:-1])
    else:
        combined = "\n".join(user_texts)
        
    normalized = normalize_text(combined)
    state = ConversationState(
        existing_messages=len(messages),
        user_texts=user_texts,
        assistant_texts=assistant_texts,
        last_user_text=last,
        combined_user_text=combined,
        normalized_text=normalized,
        confirmation=is_confirmation,
        legal=bool(LEGAL_RE.search(last)),
        off_topic=bool(OFF_TOPIC_RE.search(last)),
        injection=_detect_injection(last),
        comparison=bool(EXPLAIN_RE.search(last)),
        terminal_response=len(messages) == 7,
        clarification_count=sum(text.rstrip().endswith("?") for text in assistant_texts),
    )

    for code, terms in TYPE_TERMS.items():
        if any(term in normalized for term in terms):
            state.requested_types.add(code)

    _extract_type_constraints(state)
    _extract_language_constraints(state)
    _extract_duration_constraints(state)
    _extract_edits(state, catalog, aliases)
    return state


def _extract_type_constraints(state: ConversationState) -> None:
    """Parse explicit type exclusions like 'no simulations', 'exclude cognitive',
    'knowledge only', 'only technical tests'."""
    text = state.normalized_text

    # "exclude X" / "no X" / "without X" type constraints
    exclusion_patterns = [
        (r"\b(?:exclude|no|without|remove|drop|not?)\s+(?:any\s+)?(?:cognitive|ability|aptitude)\b", "A"),
        (r"\b(?:exclude|no|without|remove|drop|not?)\s+(?:any\s+)?(?:personality|behavior|behaviour)\b", "P"),
        (r"\b(?:exclude|no|without|remove|drop|not?)\s+(?:any\s+)?simulation", "S"),
        (r"\b(?:exclude|no|without|remove|drop|not?)\s+(?:any\s+)?competenc", "C"),
        (r"\b(?:exclude|no|without|remove|drop|not?)\s+(?:any\s+)?(?:knowledge|technical)", "K"),
        (r"\b(?:exclude|no|without|remove|drop|not?)\s+(?:any\s+)?(?:situational|sjt|biodata)", "B"),
        (r"\bdo not include\s+(?:any\s+)?(?:cognitive|ability)\b", "A"),
        (r"\bdo not include\s+(?:any\s+)?(?:personality|behavior)\b", "P"),
        (r"\bdon.t include\s+(?:any\s+)?(?:cognitive|ability)\b", "A"),
        (r"\bdon.t include\s+(?:any\s+)?(?:personality|behavior)\b", "P"),
    ]
    for pattern, code in exclusion_patterns:
        if re.search(pattern, text, re.I):
            state.excluded_types.add(code)

    # "only X" / "X only" constraints
    only_patterns = [
        (r"\b(?:only|exclusively)\s+(?:knowledge|technical)\b", {"K"}),
        (r"\bknowledge\s+only\b", {"K"}),
        (r"\btechnical\s+only\b", {"K"}),
        (r"\b(?:only|exclusively)\s+cognitive\b", {"A"}),
        (r"\bcognitive\s+only\b", {"A"}),
        (r"\b(?:only|exclusively)\s+(?:personality|behavior)\b", {"P"}),
        (r"\b(?:keep|want)\s+only\s+technical\b", {"K"}),
    ]
    for pattern, allowed_codes in only_patterns:
        if re.search(pattern, text, re.I):
            all_codes = set(TYPE_TERMS.keys())
            state.excluded_types |= (all_codes - allowed_codes)


def _extract_language_constraints(state: ConversationState) -> None:
    """Parse explicit language requirements."""
    text = state.normalized_text

    # Check for "X only" language pattern
    only_match = re.search(
        r"\b(\w+)\s+only\b", text, re.I
    )

    detected_languages: list[str] = []
    for keyword, canonical in LANGUAGE_MAP.items():
        if keyword in text:
            if canonical not in detected_languages:
                detected_languages.append(canonical)

    # If user said "<language> only", enforce strictly
    if only_match:
        lang_word = only_match.group(1).lower()
        if lang_word in LANGUAGE_MAP:
            state.required_languages = [LANGUAGE_MAP[lang_word]]
            return

    state.required_languages = detected_languages


def _extract_duration_constraints(state: ConversationState) -> None:
    """Parse duration constraints like 'under 15 minutes'."""
    text = state.combined_user_text
    match = DURATION_RE.search(text)
    if match:
        state.max_duration_minutes = int(match.group(1))


def _parse_duration_minutes(duration_str: str) -> int | None:
    """Parse a catalog duration string into minutes. Returns None if unparseable."""
    if not duration_str or duration_str.lower() in ("", "not specified", "n/a", "varies"):
        return None
    # Common formats: "30 minutes", "1 hour", "45", "15-20 minutes"
    # Take the maximum in a range
    numbers = re.findall(r"(\d+)", duration_str)
    if not numbers:
        return None
    max_val = max(int(n) for n in numbers)
    if "hour" in duration_str.lower():
        return max_val * 60
    return max_val


def _extract_edits(
    state: ConversationState, catalog: Catalog, aliases: dict[str, str]
) -> None:
    include_patterns = (
        "add ",
        "include ",
        "also need ",
        "also add ",
        "keep ",
        "final list",
        "swap in ",
        "switch to ",
        "instead use ",
        "replace with ",
    )
    exclude_patterns = (
        "drop ",
        "remove ",
        "exclude ",
        "without ",
        "do not include ",
        "don't include ",
        "swap out ",
        "instead of ",
        "replace ",
        "switch from ",
    )

    for text in state.user_texts:
        normalized = normalize_text(text)
        for alias, entity_id in aliases.items():
            if not _word_boundary_match(alias, normalized):
                continue
            alias_position = normalized.find(alias)
            before = normalized[:alias_position]
            last_include = max(
                (before.rfind(marker.strip()) for marker in include_patterns), default=-1
            )
            last_exclude = max(
                (before.rfind(marker.strip()) for marker in exclude_patterns), default=-1
            )
            if last_exclude > last_include and alias_position - last_exclude <= 35:
                state.excluded_ids.add(entity_id)
                if entity_id in state.included_ids:
                    state.included_ids.remove(entity_id)
            elif last_include >= 0 and alias_position - last_include <= 35:
                if entity_id not in state.included_ids:
                    state.included_ids.append(entity_id)
                state.excluded_ids.discard(entity_id)

        for item in catalog.items:
            if item.normalized_name not in normalized:
                continue
            position = normalized.find(item.normalized_name)
            before = normalized[:position]
            last_include = max(
                (before.rfind(marker.strip()) for marker in include_patterns), default=-1
            )
            last_exclude = max(
                (before.rfind(marker.strip()) for marker in exclude_patterns), default=-1
            )
            if last_exclude > last_include and position - last_exclude <= 50:
                state.excluded_ids.add(item.entity_id)
                if item.entity_id in state.included_ids:
                    state.included_ids.remove(item.entity_id)

        # Explicit replace parsing
        replaced_in_themes = set()
        replace_match = re.search(r"\breplace\s+([\w\s+.-]+?)\s+with\s+([\w\s+.-]+)", normalized)
        if replace_match:
            old_theme = replace_match.group(1).strip()
            new_theme = replace_match.group(2).strip()
            if old_theme:
                state.excluded_themes.add(old_theme)
                if new_theme:
                    replaced_in_themes.add(new_theme)
                
                # Exclude any item that matches the old theme
                for item in catalog.items:
                    if _word_boundary_match(old_theme, item.normalized_name):
                        state.excluded_ids.add(item.entity_id)
                        if item.entity_id in state.included_ids:
                            state.included_ids.remove(item.entity_id)

        # Check for broader tech/theme exclusions (e.g. "swap out java")
        themes = (
            "java", "python", "sql", "aws", "docker", "c++", "c#", ".net", "javascript",
            "typescript", "angular", "react", "linux", "spring", "excel", "word"
        )
        for theme in themes:
            if theme in replaced_in_themes:
                continue
            if not _word_boundary_match(theme, normalized):
                continue
            position = normalized.find(theme)
            before = normalized[:position]
            last_include = max(
                (before.rfind(marker.strip()) for marker in include_patterns), default=-1
            )
            last_exclude = max(
                (before.rfind(marker.strip()) for marker in exclude_patterns), default=-1
            )
            # If the theme was excluded via text patterns (and not part of a "replace X with Y" that already handled it)
            if last_exclude > last_include and position - last_exclude <= 35:
                state.excluded_themes.add(theme)
                for item in catalog.items:
                    if _word_boundary_match(theme, item.normalized_name):
                        state.excluded_ids.add(item.entity_id)
                        if item.entity_id in state.included_ids:
                            state.included_ids.remove(item.entity_id)




def _word_boundary_match(alias: str, text: str) -> bool:
    """Check if alias appears in text at word boundaries, preventing
    'rest' from matching inside 'restaurant'."""
    if len(alias) <= 2:
        # Very short aliases (like "g+") — require exact word match
        return alias in text.split()
    pattern = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
    return bool(re.search(pattern, text))


def refusal_kind(state: ConversationState) -> str | None:
    if state.injection:
        return "injection"
    if state.legal:
        return "legal"
    if state.off_topic:
        return "off_topic"
    return None


def clarification_question(
    state: ConversationState, catalog: Catalog, aliases: dict[str, str], candidates: list = None
) -> str | None:
    if state.terminal_response or state.confirmation or state.clarification_count >= 2:
        return None
    text = state.normalized_text

    role_terms = (
        "developer",
        "engineer",
        "analyst",
        "manager",
        "front desk",
        "receptionist",
        "agent",
        "concierge",
        "leader",
        "leadership",
        "operator",
        "assistant",
        "admin",
        "sales",
        "contact center",
        "contact centre",
        "call center",
        "customer service",
        "graduate",
        "healthcare",
        "nurse",
        "accountant",
        "finance",
        "marketing",
        "teacher",
        "trainer",
        "consultant",
        "designer",
        "writer",
        "data scientist",
        "recruiter",
        "lawyer",
        "attorney",
        "supervisor",
        "technician",
        "specialist",
        "coordinator",
        "representative",
        "clerk",
        "executive",
        "director",
        "officer",
        "intern",
        "apprentice",
        "architect",
        "scientist",
        "researcher",
        "pharmacist",
        "mechanic",
        "electrician",
        "plumber",
        "driver",
        "warehouse",
        "logistics",
        "supply chain",
        "project manager",
        "product manager",
        "scrum master",
        "devops",
        "sysadmin",
        "security",
        "compliance",
        "auditor",
        "receptionist",
        "teller",
        "cashier",
        "barista",
        "chef",
        "retail",
        "manufacturing",
        "plant",
        "chemical",
        "construction",
        "banking",
        "insurance",
        "pharmaceutical",
        "telecom",
        "staff",
        "shop",
        "checkout",
        "store",
        "boutique",
    )
    named_or_skilled = any(
        token in text
        for token in (
            "java",
            "rust",
            "python",
            "sql",
            "excel",
            "word",
            "hipaa",
            "opq",
            "cognitive",
            "personality",
            "numerical",
            "simulation",
            "c#",
            "c++",
            ".net",
            "javascript",
            "typescript",
            "angular",
            "react",
            "linux",
            "aws",
            "docker",
            "networking",
            "spring",
            "accounting",
            "statistics",
            "mechanical",
            "electrical",
            "safety",
            "dependability",
            "verify",
            "aptitude",
            "reasoning",
            "competency",
            "situational",
            "enterprise",
            "beans",
        )
    )
    
    catalog_product_mentioned = False
    for alias in aliases:
        if _word_boundary_match(alias, text):
            catalog_product_mentioned = True
            break
    if not catalog_product_mentioned:
        for item in catalog.items:
            if item.normalized_name in text:
                catalog_product_mentioned = True
                break
    
    if not any(term in text for term in role_terms) and not named_or_skilled and not catalog_product_mentioned:
        return "What role or job family are you assessing, and what capability matters most?"

    if _word_boundary_match("analyst", text) and not any(term in text for term in ("business", "data", "finance", "financial", "marketing", "systems", "security", "financial", "pricing")):
        return "What type of analyst role is this? (e.g., financial, data, business, or marketing?)"

    if "leadership" in text and not any(
        term in text for term in ("selection", "development", "reskill", "re-skill")
    ):
        return "Is this for selecting leaders or developing leaders already in role?"

    if any(term in text for term in ("contact center", "contact centre", "call center")):
        if not any(term in text for term in ("english", "spanish", "french", "german")):
            return "What language and customer accent or locale should the assessment support?"

    if candidates and state.required_languages:
        knowledge = []
        for cand in candidates:
            item = catalog.by_id.get(cand.entity_id)
            if item and item.test_type and "K" in item.test_type:
                knowledge.append(item)
                
        if knowledge:
            incompatible = []
            for item in knowledge:
                supported = False
                for req_lang in state.required_languages:
                    for lang in item.languages:
                        if req_lang.lower() in lang.lower():
                            supported = True
                            break
                    if supported:
                        break
                # If the item doesn't support ANY of the required languages
                if not supported:
                    incompatible.append(item)
            
            if incompatible and not any(term in text for term in ("bilingual", "english fluent", "hybrid", "english")):
                req_lang_str = " or ".join(lang.capitalize() for lang in state.required_languages)
                return (
                    f"The role-knowledge assessments are only available in English. "
                    f"Are candidates comfortable taking those in English, or do you want {req_lang_str}-only assessments?"
                )

    if any(term in text for term in ("admin assistant", "administrative assistant")):
        if "quick" in text and not any(
            term in text for term in ("simulation", "capability", "hands on", "hands-on")
        ):
            return "Do you want only short knowledge checks, or should the shortlist also include longer hands-on Office simulations?"

    if any(term in text for term in ("full stack", "full-stack")) and not any(
        term in text for term in ("backend leaning", "backend-leaning", "frontend leaning", "balanced")
    ):
        return "Is the role backend-leaning, frontend-leaning, or genuinely balanced full-stack?"

    if any(term in text for term in ("developer", "engineer")) and not any(
        term in text
        for term in ("entry", "graduate", "junior", "mid", "senior", "lead", "years",
                      "advanced", "checkout", "store", "shop", "boutique")
    ):
        return "What seniority level is the role?"
    return None


def comparison_entities(
    state: ConversationState, catalog: Catalog, aliases: dict[str, str]
) -> list[str]:
    normalized = normalize_text(state.last_user_text)
    matches: list[tuple[int, str]] = []
    for alias, entity_id in aliases.items():
        if _word_boundary_match(alias, normalized):
            position = normalized.find(alias)
            if position >= 0:
                matches.append((position, entity_id))
    for item in catalog.items:
        position = normalized.find(item.normalized_name)
        if position >= 0:
            matches.append((position, item.entity_id))
    result: list[str] = []
    for _, entity_id in sorted(matches):
        if entity_id not in result:
            result.append(entity_id)
    return result[:3]
