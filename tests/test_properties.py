from __future__ import annotations
from hypothesis import settings

from hypothesis import given, settings, strategies as st
import pytest

from app.schemas import ChatRequest

@pytest.fixture(autouse=True)
def disable_llm(service):
    """Disable LLM to avoid 429 quota errors during property testing."""
    service.gemini.client = None
    return service


ROLES = [
    "Java engineer",
    "sales manager",
    "plant operator",
    "graduate analyst",
    "contact center agent",
    "healthcare administrator",
    "executive leadership",
    "admin assistant",
    "Rust systems engineer",
    "financial accountant",
]

EDITS = [
    "Add OPQ personality assessment.",
    "Remove REST. Add Docker.",
    "Drop the cognitive test.",
    "Also include a simulation.",
    "Exclude OPQ.",
]


@pytest.mark.asyncio

@given(st.sampled_from(ROLES))
@settings(max_examples=len(ROLES))
async def test_recommendations_are_unique_and_catalog_backed(service, role: str) -> None:
    """Every response must have unique catalog-backed URLs with count <= 10."""
    response = await service.chat(
        ChatRequest(messages=[{"role": "user", "content": role}])
    )
    urls = [item.url for item in response.recommendations]
    assert len(urls) <= 10
    assert len(urls) == len(set(urls))
    assert all(item.url in service.catalog.by_url for item in response.recommendations)


@pytest.mark.asyncio

@given(st.sampled_from(ROLES))
@settings(max_examples=len(ROLES))
async def test_no_url_in_reply_text(service, role: str) -> None:
    """reply must never contain http/https URLs — only recommendation objects carry URLs."""
    response = await service.chat(
        ChatRequest(messages=[{"role": "user", "content": role}])
    )
    assert "http://" not in response.reply
    assert "https://" not in response.reply


@pytest.mark.asyncio

@given(st.sampled_from(ROLES))
@settings(max_examples=len(ROLES))
async def test_recommendations_always_array(service, role: str) -> None:
    """recommendations is always an array, never None."""
    response = await service.chat(
        ChatRequest(messages=[{"role": "user", "content": role}])
    )
    assert isinstance(response.recommendations, list)


@pytest.mark.asyncio

@given(st.sampled_from(ROLES))
@settings(max_examples=len(ROLES))
async def test_end_of_conversation_is_bool(service, role: str) -> None:
    """end_of_conversation is always a boolean."""
    response = await service.chat(
        ChatRequest(messages=[{"role": "user", "content": role}])
    )
    assert isinstance(response.end_of_conversation, bool)


@pytest.mark.asyncio

@given(st.sampled_from(ROLES))
@settings(max_examples=len(ROLES))
async def test_reply_is_nonempty_string(service, role: str) -> None:
    """reply is always a nonempty string."""
    response = await service.chat(
        ChatRequest(messages=[{"role": "user", "content": role}])
    )
    assert isinstance(response.reply, str)
    assert len(response.reply) >= 1


@pytest.mark.asyncio

@given(st.sampled_from(ROLES))
@settings(max_examples=len(ROLES))
async def test_no_duplicate_entity_ids(service, role: str) -> None:
    """No two recommendations share the same entity ID."""
    response = await service.chat(
        ChatRequest(messages=[{"role": "user", "content": role}])
    )
    urls = [item.url for item in response.recommendations]
    names = [item.name for item in response.recommendations]
    assert len(urls) == len(set(urls))
    assert len(names) == len(set(names))


@pytest.mark.asyncio

@given(st.sampled_from(["legal advice please", "write offer letter", "salary negotiation"]))
@settings(max_examples=3)
async def test_refusal_always_empty_array(service, msg: str) -> None:
    """Refusal/off-topic always returns empty recommendations."""
    response = await service.chat(
        ChatRequest(messages=[{"role": "user", "content": msg}])
    )
    assert response.recommendations == []


@pytest.mark.asyncio
async def test_excluded_ids_never_reappear(service) -> None:
    """Explicit exclusions must never reappear in recommendations."""
    messages = [
        {"role": "user", "content": "Senior Java engineer using Spring, REST, and SQL. Also include OPQ."},
        {"role": "assistant", "content": "Here is a shortlist."},
        {"role": "user", "content": "Remove OPQ. Drop REST."},
    ]
    response = await service.chat(ChatRequest(messages=messages))
    names = {item.name for item in response.recommendations}
    assert "Occupational Personality Questionnaire OPQ32r" not in names
    assert "RESTful Web Services (New)" not in names


@pytest.mark.asyncio
async def test_first_turn_clarification_has_empty_array(service) -> None:
    """Clarification always returns empty recommendations array."""
    response = await service.chat(
        ChatRequest(messages=[{"role": "user", "content": "I need an assessment"}])
    )
    assert response.recommendations == []
    assert response.end_of_conversation is False


@pytest.mark.asyncio
async def test_confirmation_has_true_eoc(service) -> None:
    """Confirmation always sets end_of_conversation to true."""
    messages = [
        {"role": "user", "content": "Senior Java engineer"},
        {"role": "assistant", "content": "Here is a shortlist."},
        {"role": "user", "content": "Thanks, that's all. Lock it in."},
    ]
    response = await service.chat(ChatRequest(messages=messages))
    assert response.end_of_conversation is True
