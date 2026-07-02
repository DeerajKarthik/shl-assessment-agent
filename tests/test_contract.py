from __future__ import annotations

import httpx
import pytest

from app.main import app
from app.service import RecommenderService
from app.settings import settings


@pytest.fixture
def transport(service: RecommenderService) -> httpx.ASGITransport:
    app.state.recommender = service
    return httpx.ASGITransport(app=app)


# --- Existing contract tests ---

@pytest.mark.asyncio
async def test_health_contract(transport) -> None:
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_vague_chat_contract(transport) -> None:
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "I need an assessment"}]},
        )
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"reply", "recommendations", "end_of_conversation"}
    assert payload["recommendations"] == []
    assert payload["end_of_conversation"] is False


@pytest.mark.asyncio
async def test_invalid_role_sequence_is_422(transport) -> None:
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat",
            json={"messages": [{"role": "assistant", "content": "hello"}]},
        )
    assert response.status_code == 422


def test_only_two_routes_are_exposed() -> None:
    paths = {route.path for route in app.routes}
    assert paths == {"/health", "/chat"}


# --- New error-code and edge-case tests ---

@pytest.mark.asyncio
async def test_empty_messages_is_422(transport) -> None:
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/chat", json={"messages": []})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_missing_messages_field_is_422(transport) -> None:
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/chat", json={})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_extra_field_is_422(transport) -> None:
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "temperature": 0.5,
            },
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_consecutive_user_messages_is_422(transport) -> None:
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat",
            json={
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "user", "content": "world"},
                ]
            },
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_too_many_messages_is_422(transport) -> None:
    """8 messages exceeds the 7-message cap."""
    messages = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
        {"role": "user", "content": "e"},
        {"role": "assistant", "content": "f"},
        {"role": "user", "content": "g"},
        {"role": "assistant", "content": "h"},
    ]
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/chat", json={"messages": messages})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_chat_is_405(transport) -> None:
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/chat")
    assert response.status_code == 405


@pytest.mark.asyncio
async def test_post_health_is_405(transport) -> None:
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/health")
    assert response.status_code == 405


@pytest.mark.asyncio
async def test_nonexistent_route_is_404(transport) -> None:
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v2/chat")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_response_has_exact_schema_keys(transport) -> None:
    """Response must have exactly reply, recommendations, end_of_conversation — no extras."""
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "Senior Java engineer"}]},
        )
    payload = response.json()
    assert set(payload.keys()) == {"reply", "recommendations", "end_of_conversation"}
    # Verify recommendation sub-schema
    if payload["recommendations"]:
        rec = payload["recommendations"][0]
        assert set(rec.keys()) == {"name", "url", "test_type"}


@pytest.mark.asyncio
async def test_recommendations_count_bounded(transport) -> None:
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "Senior Java engineer using Spring, REST, SQL, AWS, Docker, Angular, React, Vue, Python, Go, Rust"}]},
        )
    payload = response.json()
    assert len(payload["recommendations"]) <= 10
