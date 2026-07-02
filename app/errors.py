from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Structured metadata for a single /chat request."""

    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: float = field(default_factory=time.perf_counter)

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started_at) * 1000

    def log_fields(self, **extras: Any) -> dict[str, Any]:
        return {"request_id": self.request_id, "elapsed_ms": round(self.elapsed_ms(), 1), **extras}


class RecommenderError(Exception):
    """Base error for the recommender application."""


class CatalogLoadError(RecommenderError):
    """Raised when the catalog cannot be loaded or validated."""


class ProviderTimeoutError(RecommenderError):
    """Raised when the LLM provider exceeds the configured deadline."""


class ProviderError(RecommenderError):
    """Raised on any non-timeout LLM provider failure."""


def safe_fallback_reply() -> str:
    """Return a valid reply string when all else fails."""
    return (
        "I need a little more detail to identify a grounded SHL shortlist. "
        "Could you share the role title and the main assessment objective?"
    )


def log_request_start(ctx: RequestContext, message_count: int) -> None:
    logger.info("chat_start", extra=ctx.log_fields(message_count=message_count))


def log_request_end(
    ctx: RequestContext,
    *,
    action: str,
    recommendation_count: int,
    candidate_count: int = 0,
    model_used: bool = False,
    fallback_used: bool = False,
    validation_failure: str | None = None,
) -> None:
    logger.info(
        "chat_end",
        extra=ctx.log_fields(
            action=action,
            recommendation_count=recommendation_count,
            candidate_count=candidate_count,
            model_used=model_used,
            fallback_used=fallback_used,
            validation_failure=validation_failure,
        ),
    )
