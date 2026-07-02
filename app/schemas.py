from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20_000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[Message] = Field(min_length=1, max_length=7)

    @field_validator("messages")
    @classmethod
    def validate_sequence(cls, messages: list[Message]) -> list[Message]:
        if messages[0].role != "user":
            raise ValueError("conversation must begin with a user message")
        if messages[-1].role != "user":
            raise ValueError("conversation must end with a user message")
        for previous, current in zip(messages, messages[1:]):
            if previous.role == current.role:
                raise ValueError("message roles must alternate")
        return messages


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reply: str = Field(min_length=1)
    recommendations: list[Recommendation] = Field(max_length=10)
    end_of_conversation: bool
    provider: str | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def validate_recommendations(self) -> "ChatResponse":
        urls = [item.url for item in self.recommendations]
        if len(urls) != len(set(urls)):
            raise ValueError("recommendation URLs must be unique")
        return self


class ModelDecision(BaseModel):
    """Untrusted internal model output; never returned directly."""

    action: Literal["recommend", "clarify", "compare", "refuse", "confirm"] = (
        "recommend"
    )
    selected_entity_ids: list[str] = Field(default_factory=list, max_length=10)
    inferred_skills: list[str] = Field(default_factory=list, max_length=20)
    inferred_categories: list[Literal["A", "B", "C", "D", "E", "K", "P", "S"]] = (
        Field(default_factory=list, max_length=8)
    )
    clarification_question: str = ""

