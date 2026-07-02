from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

from app.catalog import Catalog
from app.llm import GeminiAdapter
from app.policy import (
    ConversationState,
    build_state,
    clarification_question,
    comparison_entities,
    refusal_kind,
    _parse_duration_minutes,
)
from app.render import comparison_reply, recommendation_reply, refusal_reply
from app.retrieval import Candidate, HybridRetriever
from app.schemas import ChatRequest, ChatResponse
from app.settings import Settings
from app.errors import RequestContext, log_request_start, log_request_end, safe_fallback_reply
from app.render import sanitize_reply
from app.dialogue import DialogueGenerator

logger = logging.getLogger(__name__)


class RecommenderService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.catalog = Catalog.load(settings.catalog_path)
        self.retriever = HybridRetriever(
            catalog=self.catalog,
            aliases_path=settings.aliases_path,
            ontology_path=settings.ontology_path,
            embeddings_path=settings.embeddings_path,
            embeddings_meta_path=settings.embeddings_meta_path,
        )
        self.aliases = self.retriever.aliases
        self.dependencies: dict[str, list[str]] = json.loads(
            settings.dependencies_path.read_text()
        )
        self.gemini = GeminiAdapter(settings, self.catalog)
        self.dialogue = DialogueGenerator(settings, self.gemini.mistral_client)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        ctx = RequestContext()
        log_request_start(ctx, len(request.messages))
        try:
            async with asyncio.timeout(self.settings.application_timeout_seconds):
                response = await self._chat(request, ctx)
                log_request_end(
                    ctx,
                    action="ok",
                    recommendation_count=len(response.recommendations),
                )
                return response
        except TimeoutError:
            logger.error("application timeout — falling back to deterministic retrieval", extra=ctx.log_fields())
            # On timeout, fall back to deterministic retrieval instead of empty response
            try:
                response = await self._deterministic_fallback(request)
                log_request_end(ctx, action="timeout_fallback", recommendation_count=len(response.recommendations), fallback_used=True)
                return response
            except Exception:
                logger.exception("deterministic fallback also failed", extra=ctx.log_fields())
                return ChatResponse(
                    reply=safe_fallback_reply(),
                    recommendations=[],
                    end_of_conversation=len(request.messages) == 7,
                )
        except Exception:
            logger.exception("unhandled error", extra=ctx.log_fields())
            return ChatResponse(
                reply=safe_fallback_reply(),
                recommendations=[],
                end_of_conversation=len(request.messages) == 7,
            )

    async def _deterministic_fallback(self, request: ChatRequest) -> ChatResponse:
        """Run retrieval without any LLM calls for timeout fallback."""
        state = build_state(request.messages, self.catalog, self.aliases)
        kind = refusal_kind(state)
        if kind:
            return ChatResponse(reply=refusal_reply(kind), recommendations=[], end_of_conversation=state.terminal_response)

        candidates = self.retriever.search(state, query_embedding=None, limit=self.settings.candidate_limit)
        candidate_ids = [c.entity_id for c in candidates]
        selected = self._apply_hard_constraints(state, candidate_ids, [])
        if not selected:
            return ChatResponse(
                reply="I need a little more role or capability detail to identify a grounded SHL shortlist. What is the role and the main assessment objective?",
                recommendations=[],
                end_of_conversation=state.terminal_response,
            )
        items = [self.catalog.by_id[eid] for eid in selected]
        reply = recommendation_reply(items, confirmed=state.confirmation)
        return ChatResponse(
            reply=reply,
            recommendations=[item.recommendation() for item in items],
            end_of_conversation=state.confirmation or state.terminal_response,
        )

    async def _chat(self, request: ChatRequest, ctx: RequestContext) -> ChatResponse:
        state = build_state(request.messages, self.catalog, self.aliases)

        kind = refusal_kind(state)
        if kind:
            base_reply = refusal_reply(kind)
            reply, mistral_used = await self.dialogue.generate(
                action="refuse",
                context={"refusal_kind": kind, "user_input": state.last_user_text},
                fallback=base_reply
            )
            provider_str = "Policy Engine (Narrated by Mistral)" if mistral_used else "Policy Engine"
            return ChatResponse(
                reply=reply,
                recommendations=[],
                end_of_conversation=state.terminal_response,
                provider=provider_str
            )

        if (
            state.comparison
            and not state.terminal_response
            and not _comparison_is_also_edit(state.last_user_text)
        ):
            entity_ids = comparison_entities(state, self.catalog, self.aliases)
            items = [self.catalog.by_id[entity_id] for entity_id in entity_ids]
            base_reply = comparison_reply(items)
            reply, mistral_used = await self.dialogue.generate(
                action="compare",
                context={
                    "items": [item.prompt_record() for item in items],
                    "user_input": state.last_user_text
                },
                fallback=base_reply
            )
            provider_str = "Policy Engine (Narrated by Mistral)" if mistral_used else "Policy Engine"
            return ChatResponse(
                reply=reply,
                recommendations=[],
                end_of_conversation=state.terminal_response,
                provider=provider_str
            )

        query_embedding = await self.gemini.embed_query(state.combined_user_text)
        candidates = self.retriever.search(
            state,
            query_embedding=query_embedding,
            limit=self.settings.candidate_limit,
        )
        
        # Check clarification AFTER getting candidates so we can inspect candidate languages
        question = clarification_question(state, self.catalog, self.aliases, candidates=candidates)
        if question:
            reply, mistral_used = await self.dialogue.generate(
                action="clarify",
                context={"clarification_reason": question, "user_input": state.last_user_text},
                fallback=question
            )
            provider_str = "Policy Engine (Narrated by Mistral)" if mistral_used else "Policy Engine"
            return ChatResponse(
                reply=reply,
                recommendations=[],
                end_of_conversation=False,
                provider=provider_str
            )
        selected_ids_tuple = await self._select_ids(state, candidates)
        selected_ids, provider = selected_ids_tuple
        if not selected_ids:
            base_reply = (
                "I need a little more role or capability detail to identify a grounded "
                "SHL shortlist. What is the role and the main assessment objective?"
            )
            reply, mistral_used = await self.dialogue.generate(
                action="clarify",
                context={"clarification_reason": "Missing role or objective details", "user_input": state.last_user_text},
                fallback=base_reply
            )
            provider_str = "Policy Engine (Narrated by Mistral)" if mistral_used else "Policy Engine"
            return ChatResponse(
                reply=reply,
                recommendations=[],
                end_of_conversation=state.terminal_response,
                provider=provider_str
            )

        items = [self.catalog.by_id[entity_id] for entity_id in selected_ids]
        base_reply = recommendation_reply(items, confirmed=state.confirmation)
        
        context_data = {
            "user_input": state.last_user_text,
            "constraints": state.current_constraints(),
            "recommended_types": list(set(t for item in items for t in item.test_type.split(","))),
            "is_confirmation": state.confirmation
        }
        
        if state.comparison:
            compared_ids = comparison_entities(state, self.catalog, self.aliases)
            compared_items = [self.catalog.by_id[entity_id] for entity_id in compared_ids]
            if len(compared_items) >= 2:
                base_reply = comparison_reply(compared_items) + " " + base_reply
                context_data["comparison_items"] = [item.prompt_record() for item in compared_items]
                
        action = "confirm" if state.confirmation else "recommend"
        if state.excluded_themes and any(theme in state.last_user_text.lower() for theme in state.excluded_themes):
            action = "edit_recommendation"
            context_data["dropped_themes"] = list(state.excluded_themes)
            
        reply, mistral_used = await self.dialogue.generate(action, context_data, base_reply)
        
        provider_str = provider
        if mistral_used and "Narrated by Mistral" not in provider_str:
            provider_str += " (Narrated by Mistral)"
            
        return ChatResponse(
            reply=sanitize_reply(reply),
            recommendations=[item.recommendation() for item in items],
            end_of_conversation=state.confirmation or state.terminal_response,
            provider=provider_str
        )

    async def _select_ids(
        self, state: ConversationState, candidates: list[Candidate]
    ) -> tuple[list[str], str]:
        # Extract Top N candidate IDs
        raw_candidate_ids = [candidate.entity_id for candidate in candidates]
        
        # 1. Deterministic pre-filter: ensure hard constraints before sending to LLM
        # This allows us to safely retrieve 20, filter out invalid ones, and send the top 12 valid to Gemini
        pre_filtered_ids = self._apply_hard_constraints(state, raw_candidate_ids, [], desired=20)
        candidate_ids = pre_filtered_ids[:20]
        
        anchor_ids = self._ontology_anchor_ids(state)
        decision = await self.gemini.rerank(state, candidate_ids)

        proposed: list[str] = []
        proposed.extend(state.included_ids)
        proposed.extend(anchor_ids)
        
        # LLM results get priority over general retrieval
        if decision is not None:
            proposed.extend(decision.selected_entity_ids)
            boost_language = False
        else:
            boost_language = True
            
        proposed.extend(candidate_ids)

        desired = self._desired_count(state, anchor_ids)
        selected = self._apply_hard_constraints(state, proposed, anchor_ids, desired, boost_language=boost_language)

        selected = self._apply_dependencies(selected, state.excluded_ids)
        provider = "Mistral" if decision is not None else "Deterministic"
        return selected[:10], provider

    def _apply_hard_constraints(
        self,
        state: ConversationState,
        proposed: list[str],
        anchors: list[str],
        desired: int | None = None,
        boost_language: bool = True,
    ) -> list[str]:
        """Filter proposed IDs through hard constraints: type exclusions, duration."""
        if desired is None:
            desired = min(6, max(1, len([t for t in state.requested_types]) + 3))

        # Boost by language instead of strictly filtering
        if state.required_languages and boost_language:
            def lang_score(eid: str) -> int:
                if eid not in self.catalog.by_id:
                    return 0
                item = self.catalog.by_id[eid]
                if not item.languages:
                    return 0
                for required in state.required_languages:
                    for lang in item.languages:
                        if required.lower() in lang.lower():
                            return 1
                return 0
            # Stable sort to preserve retriever/LLM rank while boosting language matches
            proposed = sorted(proposed, key=lang_score, reverse=True)

        selected: list[str] = []
        for entity_id in self.catalog.validate_ids(proposed):
            if entity_id in state.excluded_ids or entity_id in selected:
                continue

            item = self.catalog.by_id[entity_id]

            # Hard type/category filtering
            if state.excluded_types:
                item_types = set(item.test_type.split(","))
                if item_types & state.excluded_types:
                    continue

            # Hard duration filtering
            if state.max_duration_minutes is not None:
                item_duration = _parse_duration_minutes(item.duration)
                if item_duration is not None and item_duration > state.max_duration_minutes:
                    continue

            selected.append(entity_id)
            if len(selected) == desired:
                break

        return selected

    def _ontology_anchor_ids(self, state: ConversationState) -> list[str]:
        query = state.normalized_text
        result: list[str] = []
        for rule in self.retriever.ontology:
            all_terms = [str(term).casefold() for term in rule.get("all", [])]
            any_terms = [str(term).casefold() for term in rule.get("any", [])]
            
            # If the rule's core triggers (all or name) overlap with a theme we explicitly excluded,
            # this means the rule represents the excluded concept (e.g. Java). 
            # We should explicitly exclude its entities (like Spring).
            core_keywords = set(all_terms + [rule.get("name", "").casefold()])
            if any(exc in core_keywords or exc in rule.get("name", "").casefold() for exc in state.excluded_themes):
                for entity_id in rule.get("entity_ids", []):
                    state.excluded_ids.add(str(entity_id))
                continue
                
            if all_terms and not all(term in query for term in all_terms):
                continue
            if any_terms and not any(term in query for term in any_terms):
                continue
            for entity_id in rule.get("entity_ids", []):
                entity_id = str(entity_id)
                if entity_id not in result and entity_id not in state.excluded_ids:
                    result.append(entity_id)

        return result

    def _desired_count(self, state: ConversationState, anchors: list[str]) -> int:
        match = re.search(r"\b(?:top|only|exactly)\s+(\d{1,2})\b", state.last_user_text, re.I)
        if match:
            return max(1, min(10, int(match.group(1))))
        return 10

    def _apply_dependencies(
        self, selected: list[str], excluded_ids: set[str]
    ) -> list[str]:
        output = list(selected)
        for entity_id in list(output):
            for required_id in self.dependencies.get(entity_id, []):
                if required_id not in output and required_id not in excluded_ids:
                    output.insert(0, required_id)
        deduplicated: list[str] = []
        for entity_id in output:
            if entity_id not in deduplicated:
                deduplicated.append(entity_id)
        return deduplicated[:10]


_EDIT_IN_COMPARISON_RE = re.compile(
    r"\b(add|remove|drop|update|replace|swap|exclude|include|instead of)\b", re.I
)


def _comparison_is_also_edit(text: str) -> bool:
    return bool(_EDIT_IN_COMPARISON_RE.search(text))
