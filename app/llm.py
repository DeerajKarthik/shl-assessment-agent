from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import numpy as np

from app.catalog import Catalog
from app.policy import ConversationState
from app.schemas import ModelDecision
from app.settings import Settings

try:
    from google import genai
    from google.genai import types
except ImportError:  # Local deterministic mode remains fully usable.
    genai = None
    types = None

try:
    import groq
    from groq import AsyncGroq
except ImportError:
    groq = None
    AsyncGroq = None

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are the ranking component of a stateless SHL assessment recommender.

RULES:
1. Return ONLY JSON matching this exact schema: {"action": "recommend" | "clarify" | "compare", "selected_entity_ids": ["id1", "id2"]}
2. Every selected ID MUST exist in candidates.
3. Never omit an explicitly requested assessment.
4. Prefer recommendations that satisfy the user's industry/domain before adding general-purpose assessments.
5. Do NOT exclude highly relevant domain-specific assessments just because they lack the requested language. Domain match overrides language match.
6. Never output explanations.
7. Maximum 10 IDs.
8. For "action": ALWAYS use "recommend" unless the user's request is completely unanswerable. Do not ask for optional preferences like format or duration if you already have highly relevant domain candidates.
9. Use "compare" only for explicit comparison requests.
"""


class GeminiAdapter:
    def __init__(self, settings: Settings, catalog: Catalog) -> None:
        self.settings = settings
        self.catalog = catalog
        self.client: Any | None = None
        self.cerebras_client: Any | None = None
        self.mistral_client: Any | None = None
        
        if genai is not None and settings.gemini_api_key:
            self.client = genai.Client(api_key=settings.gemini_api_key)
            
        if self.settings.mistral_api_key:
            try:
                from mistralai.client import Mistral
                self.mistral_client = Mistral(api_key=self.settings.mistral_api_key)
            except ImportError:
                pass

    @property
    def enabled(self) -> bool:
        return self.mistral_client is not None

    async def embed_query(self, text: str) -> np.ndarray | None:
        if "nomic" in self.settings.embedding_model or "bge" in self.settings.embedding_model or "sentence-transformers" in self.settings.embedding_model:
            from sentence_transformers import SentenceTransformer
            if not hasattr(self, "_st_model"):
                self._st_model = SentenceTransformer(self.settings.embedding_model, trust_remote_code=True)
            
            prefix = ""
            if "bge" in self.settings.embedding_model.lower():
                prefix = "Represent this sentence for searching relevant passages: "
            elif "nomic" in self.settings.embedding_model.lower():
                prefix = "search_query: "
                
            embedding = self._st_model.encode(f"{prefix}{text}")
            return np.asarray(embedding, dtype=np.float32)
        if not self.enabled or types is None:
            return None
        try:
            async with asyncio.timeout(self.settings.gemini_timeout_seconds):
                result = await self.client.aio.models.embed_content(
                    model=self.settings.embedding_model,
                    contents=text,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_QUERY", output_dimensionality=768
                    ),
                )
            if not result.embeddings:
                return None
            return np.asarray(result.embeddings[0].values, dtype=np.float32)
        except TimeoutError:
            logger.warning("embed_query timed out")
            return None
        except Exception:
            logger.warning("embed_query failed", exc_info=True)
            return None

    async def rerank(
        self, state: ConversationState, candidate_ids: list[str]
    ) -> ModelDecision | None:
        if not self.enabled:
            return None
        candidates = [self.catalog.by_id[entity_id].prompt_record() for entity_id in candidate_ids]

        payload = {
            "conversation_summary": state.current_constraints(),
            "requested_type_codes": sorted(state.requested_types),
            "excluded_type_codes": sorted(state.excluded_types),
            "must_include_ids": state.included_ids,
            "must_exclude_ids": sorted(state.excluded_ids),
            "required_languages": state.required_languages,
            "max_duration_minutes": state.max_duration_minutes,
            "candidates": candidates,
        }
        try:
            async with asyncio.timeout(self.settings.gemini_timeout_seconds):
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"User's request:\n{state.combined_user_text}\n\nPlease fulfill the ranking constraints based on this payload:\n{json.dumps(payload, ensure_ascii=False)}"}
                ]
                
                # Using synchronous call in executor as Mistral's async client can be flaky
                loop = asyncio.get_running_loop()
                
                from mistralai.client.errors.sdkerror import SDKError
                
                retry_delays = [1, 2]
                for attempt in range(len(retry_delays) + 1):
                    try:
                        response = await loop.run_in_executor(
                            None,
                            lambda: self.mistral_client.chat.complete(
                                model=self.settings.model_name,
                                messages=messages,
                                temperature=0.0,
                                response_format={"type": "json_object"}
                            )
                        )
                        break
                    except SDKError as e:
                        if getattr(e, "status_code", 0) == 429 or "429" in str(e):
                            if attempt < len(retry_delays):
                                delay = retry_delays[attempt]
                                logger.warning(f"Mistral rate limit hit (429). Sleeping for {delay}s...")
                                await asyncio.sleep(delay)
                            else:
                                raise e
                        else:
                            raise e
                
            response_text = response.choices[0].message.content
            return ModelDecision.model_validate_json(response_text)
        except TimeoutError:
            logger.warning("rerank timed out after %.1fs", self.settings.gemini_timeout_seconds)
            return None
        except Exception as e:
            if "429" in str(e):
                logger.warning("Mistral rate limit exceeded (429). Falling back to deterministic retrieval.")
            else:
                logger.warning("rerank failed: %s", e)
            return None
