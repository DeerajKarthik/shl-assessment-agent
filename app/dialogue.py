import asyncio
import logging
from typing import Any, Dict, List
import json

from app.settings import Settings

logger = logging.getLogger(__name__)

DIALOGUE_SYSTEM_PROMPT = """\
You are an expert SHL Assessment Consultant. Your job is to narrate the reasoning of our deterministic recommendation engine in a natural, consultative, and professional tone.

RULES:
1. DO NOT invent assessments, URLs, languages, durations, or facts.
2. Rely strictly on the provided context.
3. Keep responses concise but consultative.
4. Do not offer partial credit or apologize excessively.
5. Never ask for information that is already present in the extracted conversation state.
6. Begin every recommendation by summarizing the understood hiring requirements in one sentence before presenting recommendations. Translate raw user terms into professional recruiter language (e.g., "good at conversations" -> "communication and customer service behaviour").
7. Explicitly explain WHY the shortlist exists (e.g. "For this role I've balanced technical assessment with reasoning ability and workplace behaviour").
8. If clarifying, explain the trade-off or missing information.
9. Return plain text only (no JSON, no markdown).
"""

class DialogueGenerator:
    def __init__(self, settings: Settings, mistral_client: Any) -> None:
        self.settings = settings
        self.mistral_client = mistral_client

    async def generate(self, action: str, context: Dict[str, Any], fallback: str) -> tuple[str, bool]:
        if not self.mistral_client:
            return fallback, False

        prompt = f"Action: {action}\n\nContext:\n{json.dumps(context, indent=2)}\n\nPlease generate the natural conversational response. Ensure it matches the consultant persona."

        try:
            async with asyncio.timeout(self.settings.gemini_timeout_seconds):
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.mistral_client.chat.complete(
                        model=self.settings.model_name,
                        messages=[
                            {"role": "system", "content": DIALOGUE_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3
                    )
                )
            return response.choices[0].message.content.strip(), True
        except Exception as e:
            logger.warning(f"Dialogue generation failed: {e}. Using fallback.")
            return fallback, False
