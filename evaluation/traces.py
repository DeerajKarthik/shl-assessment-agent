from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


USER_BLOCK_RE = re.compile(r"\*\*User\*\*\n\n((?:>.*\n?)+)")
TABLE_BLOCK_RE = re.compile(r"(?:^\|.*\|\n)+", re.MULTILINE)
URL_RE = re.compile(r"<(https://www\.shl\.com/[^>]+)>")


@dataclass(frozen=True, slots=True)
class PublicTrace:
    trace_id: str
    user_turns: tuple[str, ...]
    expected_urls: tuple[str, ...]


def load_public_traces(directory: Path) -> list[PublicTrace]:
    traces: list[PublicTrace] = []
    paths = sorted(directory.glob("C*.md"), key=lambda path: int(path.stem[1:]))
    for path in paths:
        text = path.read_text()
        user_turns = tuple(_clean_quote(block) for block in USER_BLOCK_RE.findall(text))
        terminal = text.rfind("_`end_of_conversation`: **true**_")
        table_blocks = TABLE_BLOCK_RE.findall(text[:terminal])
        expected_urls = tuple(URL_RE.findall(table_blocks[-1])) if table_blocks else ()
        traces.append(PublicTrace(path.stem, user_turns, expected_urls))
    return traces


def _clean_quote(block: str) -> str:
    return "\n".join(line[1:].lstrip() for line in block.splitlines()).strip()


def recall_at_10(predicted_urls: list[str], expected_urls: tuple[str, ...]) -> float:
    if not expected_urls:
        return 1.0
    predicted = set(predicted_urls[:10])
    return len(predicted & set(expected_urls)) / len(set(expected_urls))

