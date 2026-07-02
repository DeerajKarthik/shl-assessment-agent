from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import urlsplit, urlunsplit

from app.schemas import Recommendation


SHL_URL_PREFIX = "https://www.shl.com/products/product-catalog/view/"

KEY_TO_CODE = {
    "Ability & Aptitude": "A",
    "Assessment Exercises": "E",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

PRIMARY_TYPE_OVERRIDES = {
    "4302": "D",  # Global Skills Development Report, matching the public trace.
}


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = value.replace("&", " and ")
    value = re.sub(r"[^\w+#.]+", " ", value)
    return " ".join(value.split())


def canonicalize_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    path = re.sub(r"/+", "/", parsed.path)
    if not path.endswith("/"):
        path += "/"
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, "", ""))


@dataclass(frozen=True, slots=True)
class CatalogItem:
    entity_id: str
    name: str
    link: str
    job_levels: tuple[str, ...]
    languages: tuple[str, ...]
    duration: str
    adaptive: bool
    description: str
    keys: tuple[str, ...]
    test_type: str
    normalized_name: str
    search_text: str

    def recommendation(self) -> Recommendation:
        return Recommendation(name=self.name, url=self.link, test_type=self.test_type)

    def prompt_record(self) -> dict[str, object]:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "test_type": self.test_type,
            "job_levels": list(self.job_levels),
            "languages": list(self.languages),
            "keys": list(self.keys),
            "duration": self.duration,
        }


class Catalog:
    def __init__(self, items: list[CatalogItem], source_sha256: str) -> None:
        self.items = items
        self.source_sha256 = source_sha256
        self.by_id = {item.entity_id: item for item in items}
        self.by_url = {canonicalize_url(item.link): item for item in items}
        self.by_normalized_name = {item.normalized_name: item for item in items}

    @classmethod
    def load(cls, path: Path) -> "Catalog":
        raw_bytes = path.read_bytes()
        source_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        raw = json.loads(raw_bytes)
        if not isinstance(raw, list) or not raw:
            raise ValueError("catalog root must be a non-empty JSON array")

        items: list[CatalogItem] = []
        seen_ids: set[str] = set()
        seen_names: set[str] = set()
        seen_urls: set[str] = set()
        for position, record in enumerate(raw):
            item = cls._parse_record(record, position)
            canonical_url = canonicalize_url(item.link)
            if item.entity_id in seen_ids:
                raise ValueError(f"duplicate entity_id: {item.entity_id}")
            if item.name in seen_names:
                raise ValueError(f"duplicate catalog name: {item.name}")
            if canonical_url in seen_urls:
                raise ValueError(f"duplicate catalog URL: {canonical_url}")
            seen_ids.add(item.entity_id)
            seen_names.add(item.name)
            seen_urls.add(canonical_url)
            items.append(item)

        return cls(items, source_sha256)

    @staticmethod
    def _parse_record(record: object, position: int) -> CatalogItem:
        if not isinstance(record, dict):
            raise ValueError(f"catalog record {position} must be an object")
        required = {
            "entity_id",
            "name",
            "link",
            "job_levels",
            "languages",
            "duration",
            "adaptive",
            "description",
            "keys",
        }
        missing = required - record.keys()
        if missing:
            raise ValueError(f"catalog record {position} missing {sorted(missing)}")

        entity_id = str(record["entity_id"]).strip()
        name = str(record["name"]).strip()
        link = canonicalize_url(str(record["link"]))
        description = " ".join(str(record["description"]).split())
        keys = tuple(str(value).strip() for value in record["keys"])
        unknown_keys = set(keys) - KEY_TO_CODE.keys()
        if unknown_keys:
            raise ValueError(f"unknown catalog keys for {entity_id}: {sorted(unknown_keys)}")
        if not entity_id or not name or not description or not keys:
            raise ValueError(f"empty required catalog value at record {position}")
        if not link.startswith(SHL_URL_PREFIX):
            raise ValueError(f"non-SHL product URL for {entity_id}: {link}")
        if any(ord(character) < 32 for character in name):
            raise ValueError(f"control character in catalog name for {entity_id}")

        test_type = PRIMARY_TYPE_OVERRIDES.get(
            entity_id, ",".join(KEY_TO_CODE[key] for key in keys)
        )
        levels = tuple(str(value).strip() for value in record["job_levels"])
        languages = tuple(str(value).strip() for value in record["languages"])
        search_text = " ".join(
            [
                name,
                name,
                description,
                " ".join(keys),
                " ".join(levels),
                " ".join(languages),
            ]
        )
        return CatalogItem(
            entity_id=entity_id,
            name=name,
            link=link,
            job_levels=levels,
            languages=languages,
            duration=str(record["duration"]).strip(),
            adaptive=str(record["adaptive"]).casefold() == "yes",
            description=description,
            keys=keys,
            test_type=test_type,
            normalized_name=normalize_text(name),
            search_text=normalize_text(search_text),
        )

    def resolve_name(self, value: str) -> CatalogItem | None:
        return self.by_normalized_name.get(normalize_text(value))

    def validate_ids(self, entity_ids: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for entity_id in entity_ids:
            entity_id = str(entity_id)
            if entity_id in self.by_id and entity_id not in seen:
                result.append(entity_id)
                seen.add(entity_id)
        return result

