from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]


def _path_from_env(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default))
    return value if value.is_absolute() else ROOT / value


@dataclass(frozen=True, slots=True)
class Settings:
    catalog_path: Path = _path_from_env("CATALOG_PATH", "data/shl_product_catalog.json")
    aliases_path: Path = _path_from_env("ALIASES_PATH", "data/aliases.json")
    ontology_path: Path = _path_from_env("ONTOLOGY_PATH", "data/ontology.json")
    dependencies_path: Path = _path_from_env("DEPENDENCIES_PATH", "data/dependencies.json")
    embeddings_path: Path = _path_from_env("EMBEDDINGS_PATH", "data/catalog_embeddings.npy")
    embeddings_meta_path: Path = _path_from_env(
        "EMBEDDINGS_META_PATH", "data/catalog_embeddings.meta.json"
    )
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY") or None
    groq_api_key: str | None = os.getenv("GROQ_API_KEY") or None
    cerebras_api_key: str | None = os.getenv("CEREBRAS_API_KEY") or None
    mistral_api_key: str | None = os.getenv("MISTRAL_API_KEY") or None
    model_name: str = os.getenv("MODEL_NAME", "mistral-medium-latest")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    gemini_timeout_seconds: float = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "28.0"))
    application_timeout_seconds: float = float(
        os.getenv("APPLICATION_TIMEOUT_SECONDS", "30")
    )
    candidate_limit: int = int(os.getenv("CANDIDATE_LIMIT", "20"))


settings = Settings()
