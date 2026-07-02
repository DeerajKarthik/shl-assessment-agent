from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.catalog import Catalog


def test_catalog_is_strict_and_complete(service) -> None:
    assert len(service.catalog.items) == 377
    assert len(service.catalog.by_id) == 377
    assert len(service.catalog.by_url) == 377
    assert service.catalog.by_id["4207"].name == "Microsoft Excel 365 (New)"


def test_original_control_character_fixture_is_rejected(tmp_path: Path) -> None:
    malformed = (
        '[{"entity_id":"4207","name":"Microsoft \n 365 (New)",'
        '"link":"https://www.shl.com/products/product-catalog/view/microsoft-excel-365-new/"}]'
    )
    path = tmp_path / "bad.json"
    path.write_text(malformed)
    with pytest.raises(json.JSONDecodeError):
        Catalog.load(path)


def test_every_public_recommendation_is_catalog_backed(service) -> None:
    for item in service.catalog.items:
        recommendation = item.recommendation()
        assert recommendation.url == item.link
        assert recommendation.name == item.name
        assert recommendation.test_type


def test_alias_resolution(service) -> None:
    """Verify that common aliases resolve to the correct entity ID."""
    aliases = service.aliases
    assert "opq" in aliases
    assert aliases["opq"] == "720"
    assert aliases["g+"] == "3971"
    assert aliases["aws"] == "4028"
    assert aliases["docker"] == "4059"
    assert aliases["rest"] == "4126"


def test_normalize_text_behavior() -> None:
    from app.catalog import normalize_text
    
    assert normalize_text("OPQ32r") == "opq32r"
    assert normalize_text("C++") == "c++"
    assert normalize_text("C#") == "c#"
    assert normalize_text(".NET") == ".net"
    assert normalize_text("G+") == "g+"
    assert normalize_text("Problem-solving") == "problem solving"
    assert normalize_text("Health & Safety") == "health and safety"
    assert normalize_text("A, B, and C") == "a b and c"
