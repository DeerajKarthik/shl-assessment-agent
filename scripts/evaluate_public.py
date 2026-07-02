from __future__ import annotations

import asyncio
import json
from pathlib import Path
import statistics
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from app.catalog import canonicalize_url, Catalog
from app.main import app
from app.schemas import ChatRequest, ChatResponse
from app.service import RecommenderService
from app.settings import settings
from evaluation.traces import load_public_traces, recall_at_10


def _schema_valid_json(payload: dict) -> bool:
    """Check hard-eval schema requirements on raw HTTP JSON, not Pydantic objects."""
    if not isinstance(payload.get("reply"), str) or len(payload.get("reply", "")) == 0:
        return False
    recs = payload.get("recommendations")
    if not isinstance(recs, list):
        return False
    if not isinstance(payload.get("end_of_conversation"), bool):
        return False
    if len(recs) > 10:
        return False
    for rec in recs:
        if not isinstance(rec, dict):
            return False
        if not all(k in rec for k in ("name", "url", "test_type")):
            return False
    # No extra keys allowed
    allowed_keys = {"reply", "recommendations", "end_of_conversation"}
    if set(payload.keys()) != allowed_keys:
        return False
    return True


def _catalog_membership_json(recs: list[dict], catalog_urls: set[str]) -> bool:
    """Every recommended URL must exist in the catalog."""
    return all(
        canonicalize_url(rec.get("url", "")) in catalog_urls for rec in recs
    )


def _has_url_in_reply(reply: str) -> bool:
    return "http://" in reply or "https://" in reply


def _hallucinated_names(recs: list[dict], catalog_names: set[str]) -> list[str]:
    return [rec["name"] for rec in recs if rec.get("name") not in catalog_names]


def _duplicate_urls(recs: list[dict]) -> bool:
    urls = [rec.get("url", "") for rec in recs]
    return len(urls) != len(set(urls))


def _candidate_recall(
    candidates: list[str], expected_urls: tuple[str, ...], k: int
) -> float:
    """Candidate recall from the retriever's full candidate set, not just emitted recs."""
    if not expected_urls:
        return 1.0
    candidate_set = set(candidates[:k])
    return len(candidate_set & set(expected_urls)) / len(set(expected_urls))


async def evaluate() -> dict[str, object]:
    service = RecommenderService(settings)
    app.state.recommender = service
    traces = load_public_traces(Path("GenAI_SampleConversations"))
    catalog_urls = set(service.catalog.by_url.keys())
    catalog_names = {item.name for item in service.catalog.items}
    rows: list[dict[str, object]] = []
    latencies: list[float] = []

    # Aggregate metric counters
    total_responses = 0
    schema_passes = 0
    catalog_membership_passes = 0
    url_in_reply_count = 0
    hallucination_count = 0
    duplicate_count = 0
    turn_cap_violations = 0

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for trace in traces:
            messages: list[dict[str, str]] = []
            first_urls: list[str] = []
            last_urls: list[str] = []
            all_candidate_urls: list[str] = []
            trace_schema_pass = True
            trace_catalog_pass = True
            trace_hallucinated: list[str] = []
            turn_count = 0

            for user_turn in trace.user_turns:
                if len(messages) >= 7:
                    break
                messages.append({"role": "user", "content": user_turn})
                started = time.perf_counter()
                # Sleep for 3s to respect Mistral Free Tier limits
                await asyncio.sleep(3.0)
                # Evaluate via HTTP to test the actual JSON contract
                http_response = await client.post("/chat", json={"messages": messages})
                latencies.append((time.perf_counter() - started) * 1000)
                total_responses += 1
                turn_count += 1

                assert http_response.status_code == 200, f"HTTP {http_response.status_code}"
                payload = http_response.json()

                # Schema validation on raw JSON
                if _schema_valid_json(payload):
                    schema_passes += 1
                else:
                    trace_schema_pass = False

                recs = payload.get("recommendations", [])

                # Catalog membership
                if _catalog_membership_json(recs, catalog_urls):
                    catalog_membership_passes += 1
                else:
                    trace_catalog_pass = False

                # URL in reply
                if _has_url_in_reply(payload.get("reply", "")):
                    url_in_reply_count += 1

                # Hallucination check
                hallucinated = _hallucinated_names(recs, catalog_names)
                trace_hallucinated.extend(hallucinated)
                hallucination_count += len(hallucinated)

                # Duplicate check
                if _duplicate_urls(recs):
                    duplicate_count += 1

                urls = [rec.get("url", "") for rec in recs]
                all_candidate_urls.extend(urls)
                if urls and not first_urls:
                    first_urls = urls
                if urls:
                    last_urls = urls
                if len(messages) < 7:
                    messages.append({"role": "assistant", "content": payload.get("reply", "")})

            # Turn cap check: did we exceed 4 exchanges (7 messages)?
            if len(messages) > 7:
                turn_cap_violations += 1

            # Compacted final-state
            compacted_messages = [
                {
                    "role": "user",
                    "content": "\n\nEarlier requirement:\n".join(trace.user_turns[:-1])
                    or trace.user_turns[-1],
                }
            ]
            if len(trace.user_turns) > 1:
                compacted_messages.extend(
                    [
                        {"role": "assistant", "content": "I have retained those requirements."},
                        {"role": "user", "content": trace.user_turns[-1]},
                    ]
                )
            compacted_resp = await client.post("/chat", json={"messages": compacted_messages})
            compacted_payload = compacted_resp.json()
            compacted_urls = [rec.get("url", "") for rec in compacted_payload.get("recommendations", [])]

            all_unique = list(dict.fromkeys(all_candidate_urls))
            first_recall = recall_at_10(first_urls, trace.expected_urls)
            within_cap_recall = recall_at_10(last_urls, trace.expected_urls)
            compacted_recall = recall_at_10(compacted_urls, trace.expected_urls)

            if first_recall < 1.0:
                expected_set = set(trace.expected_urls)
                missing = expected_set - set(first_urls[:10])
                in_candidates = missing & set(all_unique)
                retrieval_miss = missing - set(all_unique)
                failure_type = (
                    "retrieval" if retrieval_miss else
                    "ranking" if in_candidates else
                    "policy"
                )
            else:
                failure_type = "none"

            rows.append(
                {
                    "trace": trace.trace_id,
                    "first_commit_recall_at_10": first_recall,
                    "within_cap_recall_at_10": within_cap_recall,
                    "compacted_final_state_recall_at_10": compacted_recall,
                    "schema_valid": trace_schema_pass,
                    "catalog_backed": trace_catalog_pass,
                    "hallucinated_names": trace_hallucinated,
                    "failure_attribution": failure_type,
                    "first_urls": first_urls,
                    "within_cap_urls": last_urls,
                    "expected_urls": list(trace.expected_urls),
                }
            )

    result = {
        "catalog_sha256": service.catalog.source_sha256,
        "model_enabled": service.gemini.enabled,
        "dense_retrieval_enabled": service.retriever.embedding_matrix is not None,
        "mean_first_commit_recall_at_10": statistics.mean(
            row["first_commit_recall_at_10"] for row in rows
        ),
        "mean_within_cap_recall_at_10": statistics.mean(
            row["within_cap_recall_at_10"] for row in rows
        ),
        "mean_compacted_final_state_recall_at_10": statistics.mean(
            row["compacted_final_state_recall_at_10"] for row in rows
        ),
        "schema_pass_rate": schema_passes / max(total_responses, 1),
        "catalog_membership_pass_rate": catalog_membership_passes / max(total_responses, 1),
        "turn_cap_violations": turn_cap_violations,
        "url_in_reply_count": url_in_reply_count,
        "hallucinated_recommendation_count": hallucination_count,
        "duplicate_recommendation_count": duplicate_count,
        "latency_ms": {
            "mean": statistics.mean(latencies),
            "p50": statistics.median(latencies),
            "p95": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
            "p99": sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0,
            "max": max(latencies),
        },
        "total_responses": total_responses,
        "traces": rows,
    }
    return result


def main() -> None:
    print(json.dumps(asyncio.run(evaluate()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
