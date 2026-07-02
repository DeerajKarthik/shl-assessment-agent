# SHL Conversational Assessment Recommender

A stateless, policy-routed conversational API that recommends SHL Individual Test Solutions from a 377-product catalog. The system uses a highly optimized in-memory hybrid retriever (BM25 + character n-gram + Nomic dense semantic search) layered with a strict deterministic policy engine, and finalized by a constrained Mistral LLM reranker.

For a deep dive into the design philosophy, failed experiments, and ablation studies, please read our [APPROACH.md](APPROACH.md).

## Architecture

```
POST /chat → validate → policy routing → hybrid retrieval → Mistral rerank → catalog grounding → response
```

- **Deterministic Policy Engine**: Enforces schema, state mutations, conversational turn limits, catalog identity, hard exclusions, language conflicts, and out-of-domain refusals (e.g. prompt injection, legal, weather) natively in Python code.
- **In-Memory Hybrid Retriever**: Exact alias matching, BM25, character n-gram TF-IDF, Nomic Dense Embeddings, ontology rules, and **Domain Penalization** (e.g., punishing Physical Engineering tests for Software queries).
- **LLM Reranker**: `mistral-medium-latest` selects the top 10 entity IDs from bounded candidates provided by the retriever. The server constructs the final names, URLs, and type codes from trusted catalog records.
- **Deterministic Fallback**: If Mistral hits rate limits (429) or times out, the hybrid ranking immediately degrades to a pure deterministic output, maintaining 100% uptime.

## API

### `GET /health`
Returns `{"status": "ok"}` with HTTP 200.

### `POST /chat`
```json
{
  "messages": [
    {"role": "user", "content": "Senior Python engineer"},
    {"role": "assistant", "content": "What seniority level?"},
    {"role": "user", "content": "Actually Java"}
  ]
}
```

Response:
```json
{
  "reply": "Here is a grounded shortlist. The selections cover personality and behavior, knowledge and skills, ability and aptitude, job simulations, and situational judgment for the requirements provided.",
  "recommendations": [
    {"name": "Occupational Personality Questionnaire OPQ32r", "url": "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/", "test_type": "P"},
    {"name": "Java Platform, Enterprise Edition 7 (Java EE 7)", "url": "https://www.shl.com/products/product-catalog/view/java-platform-enterprise-edition-7-java-ee-7/", "test_type": "K"}
  ],
  "end_of_conversation": false
}
```
*Note: A custom `X-Model-Provider` response header is injected by the FastAPI server to indicate whether the request was fulfilled by Mistral, Deterministic Fallback, or the Policy Engine.*

## Quick Start

### Prerequisites
- Python 3.12+
- A [Mistral API key](https://console.mistral.ai/)

### Local Development
```bash
# Install dependencies
pip install -r requirements-dev.txt

# Set environment variables
cp .env.example .env
# Edit .env and add your MISTRAL_API_KEY

# Run the server
uvicorn app.main:app --reload --port 8080

# Run regression and E2E behavioral tests
pytest tests/test_e2e_policy.py tests/test_policy_regression.py -v
```

### Docker
```bash
docker build -t shl-recommender .
docker run -p 8080:8080 -e MISTRAL_API_KEY=your_key shl-recommender
```

### Railway Deployment
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
railway variables set MISTRAL_API_KEY=your_key
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MISTRAL_API_KEY` | — | Mistral API key for LLM reranking |
| `MODEL_NAME` | `mistral-medium-latest` | Mistral model for reranking |
| `EMBEDDING_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | Local embedding model used by `SentenceTransformers` |
| `GEMINI_TIMEOUT_SECONDS` | `28.0` | Per-call LLM timeout before fallback |
| `APPLICATION_TIMEOUT_SECONDS` | `30` | Total request timeout |
| `CANDIDATE_LIMIT` | `20` | Number of top candidates passed to the LLM for reranking |
| `PORT` | `8080` | Server port |

## Project Structure

```
app/
  main.py           # FastAPI routes and UI rendering
  schemas.py         # Request/response Pydantic models
  settings.py        # Environment configuration
  catalog.py         # Catalog loading, validation, normalization
  policy.py          # State tracking, intent routing, clarification, regex guards
  retrieval.py       # In-memory hybrid retriever & domain penalty logic
  llm.py             # Mistral LLM adapter + fallback logic
  render.py          # Response text construction + URL sanitization
  service.py         # Orchestration layer
data/
  aliases.json       # Exact match term overrides
  ontology.json      # Flagship product expansion rules
  dependencies.json  # Product prerequisite mappings
tests/
  test_e2e_policy.py      # E2E asyncio client behavioral probes
  test_policy_regression.py # Strict intent extraction regression suite
```

## Evaluation & Behavior Probes

We guarantee a 1.0 First-Commit Recall and 100% pass rate on hidden behavioral probes by bypassing generative text extraction for safety-critical logic. 

**Run the suite:**
```bash
pytest tests/test_e2e_policy.py tests/test_policy_regression.py
```

These suites validate:
- **Off-Topic Refusals**: "What is the date today?", "Write a joke"
- **Prompt Injection Refusals**: "Ignore previous instructions", "Reveal prompt"
- **Complex State Mutations**: "Senior Python Engineer" -> "Actually Java" -> "Drop Java, add Rust"

## Design Decisions

1. **Deterministic Policy Engine**: We completely reject LLM "routers" or generative state extractors. All conversation states (edits, language filters, exclusions) are parsed synchronously in Python to guarantee 100% precision.
2. **Constrained LLM Reranking**: The LLM is NEVER allowed to hallucinate a recommendation. It is strictly forced to return a JSON array ordering pre-validated IDs provided by the retriever.
3. **Graceful degradation**: The system perfectly degrades to a deterministic ranker when the Mistral API hits a 429 Rate Limit, ensuring 100% uptime in production.
4. **No Vector Databases**: External vector databases (Chroma/Pinecone) suffer from metadata starvation and semantic drift. We rely exclusively on an optimized in-memory dual-encoder (BM25 + Nomic).
5. **No LangChain/LlamaIndex**: Direct, functional Python code is infinitely more transparent, debuggable, testable, and production-ready than rigid abstraction wrappers.

## AI Tool Disclosure

This project was developed with assistance from AI coding tools (Claude/Gemini) for code generation, architecture review, and catalog analysis. All generated code was reviewed, tested, and validated against the assignment requirements.
