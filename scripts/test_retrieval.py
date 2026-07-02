import asyncio
import json
from pathlib import Path
from app.settings import settings
from app.catalog import Catalog
from app.retrieval import HybridRetriever
from app.policy import build_state

async def test_retrieval():
    catalog = Catalog.load(settings.catalog_path)
    retriever = HybridRetriever(
        catalog=catalog,
        aliases_path=settings.aliases_path,
        ontology_path=settings.ontology_path,
        embeddings_path=settings.embeddings_path,
        embeddings_meta_path=settings.embeddings_meta_path,
    )
    
    from evaluation.traces import load_public_traces
    traces = load_public_traces(Path("GenAI_SampleConversations"))
    
    total_expected = 0
    total_found = 0
    
    from app.schemas import Message
    for trace in traces:
        messages = [Message(role="user", content=turn) for turn in trace.user_turns]
        expected_urls = trace.expected_urls
        
        state = build_state(messages, catalog, retriever.aliases)
        
        # Test just the pure retriever
        candidates = retriever.search(state, query_embedding=None, limit=20)
        candidate_urls = [catalog.by_id[c.entity_id].link for c in candidates]
        
        found = [url for url in expected_urls if url in candidate_urls]
        missing = [url for url in expected_urls if url not in candidate_urls]
        
        total_expected += len(expected_urls)
        total_found += len(found)
        
        if missing:
            print(f"Trace {trace.trace_id} missing from Top 20:")
            for m in missing:
                name = catalog.by_url.get(m).name if m in catalog.by_url else m
                print(f"  - {name} ({m})")
            print(f"  Query was: {state.combined_user_text}")
            print()
            
    print(f"Candidate Recall@20: {total_found / total_expected:.3f}")

if __name__ == "__main__":
    asyncio.run(test_retrieval())
