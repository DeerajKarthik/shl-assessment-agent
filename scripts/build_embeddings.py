from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from app.catalog import Catalog


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="shl_product_catalog.json")
    parser.add_argument("--output", default="data/catalog_embeddings.npy")
    parser.add_argument("--metadata", default="data/catalog_embeddings.meta.json")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--dimension", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit("install sentence-transformers first") from exc

    catalog = Catalog.load(Path(args.catalog))
    
    print(f"Loading model {args.model}...")
    model = SentenceTransformer(args.model, trust_remote_code=True)
    
    vectors: list[list[float]] = []
    total = len(catalog.items)
    batch_size = args.batch_size

    for start in range(0, total, batch_size):
        batch = catalog.items[start : start + batch_size]
        batch_texts = []
        for item in batch:
            if "nomic" in args.model.lower():
                batch_texts.append("search_document: " + item.search_text)
            else:
                batch_texts.append(item.search_text)

        embeddings = model.encode(batch_texts, normalize_embeddings=True)
        vectors.extend(embeddings.tolist())
        
        done = min(start + len(batch), total)
        print(f"embedded {done}/{total}")

    # Final validation
    if len(vectors) != total:
        raise SystemExit(
            f"Final vector count {len(vectors)} does not match catalog size {total}"
        )

    matrix = np.asarray(vectors, dtype=np.float32)
    assert matrix.shape == (total, args.dimension), (
        f"Matrix shape {matrix.shape} does not match expected ({total}, {args.dimension})"
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, matrix)
    metadata = {
        "catalog_sha256": catalog.source_sha256,
        "model": args.model,
        "dimension": args.dimension,
        "rows": total,
        "matrix_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    Path(args.metadata).write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"\n✓ Successfully embedded {total} items")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
