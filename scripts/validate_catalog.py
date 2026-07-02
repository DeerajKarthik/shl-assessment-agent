from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.catalog import Catalog


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="shl_product_catalog.json")
    parser.add_argument("--expected-count", type=int, default=377)
    args = parser.parse_args()

    catalog = Catalog.load(Path(args.path))
    if len(catalog.items) != args.expected_count:
        raise SystemExit(
            f"expected {args.expected_count} records, found {len(catalog.items)}"
        )
    print(f"valid catalog: {len(catalog.items)} records")
    print(f"sha256: {catalog.source_sha256}")


if __name__ == "__main__":
    main()
