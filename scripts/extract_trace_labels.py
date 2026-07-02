from __future__ import annotations

import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.traces import load_public_traces


def main() -> None:
    traces = load_public_traces(Path("GenAI_SampleConversations"))
    payload = {
        trace.trace_id: {
            "user_turns": list(trace.user_turns),
            "expected_urls": list(trace.expected_urls),
        }
        for trace in traces
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
