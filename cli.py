from __future__ import annotations

import argparse
import json

from .cortex import Cortex


def main() -> None:
    parser = argparse.ArgumentParser(prog="cortex", description="Cortex CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest a message")
    p_ingest.add_argument("--user", required=True)
    p_ingest.add_argument(
        "--role", required=True, choices=["user", "assistant", "system", "event"]
    )
    p_ingest.add_argument("--message", required=True)

    args = parser.parse_args()

    if args.cmd == "ingest":
        cx = Cortex(args.user)
        res = cx.ingest(role=args.role, message=args.message)
        print(
            json.dumps(
                {
                    "episode_id": res.episode_id,
                    "tokens": res.tokens,
                    "classification": {
                        "label": res.classification.label,
                        "confidence": res.classification.confidence,
                        "data": res.classification.data,
                    },
                    "pii_masked": res.pii_masked,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
