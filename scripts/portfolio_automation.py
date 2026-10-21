#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from app.portfolio_audit import audit_repositories, write_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit non-TraceOps repositories and generate deployment dashboard artifacts."
    )
    parser.add_argument("--owner", default="yashh-stack", help="GitHub owner/user to scan")
    parser.add_argument(
        "--output-dir",
        default="docs/portfolio",
        help="Output directory for generated registry JSON and dashboard markdown",
    )
    parser.add_argument(
        "--probe-live-urls",
        action="store_true",
        help="Probe discovered live URLs and include reachability results",
    )
    parser.add_argument(
        "--token-env",
        default="GITHUB_TOKEN",
        help="Environment variable name carrying a GitHub token (optional)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.getenv(args.token_env)

    records = audit_repositories(
        owner=args.owner,
        token=token,
        probe_live_urls=args.probe_live_urls,
    )

    output_dir = Path(args.output_dir)
    json_path, markdown_path = write_outputs(records, output_dir)

    print(f"Scanned {len(records)} repositories (TraceOps repos excluded).")
    print(f"Registry: {json_path}")
    print(f"Dashboard: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
