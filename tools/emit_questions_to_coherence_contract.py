#!/usr/bin/env python3
"""Emit control-plane contract payload: questions_to_coherence.v1."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path


def _commit_sha(repo_root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            return proc.stdout.strip() or "unknown"
    except Exception:
        pass
    return "unknown"


def build_payload(repo_root: Path) -> dict:
    return {
        "contract": "questions_to_coherence",
        "version": 1,
        "run_id": f"qa-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        "user_id": "demo-user-1",
        "date": date.today().isoformat(),
        "coherence": {
            "r": 0.37,
            "theta": 1.24,
            "velocity": -0.08,
            "uncertainty": 0.11,
        },
        "identity_mask": {
            "mu": [0.14, -0.08, 0.03, 0.27],
            "sigma": [0.10, 0.09, 0.12, 0.07],
            "constraints": {
                "gcv_available": False,
                "safety_guard": "standard",
            },
        },
        "metadata": {
            "repo": "questions-agent-platform",
            "commit_sha": _commit_sha(repo_root),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit questions_to_coherence payload")
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_payload(repo_root), indent=2), encoding="utf-8")
    print(str(out))


if __name__ == "__main__":
    main()
