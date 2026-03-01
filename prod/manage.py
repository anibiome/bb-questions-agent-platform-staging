from __future__ import annotations

import argparse

from questions_agent_platform.prod.migrations import ensure_schema_ready
from questions_agent_platform.prod.settings import load_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Questions Agent production DB management")
    sub = parser.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("upgrade", help="Upgrade DB schema to Alembic head")
    up.add_argument("--baseline-existing", choices=("true", "false"), default="true")

    req = sub.add_parser("require", help="Verify DB schema is at Alembic head")
    req.add_argument("--baseline-existing", choices=("true", "false"), default="false")

    off = sub.add_parser("off", help="Skip schema checks (diagnostic only)")
    off.add_argument("--baseline-existing", choices=("true", "false"), default="false")

    args = parser.parse_args()
    settings = load_settings()
    baseline_existing = str(args.baseline_existing).strip().lower() == "true"

    mode = "off"
    if args.cmd == "upgrade":
        mode = "auto"
    elif args.cmd == "require":
        mode = "require"

    result = ensure_schema_ready(
        database_url=settings.database_url,
        mode=mode,
        baseline_existing=baseline_existing,
    )
    print(result)


if __name__ == "__main__":
    main()
