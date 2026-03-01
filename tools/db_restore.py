from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from questions_agent_platform.tools.db_backup_restore import restore_backup


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Restore Questions Agent DB from backup artifact.")
    parser.add_argument("--backup", type=str, required=True, help="Path to backup artifact")
    parser.add_argument("--database-url", type=str, default=None, help="Target DATABASE_URL (defaults to prod settings env)")
    parser.add_argument("--clean", action="store_true", help="Use destructive clean restore for supported engines")
    parser.add_argument("--out-json", type=str, default=None)
    args = parser.parse_args(argv)

    database_url = args.database_url
    if not database_url:
        from questions_agent_platform.prod.settings import load_settings

        database_url = load_settings().database_url

    payload = restore_backup(
        backup_path=str(args.backup),
        target_database_url=str(database_url),
        clean=bool(args.clean),
    )
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    print(encoded)
    if args.out_json:
        out_path = Path(args.out_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(encoded, encoding="utf-8")
        print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
