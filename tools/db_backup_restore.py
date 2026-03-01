from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from sqlalchemy import MetaData, func, select
from sqlalchemy.engine import make_url
from sqlalchemy.engine.url import URL
from sqlalchemy import create_engine


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_url(database_url: str) -> URL:
    if not database_url:
        raise ValueError("database_url is required")
    return make_url(str(database_url))


def detect_database_kind(database_url: str) -> str:
    url = _normalize_url(database_url)
    backend = str(url.get_backend_name() or "").lower()
    if backend == "sqlite":
        return "sqlite"
    if backend in {"postgresql", "postgresql+psycopg2", "postgresql+asyncpg"}:
        return "postgres"
    return backend or "unknown"


def _sqlite_path_from_url(database_url: str) -> Path:
    url = _normalize_url(database_url)
    if str(url.get_backend_name() or "").lower() != "sqlite":
        raise ValueError("database_url is not sqlite")
    db = str(url.database or "").strip()
    if not db:
        raise ValueError("sqlite database path is empty")
    return Path(db).expanduser().resolve()


def collect_row_counts(database_url: str, *, include_tables: Optional[set[str]] = None) -> Dict[str, int]:
    engine = create_engine(str(database_url), pool_pre_ping=True, future=True)
    try:
        metadata = MetaData()
        with engine.connect() as conn:
            metadata.reflect(bind=conn)
            counts: Dict[str, int] = {}
            for table_name in sorted(metadata.tables.keys()):
                if table_name.startswith("sqlite_"):
                    continue
                if include_tables is not None and table_name not in include_tables:
                    continue
                table = metadata.tables[table_name]
                value = conn.execute(select(func.count()).select_from(table)).scalar()
                counts[table_name] = int(value or 0)
            return counts
    finally:
        engine.dispose()


def create_backup(
    *,
    database_url: str,
    out_dir: str,
    backup_name: Optional[str] = None,
) -> Dict[str, Any]:
    kind = detect_database_kind(database_url)
    target_dir = Path(out_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = backup_name or f"qa_backup_{stamp}"

    if kind == "sqlite":
        source = _sqlite_path_from_url(database_url)
        if not source.exists():
            raise FileNotFoundError(f"sqlite database not found: {source}")
        backup_path = target_dir / f"{name}.sqlite3"
        shutil.copy2(source, backup_path)
        return {
            "kind": kind,
            "source": str(source),
            "backup_path": str(backup_path),
            "size_bytes": int(backup_path.stat().st_size),
        }

    if kind == "postgres":
        backup_path = target_dir / f"{name}.dump"
        cmd = [
            "pg_dump",
            "--format=custom",
            "--no-owner",
            "--no-privileges",
            "--file",
            str(backup_path),
            str(database_url),
        ]
        subprocess.run(cmd, check=True)
        return {
            "kind": kind,
            "backup_path": str(backup_path),
            "size_bytes": int(backup_path.stat().st_size),
            "command": cmd,
        }

    raise ValueError(f"Unsupported database backend for backup: {kind}")


def restore_backup(
    *,
    backup_path: str,
    target_database_url: str,
    clean: bool = True,
) -> Dict[str, Any]:
    path = Path(backup_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"backup not found: {path}")
    kind = detect_database_kind(target_database_url)

    if kind == "sqlite":
        target = _sqlite_path_from_url(target_database_url)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        return {
            "kind": kind,
            "backup_path": str(path),
            "target": str(target),
            "size_bytes": int(target.stat().st_size),
        }

    if kind == "postgres":
        cmd = [
            "pg_restore",
            "--no-owner",
            "--no-privileges",
            "--dbname",
            str(target_database_url),
        ]
        if clean:
            cmd.extend(["--clean", "--if-exists"])
        cmd.append(str(path))
        subprocess.run(cmd, check=True)
        return {
            "kind": kind,
            "backup_path": str(path),
            "target_database_url": str(target_database_url),
            "command": cmd,
        }

    raise ValueError(f"Unsupported database backend for restore: {kind}")


def run_backup_restore_drill(
    *,
    source_database_url: str,
    out_dir: str,
    restore_database_url: Optional[str] = None,
    fail_on_mismatch: bool = False,
) -> Dict[str, Any]:
    kind = detect_database_kind(source_database_url)
    baseline_counts = collect_row_counts(source_database_url)
    backup = create_backup(database_url=source_database_url, out_dir=out_dir)

    report: Dict[str, Any] = {
        "generated_at_utc": _utc_now_iso(),
        "source_database_url": str(source_database_url),
        "database_kind": kind,
        "backup": backup,
        "baseline_row_counts": baseline_counts,
        "restore_checked": False,
        "row_counts_match": None,
        "status": "backup_only",
    }

    if kind == "sqlite":
        source_path = _sqlite_path_from_url(source_database_url)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        drill_db_path = Path(out_dir).expanduser().resolve() / f"qa_restore_drill_{stamp}.sqlite3"
        shutil.copy2(source_path, drill_db_path)
        with sqlite3.connect(str(drill_db_path)) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS _drill_marker (value TEXT NOT NULL)")
            conn.execute("INSERT INTO _drill_marker(value) VALUES (?)", ("mutated-before-restore",))
            conn.commit()
        mutated_counts = collect_row_counts(f"sqlite+pysqlite:///{drill_db_path}")
        restore = restore_backup(
            backup_path=str(backup["backup_path"]),
            target_database_url=f"sqlite+pysqlite:///{drill_db_path}",
            clean=True,
        )
        restored_counts = collect_row_counts(f"sqlite+pysqlite:///{drill_db_path}")
        row_counts_match = baseline_counts == restored_counts
        report.update(
            {
                "restore_checked": True,
                "drill_restore": restore,
                "mutated_row_counts": mutated_counts,
                "restored_row_counts": restored_counts,
                "row_counts_match": bool(row_counts_match),
                "status": "pass" if row_counts_match else "fail",
            }
        )
        if fail_on_mismatch and not row_counts_match:
            raise RuntimeError("Backup/restore drill failed: restored row counts do not match baseline")
        return report

    if kind == "postgres":
        if not restore_database_url:
            report["status"] = "backup_only_no_restore_target"
            return report
        restore = restore_backup(
            backup_path=str(backup["backup_path"]),
            target_database_url=str(restore_database_url),
            clean=True,
        )
        restored_counts = collect_row_counts(
            str(restore_database_url),
            include_tables=set(baseline_counts.keys()),
        )
        row_counts_match = baseline_counts == restored_counts
        report.update(
            {
                "restore_checked": True,
                "drill_restore": restore,
                "restored_row_counts": restored_counts,
                "row_counts_match": bool(row_counts_match),
                "status": "pass" if row_counts_match else "fail",
            }
        )
        if fail_on_mismatch and not row_counts_match:
            raise RuntimeError("Backup/restore drill failed: restored row counts do not match baseline")
        return report

    raise ValueError(f"Unsupported database backend for backup/restore drill: {kind}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run Questions Agent DB backup/restore drill.")
    parser.add_argument("--database-url", type=str, default=None, help="Source DATABASE_URL (defaults to prod settings env)")
    parser.add_argument(
        "--restore-database-url",
        type=str,
        default=None,
        help="Optional restore DATABASE_URL target (required for postgres restore verification).",
    )
    parser.add_argument("--out-dir", type=str, default="questions_agent_platform/output/drills")
    parser.add_argument("--out-json", type=str, default=None, help="Optional JSON output path.")
    parser.add_argument("--fail-on-mismatch", action="store_true")
    args = parser.parse_args(argv)

    database_url = args.database_url
    if not database_url:
        from questions_agent_platform.prod.settings import load_settings

        database_url = load_settings().database_url

    report = run_backup_restore_drill(
        source_database_url=str(database_url),
        out_dir=str(args.out_dir),
        restore_database_url=args.restore_database_url,
        fail_on_mismatch=bool(args.fail_on_mismatch),
    )
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    print(encoded)
    if args.out_json:
        out_path = Path(args.out_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(encoded, encoding="utf-8")
        print(str(out_path))
    if bool(args.fail_on_mismatch) and report.get("status") == "fail":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
