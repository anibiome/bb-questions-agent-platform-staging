#!/usr/bin/env python3
"""Export source-linked user Coherence Circle timeseries rows.

This is the bridge from Questions Agent persistence to the production
Coherence Circle / Identity Mask packet. It reads only persisted snapshot
tables and emits an auditable CSV plus manifest JSON.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


CONTRACT = "questions_agent.coherence_timeseries_export.v1"
ROW_CONTRACT = "ani.coherence_circle.timeseries_row.v1"

FIELDNAMES = (
    "contract",
    "user_id",
    "date",
    "timestamp",
    "circle_snapshot_id",
    "state_snapshot_id",
    "projection_version",
    "anchor_version",
    "z_json",
    "z_star_json",
    "decoherence_radius",
    "theta",
    "velocity",
    "acceleration",
    "coherence_score_0_1",
    "coherence_tier_json",
    "uncertainty_json",
    "state_x_hat_json",
    "state_uncertainty_json",
    "identity_mask_fragment_json",
    "source",
    "created_at",
)


def export_user_coherence_timeseries(
    *,
    db_path: str,
    user_id: str,
    out_dir: str,
    since: str | None = None,
    until: str | None = None,
) -> dict[str, Any]:
    db = Path(db_path).expanduser().resolve()
    if not db.exists():
        raise FileNotFoundError(f"Questions Agent database not found: {db}")
    out_root = Path(out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(db, user_id=user_id, since=since, until=until)
    normalized = [_normalize_row(row) for row in rows]
    csv_path = out_root / f"{_safe_slug(user_id)}_coherence_timeseries.csv"
    _write_csv(csv_path, normalized)
    source_manifest = _source_manifest(db, csv_path, normalized)
    manifest_path = out_root / f"{_safe_slug(user_id)}_coherence_timeseries_manifest.json"
    manifest = {
        "contract": CONTRACT,
        "generated_at": _now_iso(),
        "user_id": user_id,
        "db_path": str(db),
        "since": since,
        "until": until,
        "row_count": len(normalized),
        "output_csv": str(csv_path),
        "output_csv_sha256": _sha256(csv_path),
        "repo_commit": _commit_sha(Path(__file__).resolve().parents[1]),
        "source_manifest": source_manifest,
        "blocked_if_empty": (
            "No production Coherence Circle claim should be emitted for this user "
            "until at least one persisted circle_snapshot row exists."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest["manifest_json"] = str(manifest_path)
    manifest["manifest_sha256"] = _sha256(manifest_path)
    return manifest


def _load_rows(
    db: Path,
    *,
    user_id: str,
    since: str | None,
    until: str | None,
) -> list[sqlite3.Row]:
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        filters = ["c.user_id = ?"]
        params: list[Any] = [user_id]
        if since:
            filters.append("c.date >= ?")
            params.append(since)
        if until:
            filters.append("c.date <= ?")
            params.append(until)
        where = " AND ".join(filters)
        return list(
            conn.execute(
                f"""
                SELECT
                  c.snapshot_id AS circle_snapshot_id,
                  c.user_id,
                  c.date,
                  c.timestamp,
                  c.state_snapshot_id,
                  c.projection_version,
                  c.anchor_version,
                  c.z_json,
                  c.z_star_json,
                  c.r,
                  c.theta,
                  c.velocity,
                  c.acceleration,
                  c.coherence_score,
                  c.coherence_tier_json,
                  c.uncertainty_json,
                  c.source,
                  c.created_at,
                  s.x_hat_json AS state_x_hat_json,
                  s.x_uncertainty_json AS state_uncertainty_json
                FROM circle_snapshots c
                LEFT JOIN state_snapshots s
                  ON s.snapshot_id = c.state_snapshot_id
                WHERE {where}
                ORDER BY c.date ASC, c.timestamp ASC, c.created_at ASC;
                """,
                params,
            )
        )
    finally:
        conn.close()


def _normalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(row)
    coherence_score = _float_or_none(row.get("coherence_score"))
    radius = _float_or_none(row.get("r"))
    if coherence_score is None and radius is not None:
        coherence_score = max(0.0, min(1.0, 1.0 - radius))
    z = _json(row.get("z_json"), fallback=[])
    z_star = _json(row.get("z_star_json"), fallback=[])
    identity_mask = {
        "contract": "questions_agent.identity_mask_fragment.v1",
        "source": "circle_snapshots+state_snapshots",
        "z_t": z,
        "z_star_t": z_star,
        "decoherence_radius": radius,
        "coherence_score_0_1": coherence_score,
        "theta_t": _float_or_none(row.get("theta")),
        "velocity": _float_or_none(row.get("velocity")),
        "acceleration": _float_or_none(row.get("acceleration")),
        "state_x_hat": _json(row.get("state_x_hat_json"), fallback={}),
        "state_uncertainty": _json(row.get("state_uncertainty_json"), fallback={}),
    }
    return {
        "contract": ROW_CONTRACT,
        "user_id": str(row.get("user_id") or ""),
        "date": str(row.get("date") or ""),
        "timestamp": str(row.get("timestamp") or ""),
        "circle_snapshot_id": str(row.get("circle_snapshot_id") or ""),
        "state_snapshot_id": str(row.get("state_snapshot_id") or ""),
        "projection_version": str(row.get("projection_version") or ""),
        "anchor_version": str(row.get("anchor_version") or ""),
        "z_json": _json_dump(z),
        "z_star_json": _json_dump(z_star),
        "decoherence_radius": "" if radius is None else f"{radius:.6f}",
        "theta": _format_float(row.get("theta")),
        "velocity": _format_float(row.get("velocity")),
        "acceleration": _format_float(row.get("acceleration")),
        "coherence_score_0_1": "" if coherence_score is None else f"{coherence_score:.6f}",
        "coherence_tier_json": _json_dump(_json(row.get("coherence_tier_json"), fallback={})),
        "uncertainty_json": _json_dump(_json(row.get("uncertainty_json"), fallback={})),
        "state_x_hat_json": _json_dump(_json(row.get("state_x_hat_json"), fallback={})),
        "state_uncertainty_json": _json_dump(_json(row.get("state_uncertainty_json"), fallback={})),
        "identity_mask_fragment_json": _json_dump(identity_mask),
        "source": str(row.get("source") or "questions_agent"),
        "created_at": str(row.get("created_at") or ""),
    }


def _source_manifest(db: Path, csv_path: Path, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "artifact": str(db),
            "artifact_type": "sqlite_database",
            "sha256": _sha256(db),
            "tables": ["circle_snapshots", "state_snapshots"],
        },
        {
            "artifact": str(csv_path),
            "artifact_type": "coherence_timeseries_csv",
            "sha256": _sha256(csv_path),
            "row_count": len(rows),
        },
    ]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _json(value: Any, *, fallback: Any) -> Any:
    if value in (None, ""):
        return fallback
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return fallback


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_float(value: Any) -> str:
    number = _float_or_none(value)
    return "" if number is None else f"{number:.6f}"


def _safe_slug(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return cleaned[:96] or "unknown_user"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _commit_sha(repo_root: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 and proc.stdout.strip() else "unknown"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export user Coherence Circle timeseries rows.")
    parser.add_argument("--db", required=True, help="Questions Agent SQLite database path")
    parser.add_argument("--user-id", required=True, help="User id to export")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--since", default=None, help="Inclusive YYYY-MM-DD lower bound")
    parser.add_argument("--until", default=None, help="Inclusive YYYY-MM-DD upper bound")
    args = parser.parse_args()
    manifest = export_user_coherence_timeseries(
        db_path=args.db,
        user_id=args.user_id,
        out_dir=args.out_dir,
        since=args.since,
        until=args.until,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
