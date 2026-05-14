from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

from questions_agent_platform.pipeline.db import connect, init_db


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "export_user_coherence_timeseries.py"
SPEC = importlib.util.spec_from_file_location("export_user_coherence_timeseries", SCRIPT)
exporter = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(exporter)


def test_export_user_coherence_timeseries_reads_circle_and_state_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "qa.sqlite"
    init_db(str(db_path))
    with connect(str(db_path)) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO users(user_id, created_at) VALUES (?, ?);",
            ("circle_user", "2026-05-14T00:00:00Z"),
        )
        conn.execute(
            """
            INSERT INTO state_snapshots(
              snapshot_id, user_id, date, timestamp,
              state_schema_version, model_version,
              x_hat_json, x_uncertainty_json, coverage_json,
              source, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                "state-1",
                "circle_user",
                "2026-05-14",
                "2026-05-14T00:00:00Z",
                "state_schema_v1",
                "state_model_v1",
                json.dumps({"sleep_quality": 0.62}),
                json.dumps({"sleep_quality": 0.12}),
                json.dumps({"sleep_quality": 1.0}),
                "questions_agent",
                "2026-05-14T00:00:01Z",
            ),
        )
        conn.execute(
            """
            INSERT INTO circle_snapshots(
              snapshot_id, user_id, date, timestamp, state_snapshot_id,
              projection_version, anchor_version,
              z_json, z_star_json,
              r, theta, velocity, acceleration, coherence_score, coherence_tier_json,
              uncertainty_json, source, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                "circle-1",
                "circle_user",
                "2026-05-14",
                "2026-05-14T00:00:00Z",
                "state-1",
                "circle_projection_v2_canonical",
                "feasible_reference_supercoherence_v1",
                json.dumps([0.2, -0.1]),
                json.dumps([0.1, -0.05]),
                0.25,
                5.8,
                0.04,
                -0.01,
                0.78,
                json.dumps({"tier": "coherent"}),
                json.dumps({"circle_uncertainty": 0.09}),
                "questions_agent",
                "2026-05-14T00:00:02Z",
            ),
        )
        conn.commit()

    manifest = exporter.export_user_coherence_timeseries(
        db_path=str(db_path),
        user_id="circle_user",
        out_dir=str(tmp_path / "export"),
    )

    assert manifest["contract"] == "questions_agent.coherence_timeseries_export.v1"
    assert manifest["row_count"] == 1
    output_csv = Path(manifest["output_csv"])
    assert output_csv.exists()
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["contract"] == "ani.coherence_circle.timeseries_row.v1"
    assert rows[0]["coherence_score_0_1"] == "0.780000"
    identity = json.loads(rows[0]["identity_mask_fragment_json"])
    assert identity["z_t"] == [0.2, -0.1]
    assert identity["state_x_hat"]["sleep_quality"] == 0.62
    assert manifest["source_manifest"][0]["tables"] == ["circle_snapshots", "state_snapshots"]
