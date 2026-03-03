import shutil
from pathlib import Path
from datetime import date, timedelta
import random
import json
from dataclasses import replace

from questions_agent_platform.pipeline.config import QuestionsAgentConfig
from questions_agent_platform.pipeline.db import connect, init_db
from questions_agent_platform.pipeline.demo_signals import demo_value_for_item
from questions_agent_platform.pipeline.service import (
    compute_user_scale_progress,
    ensure_registry_active,
    get_or_create_daily_session,
    submit_answers,
    take_next_extra_batch,
)
from questions_agent_platform.pipeline.registry import load_registry
from questions_agent_platform.pipeline.dashboard import render_user


def seed_demo_registry(registry_root: str) -> None:
    src_root = Path(__file__).resolve().parents[1] / "data" / "registry_demo"
    versions_dst = Path(registry_root) / "versions"
    if versions_dst.exists():
        shutil.rmtree(versions_dst)
    versions_dst.mkdir(parents=True, exist_ok=True)

    copied = []
    for src in sorted(src_root.glob("v*")):
        if not src.is_dir():
            continue
        dst = versions_dst / src.name
        shutil.copytree(src, dst)
        copied.append(src.name)
    if not copied:
        raise ValueError(f"No demo registry versions found in {src_root}")
    print(f"Seeded demo registry versions to: {versions_dst} ({', '.join(copied)})")


def run_demo_simulation(cfg: QuestionsAgentConfig) -> None:
    """
    Simulates 30 days of sessions for a single demo user, generating some
    drift patterns (mood -> GI) to exercise unlocks and scoring.
    """
    run_policy_data_collection(
        cfg=cfg,
        days=30,
        users=1,
        selection_mode=str(cfg.policy_default_mode or "policy_shadow"),
        epsilon=float(cfg.policy_epsilon_explore),
        seed=42,
        user_prefix="demo-user",
        reset_db=True,
    )


def run_policy_data_collection(
    *,
    cfg: QuestionsAgentConfig,
    days: int,
    users: int,
    selection_mode: str,
    epsilon: float,
    seed: int,
    user_prefix: str,
    reset_db: bool = False,
) -> None:
    """
    Collect policy logs/outcomes for offline evaluation.

    For valid IPS/DR estimates, prefer selection_mode=policy_live so logged action
    equals served action.
    """
    days = max(1, int(days))
    users = max(1, int(users))
    epsilon = max(0.0, min(1.0, float(epsilon)))
    mode = str(selection_mode).strip().lower()
    if mode not in ("deterministic", "policy_live", "policy_shadow", "policy_offline_replay"):
        raise ValueError(f"Unknown selection mode: {selection_mode}")

    if reset_db:
        init_db(cfg.database_path)

    with connect(cfg.database_path) as conn:
        versions_dir = Path(cfg.registry_root) / "versions"
        if not versions_dir.exists() or not any(versions_dir.iterdir()):
            seed_demo_registry(cfg.registry_root)

        reg_v = ensure_registry_active(conn, cfg.registry_root)
        registry = load_registry(cfg.registry_root, reg_v)
        cfg_eff = replace(cfg, policy_epsilon_explore=epsilon, policy_default_mode=mode)

        start_day = date.today() - timedelta(days=days - 1)
        end_day = start_day + timedelta(days=days - 1)
        total_sessions = 0
        for u in range(users):
            user_id = f"{str(user_prefix).strip() or 'policy-eval-user'}-{u + 1}"
            rng = random.Random(int(seed) + u)
            for i in range(days):
                day = start_day + timedelta(days=i)
                session = get_or_create_daily_session(
                    conn,
                    cfg=cfg_eff,
                    registry_root=cfg.registry_root,
                    user_id=user_id,
                    day=day,
                    selection_mode=mode,
                )
                total_sessions += 1

                core_answers = []
                for item_id in session.core_item_ids:
                    val = demo_value_for_item(tags=registry.items[item_id].tags, day_index=i)
                    core_answers.append(
                        {
                            "client_event_id": f"{user_id}::{day.isoformat()}::core::{item_id}",
                            "item_id": item_id,
                            "value": val,
                            "answered_at": f"{day.isoformat()}T12:00:00Z",
                        }
                    )
                submit_answers(
                    conn,
                    registry_root=cfg.registry_root,
                    user_id=user_id,
                    session_id=session.session_id,
                    answers=core_answers,
                )

                explain = session.selection_explain or {}
                near_unlock = bool(explain.get("near_unlock_scales"))
                want_extra = rng.random() < (0.55 if near_unlock else 0.2)
                extra_used = 0
                while want_extra and extra_used < cfg.extra_batches_max_per_day:
                    batch = take_next_extra_batch(
                        conn,
                        session.session_id,
                        user_id=user_id,
                        extra_batches_max_per_day=cfg.extra_batches_max_per_day,
                    )
                    if not batch:
                        break
                    extra_answers = []
                    for item_id in batch:
                        val = demo_value_for_item(tags=registry.items[item_id].tags, day_index=i)
                        extra_answers.append(
                            {
                                "client_event_id": f"{user_id}::{day.isoformat()}::extra{extra_used}::{item_id}",
                                "item_id": item_id,
                                "value": val,
                                "answered_at": f"{day.isoformat()}T12:{5 + extra_used:02d}:00Z",
                            }
                        )
                    submit_answers(
                        conn,
                        registry_root=cfg.registry_root,
                        user_id=user_id,
                        session_id=session.session_id,
                        answers=extra_answers,
                    )
                    extra_used += 1
                    want_extra = rng.random() < 0.25

        user_id = f"{str(user_prefix).strip() or 'policy-eval-user'}-1"
        progress = __progress_for_day(conn, cfg=cfg_eff, reg_v=reg_v, user_id=user_id, day=end_day)
        histories = {sid: __history(conn, user_id=user_id, scale_id=sid) for sid in registry.scales.keys()}
        latest_session = __latest_session(conn, user_id=user_id)
        html = render_user(user_id=user_id, scale_progress=progress, scale_histories=histories, latest_session=latest_session)
        out_path = Path(cfg.database_path).with_suffix(".report.html")
        out_path.write_text(html, encoding="utf-8")
        print(
            "Policy data collection complete. "
            f"mode={mode} users={users} days={days} sessions={total_sessions} epsilon={epsilon:.3f}. "
            f"Open report: {out_path}"
        )

def __progress_for_day(conn, *, cfg: QuestionsAgentConfig, reg_v: str, user_id: str, day: date):
    registry = load_registry(cfg.registry_root, reg_v)
    return compute_user_scale_progress(conn, registry=registry, user_id=user_id, day=day)


def __history(conn, *, user_id: str, scale_id: str):
    rows = conn.execute(
        """
        SELECT * FROM scale_scores
        WHERE user_id=? AND scale_id=?
        ORDER BY computed_at ASC
        LIMIT 200;
        """,
        (user_id, scale_id),
    ).fetchall()
    out = []
    for r in rows:
        out.append(
            {
                "normalized_score": float(r["normalized_score"]),
                "computed_at": str(r["computed_at"]),
                "window_end": str(r["window_end"]),
            }
        )
    return out


def __latest_session(conn, *, user_id: str):
    row = conn.execute(
        "SELECT * FROM daily_sessions WHERE user_id=? ORDER BY date DESC LIMIT 1;",
        (user_id,),
    ).fetchone()
    if not row:
        return None
    return {
        "session_id": str(row["session_id"]),
        "date": str(row["date"]),
        "registry_version": str(row["registry_version"]),
        "selection_explain": json.loads(row["selection_explain_json"]),
    }
