import argparse
import json
import sys
from pathlib import Path

from questions_agent_platform.pipeline.config import ensure_paths, load_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="questions-agent")
    parser.add_argument("--config", required=True, help="Path to config.json")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init-db", help="Initialize the SQLite database")
    sub.add_parser("serve", help="Run the HTTP API + dashboard")
    sub.add_parser("seed-demo", help="Load demo registry content into registry_root")
    sub.add_parser("simulate", help="Simulate 30 days for a demo user")
    simp = sub.add_parser("simulate-policy", help="Collect policy decision/outcome data with configurable days/users/mode")
    simp.add_argument("--days", type=int, default=120, help="Number of simulated days")
    simp.add_argument("--users", type=int, default=1, help="Number of demo users")
    simp.add_argument(
        "--selection-mode",
        default="policy_live",
        choices=["deterministic", "policy_live", "policy_shadow", "policy_offline_replay"],
        help="Selection mode for collection",
    )
    simp.add_argument("--epsilon", type=float, default=0.25, help="Exploration epsilon override for simulation")
    simp.add_argument("--seed", type=int, default=42, help="RNG seed")
    simp.add_argument("--user-prefix", default="policy-eval-user", help="Prefix for synthetic user ids")
    simp.add_argument("--reset-db", action="store_true", help="Reinitialize DB before simulation")

    brief_p = sub.add_parser("policy-brief", help="Generate policy readiness brief (HTML + JSON)")
    brief_p.add_argument("--db", default=None, help="Override sqlite db path")
    brief_p.add_argument("--out-dir", default="questions_agent_platform/output", help="Output directory")
    brief_p.add_argument("--label", default="Questions Agent Policy Brief", help="Brief title")
    brief_p.add_argument("--policy-root", default=None, help="Policy artifact root for offline eval (defaults to config)")
    brief_p.add_argument("--min-eval-rows", type=int, default=100, help="Minimum eval rows required by gate")
    brief_p.add_argument("--min-segment-rows", type=int, default=30, help="Minimum eval rows required per segment")

    gate_p = sub.add_parser("policy-gate", help="Run rollout gate and exit non-zero on failure")
    gate_p.add_argument("--db", default=None, help="Override sqlite db path")
    gate_p.add_argument("--out-dir", default="questions_agent_platform/output", help="Output directory")
    gate_p.add_argument("--label", default="Questions Agent Policy Gate", help="Gate report title")
    gate_p.add_argument("--policy-root", default=None, help="Policy artifact root for offline eval (defaults to config)")
    gate_p.add_argument("--min-eval-rows", type=int, default=100, help="Minimum eval rows required by gate")
    gate_p.add_argument("--min-segment-rows", type=int, default=30, help="Minimum eval rows required per segment")

    weekly_p = sub.add_parser("weekly-policy-report", help="Generate weekly policy report (HTML + JSON)")
    weekly_p.add_argument("--db", default=None, help="Override sqlite db path")
    weekly_p.add_argument("--out-dir", default="questions_agent_platform/output", help="Output directory")
    weekly_p.add_argument("--label", default="Questions Agent Weekly Policy Report", help="Report title")
    weekly_p.add_argument("--policy-root", default=None, help="Policy artifact root for offline eval (defaults to config)")
    weekly_p.add_argument("--days", type=int, default=7, help="Window size in days")
    weekly_p.add_argument("--min-eval-rows", type=int, default=100, help="Minimum eval rows required by gate")
    weekly_p.add_argument("--min-segment-rows", type=int, default=30, help="Minimum eval rows required per segment")

    val_p = sub.add_parser("validate-registry", help="Validate registry data before production")
    val_p.add_argument("--version", default=None, help="Specific version to validate (default: all)")

    cl_p = sub.add_parser("closed-loop-truth-demo", help="Run deterministic end-to-end loop proof and generate JSON + HTML")
    cl_p.add_argument("--days", type=int, default=30, help="Number of replay days")
    cl_p.add_argument("--user-id", default="closed-loop-user-1", help="Synthetic user id for replay")
    cl_p.add_argument("--out-dir", default="questions_agent_platform/output", help="Output directory")
    cl_p.add_argument("--seed", type=int, default=42, help="RNG seed")
    cl_p.add_argument("--epsilon", type=float, default=0.25, help="Policy exploration epsilon override")
    cl_p.add_argument("--reset-db", action="store_true", help="Reinitialize DB before running replay")

    replay_p = sub.add_parser(
        "replay-daily-scan-answers",
        help="Replay exported daily scan answers into Questions Agent state and export scale/projection artifacts",
    )
    replay_p.add_argument("--input-csv", required=True, help="Path to daily_scan_answers.csv")
    replay_p.add_argument(
        "--out-dir",
        default="questions_agent_platform/output/daily_scan_replay",
        help="Directory for exported replay artifacts",
    )
    replay_p.add_argument(
        "--registry-source",
        default="questions_agent_platform/data/registry_production/v3",
        help="Directory containing items.json/questionnaires.json/scales.json for the registry bundle",
    )
    replay_p.add_argument("--registry-version", default="v3", help="Registry version label to activate")
    replay_p.add_argument(
        "--date-column",
        default="scan_date_local",
        choices=["scan_date_local", "scan_date_utc"],
        help="Which export date column should define the replay session day",
    )
    replay_p.add_argument(
        "--selection-mode",
        default="deterministic",
        choices=["deterministic", "policy_live", "policy_shadow", "policy_offline_replay"],
        help="Session creation mode used while replaying answers",
    )
    replay_p.add_argument(
        "--skip-projections",
        action="store_true",
        help="Do not build AniFold-facing question projection payloads during replay",
    )

    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_paths(cfg)

    # Lazy imports (keep CLI help fast)
    if args.cmd == "init-db":
        from questions_agent_platform.pipeline.db import init_db

        init_db(cfg.database_path)
        return

    if args.cmd == "seed-demo":
        from questions_agent_platform.pipeline.demo import seed_demo_registry

        seed_demo_registry(cfg.registry_root)
        return

    if args.cmd == "simulate":
        from questions_agent_platform.pipeline.demo import run_demo_simulation

        run_demo_simulation(cfg)
        return

    if args.cmd == "simulate-policy":
        from questions_agent_platform.pipeline.demo import run_policy_data_collection

        run_policy_data_collection(
            cfg=cfg,
            days=int(args.days),
            users=int(args.users),
            selection_mode=str(args.selection_mode),
            epsilon=float(args.epsilon),
            seed=int(args.seed),
            user_prefix=str(args.user_prefix),
            reset_db=bool(args.reset_db),
        )
        return

    if args.cmd == "serve":
        from questions_agent_platform.pipeline.api import run_server

        run_server(cfg)
        return

    if args.cmd == "policy-brief":
        from questions_agent_platform.tools.generate_policy_brief import generate_policy_brief

        result = generate_policy_brief(
            db_path=str(args.db or cfg.database_path),
            out_dir=str(args.out_dir),
            label=str(args.label),
            policy_root=str(args.policy_root or cfg.policy_root),
            min_eval_rows=int(args.min_eval_rows),
            min_segment_rows=int(args.min_segment_rows),
        )
        print(f"Wrote JSON: {result.json_path}")
        print(f"Wrote HTML: {result.html_path}")
        return

    if args.cmd == "policy-gate":
        from questions_agent_platform.tools.generate_policy_brief import generate_policy_brief

        result = generate_policy_brief(
            db_path=str(args.db or cfg.database_path),
            out_dir=str(args.out_dir),
            label=str(args.label),
            policy_root=str(args.policy_root or cfg.policy_root),
            min_eval_rows=int(args.min_eval_rows),
            min_segment_rows=int(args.min_segment_rows),
        )
        print(f"Wrote JSON: {result.json_path}")
        print(f"Wrote HTML: {result.html_path}")
        payload = json.loads(Path(result.json_path).read_text(encoding="utf-8"))
        gates = payload.get("quality_gates") if isinstance(payload.get("quality_gates"), dict) else {}
        failed = [name for name, ok in gates.items() if not bool(ok)]
        if failed:
            print("GATE_STATUS: FAIL")
            print("FAILED_GATES:", ", ".join(failed))
            sys.exit(1)
        print("GATE_STATUS: PASS")
        return

    if args.cmd == "weekly-policy-report":
        from questions_agent_platform.tools.generate_weekly_policy_report import generate_weekly_policy_report

        result = generate_weekly_policy_report(
            db_path=str(args.db or cfg.database_path),
            out_dir=str(args.out_dir),
            label=str(args.label),
            policy_root=str(args.policy_root or cfg.policy_root),
            days=int(args.days),
            min_eval_rows=int(args.min_eval_rows),
            min_segment_rows=int(args.min_segment_rows),
        )
        print(f"Wrote JSON: {result.json_path}")
        print(f"Wrote HTML: {result.html_path}")
        return

    if args.cmd == "validate-registry":
        from questions_agent_platform.tools.validate_registry import (
            print_validation_result,
            run_validate_registry,
        )

        result = run_validate_registry(cfg.registry_root, version=args.version)
        print_validation_result(result)
        if not result.ok:
            sys.exit(1)
        return

    if args.cmd == "closed-loop-truth-demo":
        from questions_agent_platform.tools.generate_closed_loop_truth_demo import generate_closed_loop_truth_demo

        result = generate_closed_loop_truth_demo(
            cfg=cfg,
            out_dir=str(args.out_dir),
            days=int(args.days),
            user_id=str(args.user_id),
            seed=int(args.seed),
            epsilon=float(args.epsilon),
            reset_db=bool(args.reset_db),
        )
        print(f"Wrote JSON: {result.json_path}")
        print(f"Wrote HTML: {result.html_path}")
        return

    if args.cmd == "replay-daily-scan-answers":
        from questions_agent_platform.tools.replay_daily_scan_answers import run_daily_scan_replay

        summary = run_daily_scan_replay(
            cfg=cfg,
            input_csv=str(args.input_csv),
            out_dir=str(args.out_dir),
            registry_source=str(args.registry_source),
            registry_version=str(args.registry_version),
            date_column=str(args.date_column),
            selection_mode=str(args.selection_mode),
            build_projections=not bool(args.skip_projections),
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return


if __name__ == "__main__":
    main()
