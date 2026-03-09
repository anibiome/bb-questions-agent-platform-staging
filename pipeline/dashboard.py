from __future__ import annotations

import html
from typing import Any, Dict, List, Optional, Sequence


def render_layout(title: str, body_html: str) -> str:
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{html.escape(title)}</title>
    <style>
      :root {{
        --bg: #0b0d10;
        --panel: #11151a;
        --text: #e8eef5;
        --muted: #9fb0c0;
        --accent: #7dd3fc;
        --danger: #fb7185;
        --ok: #34d399;
        --border: rgba(255,255,255,0.08);
        --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
      }}
      html, body {{ background: var(--bg); color: var(--text); font-family: var(--sans); margin: 0; }}
      a {{ color: var(--accent); text-decoration: none; }}
      a:hover {{ text-decoration: underline; }}
      header {{ padding: 18px 20px; border-bottom: 1px solid var(--border); background: rgba(255,255,255,0.02); }}
      header .title {{ font-weight: 700; letter-spacing: 0.2px; }}
      main {{ padding: 20px; max-width: 1100px; margin: 0 auto; }}
      .grid {{ display: grid; grid-template-columns: 1fr; gap: 14px; }}
      @media (min-width: 900px) {{ .grid.two {{ grid-template-columns: 1fr 1fr; }} }}
      .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 14px; }}
      .muted {{ color: var(--muted); }}
      .mono {{ font-family: var(--mono); }}
      table {{ width: 100%; border-collapse: collapse; }}
      th, td {{ padding: 10px 8px; border-bottom: 1px solid var(--border); text-align: left; vertical-align: top; }}
      th {{ color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }}
      .pill {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; border: 1px solid var(--border); }}
      .pill.ok {{ color: var(--ok); }}
      .pill.warn {{ color: var(--danger); }}
      .spark {{ width: 140px; height: 28px; }}
      .row {{ display: flex; gap: 10px; align-items: center; justify-content: space-between; }}
      .kvs {{ display: grid; grid-template-columns: 180px 1fr; gap: 6px 10px; }}
      .kvs .k {{ color: var(--muted); }}
      code {{ font-family: var(--mono); }}
      pre {{ background: rgba(255,255,255,0.03); padding: 12px; border-radius: 10px; overflow: auto; border: 1px solid var(--border); }}
    </style>
  </head>
  <body>
    <header><span class="title">Questions Agent</span> <span class="muted">/ dashboard</span></header>
    <main>
      {body_html}
    </main>
  </body>
</html>"""


def render_index(users: Sequence[str]) -> str:
    rows = []
    for u in users:
        rows.append(f"<tr><td><a class='mono' href='/dashboard/users/{html.escape(u)}'>{html.escape(u)}</a></td></tr>")
    table = "<table><thead><tr><th>User</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    body = f"""
    <div class="card">
      <div class="row">
        <div>
          <div style="font-weight:700; font-size:18px;">Users</div>
          <div class="muted">Select a user to inspect scale unlocks, scores, and drift signals.</div>
        </div>
      </div>
      <div style="margin-top:12px;">{table}</div>
    </div>
    """
    return render_layout("Questions Agent Dashboard", body)


def render_user(
    *,
    user_id: str,
    scale_progress: Dict[str, Any],
    scale_histories: Dict[str, List[Dict[str, Any]]],
    latest_session: Optional[Dict[str, Any]],
) -> str:
    cards = []

    # Overview card
    explain = (latest_session or {}).get("selection_explain") or {}
    body_kvs = [
        ("User", f"<span class='mono'>{html.escape(user_id)}</span>"),
        ("Near unlock", html.escape(", ".join(explain.get("near_unlock_scales", [])) or "—")),
        ("Due retest", html.escape(", ".join(explain.get("due_retest_scales", [])) or "—")),
        ("Drift", html.escape(", ".join(explain.get("drift_scales", [])) or "—")),
    ]
    kv_html = "<div class='kvs'>" + "".join(f"<div class='k'>{k}</div><div>{v}</div>" for k, v in body_kvs) + "</div>"
    cards.append(f"<div class='card'><div style='font-weight:700; font-size:18px;'>Overview</div><div style='margin-top:10px;'>{kv_html}</div></div>")

    # Scales table
    rows = []
    for scale_id, prog in sorted(scale_progress.items(), key=lambda kv: kv[0]):
        last = prog.get("last_score")
        latest_values = scale_histories.get(scale_id, [])
        spark = _sparkline([s.get("normalized_score", 0.0) for s in latest_values[-20:]])
        score_str = "—"
        tier = ""
        if last:
            score_str = f"{float(last['normalized_score']):.1f}"
            tier = str(last.get("confidence_tier") or "")
        unlocked = bool(prog.get("unlocked"))
        missing = int(prog.get("missing_count", 0))
        if unlocked:
            unlock_label = "unlocked"
            pill = "pill ok"
        else:
            unlock_label = f"{missing} missing" if missing > 0 else "ready"
            pill = "pill warn" if missing > 0 else "pill ok"
        rows.append(
            "<tr>"
            f"<td class='mono'>{html.escape(scale_id)}</td>"
            f"<td>{html.escape(str(prog.get('name') or ''))}</td>"
            f"<td><span class='{pill}'>{html.escape(unlock_label)}</span></td>"
            f"<td>{html.escape(score_str)} <span class='muted'>{html.escape(tier)}</span></td>"
            f"<td>{spark}</td>"
            "</tr>"
        )
    table = (
        "<table><thead><tr>"
        "<th>Scale ID</th><th>Name</th><th>Unlock</th><th>Latest</th><th>Trend</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    cards.append(f"<div class='card'><div style='font-weight:700; font-size:18px;'>Scales</div><div style='margin-top:12px;'>{table}</div></div>")

    body = (
        "<div style='margin-bottom:12px;'><a href='/dashboard'>&larr; back</a></div>"
        + "<div class='grid'>" + "".join(cards) + "</div>"
    )
    return render_layout(f"User {user_id}", body)


def _sparkline(values: Sequence[float]) -> str:
    if not values:
        return "<span class='muted'>—</span>"
    w, h = 140, 28
    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-9:
        vmax = vmin + 1.0

    pts = []
    for i, v in enumerate(values):
        x = (i / max(1, len(values) - 1)) * (w - 2) + 1
        y = (1.0 - ((float(v) - vmin) / (vmax - vmin))) * (h - 2) + 1
        pts.append(f"{x:.1f},{y:.1f}")

    poly = " ".join(pts)
    return (
        f"<svg class='spark' viewBox='0 0 {w} {h}' xmlns='http://www.w3.org/2000/svg'>"
        f"<polyline fill='none' stroke='rgba(125,211,252,0.9)' stroke-width='2' points='{poly}' />"
        f"</svg>"
    )
