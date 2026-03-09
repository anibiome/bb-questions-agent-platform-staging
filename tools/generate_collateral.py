from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT_PDF_DIR = ROOT / "output" / "pdf"
OUT_VIDEO_DIR = ROOT / "output" / "video"
TMP_DIR = ROOT / "tmp" / "pdfs"


def main() -> None:
    OUT_PDF_DIR.mkdir(parents=True, exist_ok=True)
    OUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    pdf_path = OUT_PDF_DIR / "questions_agent_tech_specs_v1.pdf"
    gif_path = OUT_VIDEO_DIR / "questions_agent_demo.gif"

    pages = render_tech_specs_pdf()
    _write_png_previews(pages, TMP_DIR / "questions_agent_tech_specs_v1_page")
    save_images_as_pdf(pages, pdf_path)

    frames = render_demo_gif_frames()
    _write_png_previews(frames, TMP_DIR / "questions_agent_demo_frame")
    save_frames_as_gif(frames, gif_path, duration_ms=1800)

    print(f"Wrote PDF: {pdf_path}")
    print(f"Wrote demo GIF: {gif_path}")


def render_tech_specs_pdf() -> List[Image.Image]:
    """
    Generate a polished multi-page PDF as images and export via Pillow's PDF writer.

    We avoid external PDF dependencies (reportlab/weasyprint) to keep this repo runnable
    in restricted environments.
    """
    today = date.today().isoformat()

    # Letter @ 150 dpi -> 1275 x 1650
    page_w, page_h = 1275, 1650
    margin = 96
    content_w = page_w - 2 * margin

    fonts = load_fonts()
    h1_font = fonts.h1
    h2_font = fonts.h2
    body_font = fonts.body
    mono_font = fonts.mono

    pages: List[Image.Image] = []

    # Cover page
    cover = new_page(page_w, page_h)
    d = ImageDraw.Draw(cover)
    draw_header_block(
        d,
        x=margin,
        y=margin + 40,
        w=content_w,
        title="Questions Agent Platform",
        subtitle="Adaptive daily validated questionnaires -> scale evidence -> structured fusion evidence",
        meta=["Tech Specs v1 (Prod + Reference)", f"Date: {today}"],
        fonts=fonts,
    )
    draw_callout(
        d,
        x=margin,
        y=520,
        w=content_w,
        title="What this delivers",
        bullets=[
            "Daily personalized questions: 5 core + up to 3 extra context batches.",
            "Validated scale structures + within-person baselines and drift metrics.",
            "Item multiplexing: one answered item can feed multiple scales.",
            "Fusion evidence payload: projection (128d) + uncertainty + attractor + velocity.",
            "Registry versioning: upload/activate questionnaires without code changes.",
            "Production deploy: FastAPI + Postgres + Docker compose (2026-ready baseline).",
        ],
        fonts=fonts,
    )
    draw_footer(d, page_w, page_h, page_no=1, total_pages=None, fonts=fonts)
    pages.append(cover)

    # Content pages: build a long structured document and paginate.
    sections = build_spec_sections()
    doc_lines: List[Tuple[str, str]] = []  # (style, text)
    for title, paras in sections:
        doc_lines.append(("h1", title))
        for p in paras:
            if p.startswith("- "):
                doc_lines.append(("bullet", p[2:]))
            elif p.startswith("```"):
                doc_lines.append(("code", p.strip("`")))
            else:
                doc_lines.append(("body", p))
        doc_lines.append(("spacer", ""))

    # Render into pages
    current = new_page(page_w, page_h)
    draw = ImageDraw.Draw(current)
    y = margin
    page_no = 2

    def new_text_page() -> None:
        nonlocal current, draw, y, page_no
        draw_footer(draw, page_w, page_h, page_no=page_no - 1, total_pages=None, fonts=fonts)
        pages.append(current)
        current = new_page(page_w, page_h)
        draw = ImageDraw.Draw(current)
        y = margin

    for style, text in doc_lines:
        if style == "spacer":
            y += 12
            continue

        if style == "h1":
            block = wrap_text(draw, text, h1_font, content_w)
            needed = len(block) * (h1_font.size + 6) + 14
            if y + needed > page_h - margin - 60:
                new_text_page()
                page_no += 1
            y = draw_text_block(draw, block, margin, y, h1_font, fill=(230, 242, 255), line_gap=6)
            y += 10
            continue

        if style == "h2":
            block = wrap_text(draw, text, h2_font, content_w)
            needed = len(block) * (h2_font.size + 4) + 10
            if y + needed > page_h - margin - 60:
                new_text_page()
                page_no += 1
            y = draw_text_block(draw, block, margin, y, h2_font, fill=(200, 220, 240), line_gap=4)
            y += 6
            continue

        if style == "bullet":
            indent = 22
            bullet_prefix = "• "
            wrapped = wrap_text(draw, bullet_prefix + text, body_font, content_w - indent)
            needed = len(wrapped) * (body_font.size + 4) + 6
            if y + needed > page_h - margin - 60:
                new_text_page()
                page_no += 1
            # Draw bullet with hanging indent
            x0 = margin
            # first line
            draw.text((x0, y), "•", font=body_font, fill=(232, 238, 245))
            x_text = x0 + 18
            first_line = wrapped[0].lstrip("•").lstrip()
            draw.text((x_text, y), first_line, font=body_font, fill=(232, 238, 245))
            y += body_font.size + 4
            for line in wrapped[1:]:
                draw.text((x_text, y), line, font=body_font, fill=(232, 238, 245))
                y += body_font.size + 4
            y += 2
            continue

        if style == "code":
            # Render code lines in a rounded box
            code_lines = text.split("\\n")
            if not code_lines:
                continue
            box_pad = 12
            line_h = mono_font.size + 4
            box_h = box_pad * 2 + line_h * len(code_lines)
            if y + box_h > page_h - margin - 60:
                new_text_page()
                page_no += 1
            box = (margin, y, margin + content_w, y + box_h)
            draw_rounded_rect(draw, box, radius=10, fill=(18, 22, 28), outline=(40, 55, 70))
            yy = y + box_pad
            for ln in code_lines:
                draw.text((margin + box_pad, yy), ln, font=mono_font, fill=(210, 230, 255))
                yy += line_h
            y += box_h + 10
            continue

        # body
        block = wrap_text(draw, text, body_font, content_w)
        needed = len(block) * (body_font.size + 4) + 6
        if y + needed > page_h - margin - 60:
            new_text_page()
            page_no += 1
        y = draw_text_block(draw, block, margin, y, body_font, fill=(232, 238, 245), line_gap=4)
        y += 6

    draw_footer(draw, page_w, page_h, page_no=page_no - 1, total_pages=None, fonts=fonts)
    pages.append(current)

    # Add total pages in a second pass if desired (kept simple here).
    return pages


def render_demo_gif_frames() -> List[Image.Image]:
    """
    Create a short demo "video" as an animated GIF.
    """
    w, h = 960, 540
    fonts = load_fonts()

    slides: List[Tuple[str, List[str], str]] = [
        (
            "Questions Agent Demo",
            [
                "5 core questions/day + up to 3 extra context batches",
                "Validated scale unlocks + retests",
                "Within-person baselines and drift",
                "Structured fusion evidence (128d vector + uncertainty)",
            ],
            "1/7",
        ),
        (
            "1) Daily questions API",
            [
                "GET /v1/users/{user_id}/daily-questions",
                "Returns: session_id, 5 questions, unlock progress",
                "Selection is deterministic per user/day",
            ],
            "2/7",
        ),
        (
            "2) Answer events (idempotent)",
            [
                "POST /v1/users/{user_id}/answers",
                "client_event_id makes retries safe",
                "Each answer can feed multiple scales (multiplexing)",
            ],
            "3/7",
        ),
        (
            "3) Unlocks and retests",
            [
                "Scale progress: 'X missing' to unlock",
                "Once unlocked: score + confidence tier",
                "Retest interval (e.g., 60-90d) enables then-vs-now comparison",
            ],
            "4/7",
        ),
        (
            "4) Personal baseline and drift",
            [
                "EWMA baseline per user per scale",
                "Drift metrics: personal_z and delta_vs_prev",
                "Drift can trigger probing (more items from that scale/domain)",
            ],
            "5/7",
        ),
        (
            "5) Anifold evidence payload",
            [
                "GET /v1/users/{user_id}/projection/questions",
                "projection[128] + uncertainty.diag[128]",
                "attractor_candidate (baseline vector) + velocity + distance",
                "Evidence only: Anifold computes coherence/circle scores",
            ],
            "6/7",
        ),
        (
            "6) Deploy (2026-ready baseline)",
            [
                "FastAPI + Postgres + Docker compose",
                "Registry versioning: upload + activate",
                "Auth: X-API-Key header",
                "Dashboard: /dashboard (QA + clinicians)",
            ],
            "7/7",
        ),
    ]

    frames: List[Image.Image] = []
    for title, bullets, stamp in slides:
        img = Image.new("RGB", (w, h), (11, 13, 16))
        d = ImageDraw.Draw(img)
        draw_slide(d, w, h, title=title, bullets=bullets, stamp=stamp, fonts=fonts)
        frames.append(img)

    # Add a simple "dashboard preview" frame with sparklines
    frames.insert(5, render_dashboard_preview_frame(w, h, fonts, stamp="(preview)"))
    return frames


def render_dashboard_preview_frame(w: int, h: int, fonts: "Fonts", stamp: str) -> Image.Image:
    img = Image.new("RGB", (w, h), (11, 13, 16))
    d = ImageDraw.Draw(img)
    draw_slide(
        d,
        w,
        h,
        title="Dashboard preview",
        bullets=[
            "Clinician/QA view of scales, unlock state, and trends",
            "Generated from real stored scores (sparklines)",
        ],
        stamp=stamp,
        fonts=fonts,
    )
    # Fake table
    x0, y0 = 70, 250
    table_w, row_h = 820, 44
    d.rounded_rectangle((x0, y0, x0 + table_w, y0 + row_h * 5), radius=12, fill=(17, 21, 27), outline=(40, 55, 70))
    headers = ["Scale", "Unlock", "Latest", "Trend"]
    col_x = [x0 + 16, x0 + 280, x0 + 420, x0 + 540]
    for i, htxt in enumerate(headers):
        d.text((col_x[i], y0 + 12), htxt, font=fonts.small_caps, fill=(160, 176, 192))

    rows = [
        ("Mood severity", "unlocked", "62.0", spark_points([10, 15, 25, 40, 55, 62])),
        ("Sleep problems", "2 missing", "—", spark_points([5, 8, 12, 18, 22, 20])),
        ("GI burden", "unlocked", "41.5", spark_points([12, 14, 16, 18, 26, 41])),
        ("Stress load", "unlocked", "70.2", spark_points([20, 22, 30, 55, 72, 70])),
    ]
    for r, row in enumerate(rows):
        yy = y0 + row_h * (r + 1)
        d.line((x0, yy, x0 + table_w, yy), fill=(40, 55, 70), width=1)
        name, unlock, latest, pts = row
        d.text((col_x[0], yy + 12), name, font=fonts.body, fill=(232, 238, 245))
        color = (52, 211, 153) if unlock == "unlocked" else (251, 113, 133)
        pill(d, col_x[1], yy + 10, unlock, color, fonts=fonts)
        d.text((col_x[2], yy + 12), latest, font=fonts.body, fill=(232, 238, 245))
        draw_sparkline(d, col_x[3], yy + 8, pts, w=260, h=28)

    return img


def spark_points(values: Sequence[int]) -> List[float]:
    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        vmax = vmin + 1
    return [(v - vmin) / (vmax - vmin) for v in values]


def draw_slide(
    d: ImageDraw.ImageDraw,
    w: int,
    h: int,
    *,
    title: str,
    bullets: List[str],
    stamp: str,
    fonts: "Fonts",
) -> None:
    # Title
    d.text((70, 64), title, font=fonts.h1, fill=(230, 242, 255))
    d.text((w - 120, 72), stamp, font=fonts.body, fill=(140, 160, 176))

    # Bullets
    y = 140
    for b in bullets:
        d.text((80, y), "•", font=fonts.body, fill=(232, 238, 245))
        d.text((104, y), b, font=fonts.body, fill=(232, 238, 245))
        y += 38

    # Footer strip
    d.rounded_rectangle((70, h - 88, w - 70, h - 52), radius=16, fill=(17, 21, 27), outline=(40, 55, 70))
    d.text((90, h - 80), "Questions Agent Platform - deployable demo collateral", font=fonts.mono, fill=(170, 190, 210))


def draw_sparkline(
    d: ImageDraw.ImageDraw, x: int, y: int, values_0_1: Sequence[float], *, w: int, h: int
) -> None:
    if not values_0_1:
        return
    pts = []
    n = max(2, len(values_0_1))
    for i, v in enumerate(values_0_1):
        px = x + int((i / (n - 1)) * (w - 2)) + 1
        py = y + int((1.0 - float(v)) * (h - 2)) + 1
        pts.append((px, py))
    d.line(pts, fill=(125, 211, 252), width=3, joint="curve")


def pill(d: ImageDraw.ImageDraw, x: int, y: int, text: str, color: Tuple[int, int, int], *, fonts: "Fonts") -> None:
    pad_x, pad_y = 10, 6
    tw = d.textlength(text, font=fonts.body)
    box = (x, y, x + int(tw) + pad_x * 2, y + fonts.body.size + pad_y * 2 - 2)
    d.rounded_rectangle(box, radius=999, fill=(20, 24, 30), outline=(40, 55, 70))
    d.text((x + pad_x, y + pad_y - 2), text, font=fonts.body, fill=color)


def save_frames_as_gif(frames: List[Image.Image], path: Path, *, duration_ms: int) -> None:
    if not frames:
        raise ValueError("No frames to save")
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def save_images_as_pdf(pages: List[Image.Image], path: Path) -> None:
    if not pages:
        raise ValueError("No pages to save")
    # Pillow expects RGB for PDF
    pages_rgb = [p.convert("RGB") for p in pages]
    pages_rgb[0].save(path, "PDF", save_all=True, append_images=pages_rgb[1:])


def new_page(w: int, h: int) -> Image.Image:
    return Image.new("RGB", (w, h), (11, 13, 16))


@dataclass(frozen=True)
class Fonts:
    title: ImageFont.FreeTypeFont
    h1: ImageFont.FreeTypeFont
    h2: ImageFont.FreeTypeFont
    body: ImageFont.FreeTypeFont
    mono: ImageFont.FreeTypeFont
    small_caps: ImageFont.FreeTypeFont


def load_fonts() -> Fonts:
    # Try common macOS fonts; fall back to default bitmap if missing.
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/System/Library/Fonts/Supplemental/Verdana.ttf",
    ]
    mono_candidates = [
        "/System/Library/Fonts/Supplemental/Menlo-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
    ]

    def pick(paths: List[str]) -> str:
        for p in paths:
            if Path(p).exists():
                return p
        return ""

    base = pick(candidates)
    mono = pick(mono_candidates) or base
    if base:
        return Fonts(
            title=ImageFont.truetype(base, 52),
            h1=ImageFont.truetype(base, 34),
            h2=ImageFont.truetype(base, 26),
            body=ImageFont.truetype(base, 18),
            mono=ImageFont.truetype(mono, 16),
            small_caps=ImageFont.truetype(base, 14),
        )

    # Ultimate fallback
    f = ImageFont.load_default()
    return Fonts(title=f, h1=f, h2=f, body=f, mono=f, small_caps=f)


def wrap_text(
    d: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int
) -> List[str]:
    words = text.split()
    if not words:
        return [""]
    lines: List[str] = []
    current = words[0]
    for w in words[1:]:
        candidate = current + " " + w
        if d.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = w
    lines.append(current)
    return lines


def draw_text_block(
    d: ImageDraw.ImageDraw,
    lines: Sequence[str],
    x: int,
    y: int,
    font: ImageFont.ImageFont,
    *,
    fill: Tuple[int, int, int],
    line_gap: int,
) -> int:
    yy = y
    for ln in lines:
        d.text((x, yy), ln, font=font, fill=fill)
        yy += font.size + line_gap
    return yy


def draw_header_block(
    d: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    w: int,
    title: str,
    subtitle: str,
    meta: List[str],
    fonts: Fonts,
) -> None:
    d.text((x, y), title, font=fonts.title, fill=(230, 242, 255))
    y2 = y + fonts.title.size + 16
    subtitle_lines = wrap_text(d, subtitle, fonts.h2, w)
    y2 = draw_text_block(d, subtitle_lines, x, y2, fonts.h2, fill=(200, 220, 240), line_gap=4)
    y2 += 18
    for m in meta:
        d.text((x, y2), m, font=fonts.body, fill=(160, 176, 192))
        y2 += fonts.body.size + 8


def draw_callout(
    d: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    w: int,
    title: str,
    bullets: List[str],
    fonts: Fonts,
) -> None:
    box = (x, y, x + w, y + 520)
    draw_rounded_rect(d, box, radius=18, fill=(17, 21, 27), outline=(40, 55, 70))
    d.text((x + 22, y + 18), title, font=fonts.h2, fill=(230, 242, 255))
    yy = y + 70
    for b in bullets:
        d.text((x + 26, yy), "•", font=fonts.body, fill=(232, 238, 245))
        lines = wrap_text(d, b, fonts.body, w - 70)
        d.text((x + 50, yy), lines[0], font=fonts.body, fill=(232, 238, 245))
        yy += fonts.body.size + 6
        for ln in lines[1:]:
            d.text((x + 50, yy), ln, font=fonts.body, fill=(232, 238, 245))
            yy += fonts.body.size + 6
        yy += 6


def draw_footer(d: ImageDraw.ImageDraw, w: int, h: int, *, page_no: int, total_pages: int | None, fonts: Fonts) -> None:
    y = h - 54
    d.line((90, y, w - 90, y), fill=(40, 55, 70), width=1)
    d.text((90, y + 12), "Questions Agent Platform - Tech Specs", font=fonts.small_caps, fill=(140, 160, 176))
    right = f"Page {page_no}" if total_pages is None else f"Page {page_no}/{total_pages}"
    d.text((w - 240, y + 12), right, font=fonts.small_caps, fill=(140, 160, 176))


def draw_rounded_rect(
    d: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    *,
    radius: int,
    fill: Tuple[int, int, int],
    outline: Tuple[int, int, int],
) -> None:
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=2)


def build_spec_sections() -> List[Tuple[str, List[str]]]:
    """
    Returns sections as (title, paragraphs). Paragraphs can start with "- " for bullets.
    Keep ASCII hyphens only (per PDF quality expectations).
    """
    return [
        (
            "1. Purpose and vision alignment",
            [
                "Goal: deliver a production-ready Questions Agent that uses validated questionnaire items to produce daily adaptive questions, unlock and retest validated scales, and emit geometry-ready evidence that can be fused in Anifold/Unifold latent space.",
                "- 5 questions per day, always.",
                "- Optional: user can request more context up to 3 times/day (each batch adds 5 questions).",
                "- Scales are not treated as population thresholds; we track within-person baselines and change patterns.",
                "- Multiplexing: one item can feed multiple scales/questionnaires.",
                "- Evidence vs scoring separation: Questions Agent produces evidence; Anifold produces circle/coherence scoring.",
            ],
        ),
        (
            "2. System separation (evidence vs scoring)",
            [
                "This implementation follows your Multi-Omics Projection Specification separation:",
                "- Questions Agent = geometry-ready evidence layers (events, projections with uncertainty, attractor candidate, velocity).",
                "- Anifold = global scoring (coherence/circle values) and final user interpretation layer.",
                "Practically: this service never claims diagnosis. It stores validated evidence, trends, and within-person change signals.",
            ],
        ),
        (
            "3. Core data contracts",
            [
                "The system has three core contracts: (1) registry, (2) events, (3) derived outputs.",
                "Registry (versioned): items, scales, questionnaires.",
                "- Item: item_id, text, response_type, tags, sensitivity.",
                "- Scale: scale_id, questionnaire_id, scoring method, min_items_required, unlock_window_days, retest_interval_days, item mappings (reverse, weight).",
                "Events (append-only evidence):",
                "- AnswerEvent: atomic user answer with idempotency key (client_event_id).",
                "- ObservationEvent: non-question signals (wearables/voice/imaging/omics) stored for future probe triggers.",
                "Derived outputs:",
                "- ScaleScore: normalized score (0-100), confidence tier, personal_z, delta_vs_prev.",
                "- Projection payload: 128d evidence vector + uncertainty + attractor + velocity.",
            ],
        ),
        (
            "4. Daily selection algorithm (v1)",
            [
                "Each day, selection chooses 5 core items, and precomputes up to 3 extra batches (5 items each) for the 'Give more context' UX.",
                "Priorities (highest to lowest):",
                "- Near-unlock: finish scales that are within 1-2 missing items (for never-completed scales, or re-unlock when due for retest).",
                "- Due retest: if last score is older than retest_interval_days, prioritize that scale's items.",
                "- Drift probe: if rolling score differs from baseline enough (personal_z or absolute delta), probe that scale.",
                "- Multiplex gain: items that feed multiple scales are mildly preferred.",
                "- Anchors: 1-2 stable items for continuity (mood/sleep/energy/stress tagged).",
                "Safety: a repeat cooldown prevents spamming the same items unless they are needed for unlock/retest/drift.",
            ],
        ),
        (
            "5. Scoring and within-person baselines",
            [
                "Scoring is deterministic from the registry mapping and latest answers within the unlock window.",
                "- Methods supported: mean or sum, with reverse coding and weights.",
                "- Normalization: scale-specific min/max converted to 0-100.",
                "Baseline:",
                "- Each user-scale baseline is tracked using EWMA (mean and variance).",
                "- Drift metrics include personal_z (if variance is meaningful) and delta_vs_prev retest score.",
                "Confidence tier:",
                "- high: full item coverage.",
                "- medium: meets min_items_required but not full coverage.",
            ],
        ),
        (
            "6. Projection payload (structured fusion evidence)",
            [
                "We build a stable 128-dimensional vector without training a model by using feature hashing:",
                "- score:{scale_id} contributes normalized score (0..1) into a stable bucket with random sign.",
                "- z:{scale_id} contributes clipped personal_z into another stable bucket.",
                "Outputs:",
                "- projection[128] (L2-normalized).",
                "- uncertainty.diag[128] (heuristic based on how many features contribute to each dimension).",
                "- attractor_candidate[128] built from baseline means (stable attractor).",
                "- velocity[128] (diff vs previous stored projection when available).",
                "- distance_from_attractor (L2 distance).",
                "This matches the universal 'projection + uncertainty + attractor + velocity' template from your execution contract.",
            ],
        ),
        (
            "7. Production deployment stack (2026 baseline)",
            [
                "We shipped a production service in questions_agent_platform/prod:",
                "- FastAPI + Pydantic validation",
                "- Postgres schema (SQLAlchemy ORM + SQL init script)",
                "- Dockerfile + docker-compose",
                "- API key auth (X-API-Key header)",
                "This is deployable as a standalone service behind your main app backend or gateway.",
            ],
        ),
        (
            "8. Endpoints (production)",
            [
                "- GET /health",
                "- GET /dashboard (auth required)",
                "- GET /v1/users/{user_id}/daily-questions",
                "- POST /v1/users/{user_id}/answers",
                "- POST /v1/users/{user_id}/request-more-context",
                "- GET /v1/users/{user_id}/scales",
                "- GET /v1/users/{user_id}/scales/{scale_id}/history",
                "- GET /v1/users/{user_id}/projection/questions",
                "- POST /v1/users/{user_id}/observations",
                "- POST /v1/admin/registry/upload",
                "- POST /v1/admin/registry/activate/{version}",
            ],
        ),
        (
            "9. What is still needed to match the full long-term vision",
            [
                "This v1 is deployable and aligned with your docs. Remaining (optional) work to reach the full long-term vision:",
                "- Plug in your real questionnaire bank content (licensed text) via the registry upload process.",
                "- Replace API-key auth with your production auth (JWT, gateway headers, RBAC).",
                "- Add ML-based selection (bandits/RL) and multimodal triggers using ObservationEvents.",
                "- Add integrated multimodal projection (questions + omics + imaging + voice) with concordance and uncertainty calibration.",
                "- Add cohort analytics and clinical study pipelines (separate service) and validation dashboards.",
            ],
        ),
        (
            "Appendix A. Production quickstart",
            [
                "Docker compose (local):",
                "```",
                "cd questions_agent_platform/prod",
                "cp .env.example .env",
                "docker compose up --build",
                "```",
                "Call example (API key header):",
                "```",
                "curl -H 'X-API-Key: change_me' http://localhost:8080/health",
                "```",
            ],
        ),
    ]


def _write_png_previews(images: Sequence[Image.Image], prefix: Path) -> None:
    # Writes numbered PNG files, e.g. prefix_001.png
    for idx, img in enumerate(images, start=1):
        out = Path(f"{prefix}_{idx:03d}.png")
        img.save(out, format="PNG")


if __name__ == "__main__":
    main()
