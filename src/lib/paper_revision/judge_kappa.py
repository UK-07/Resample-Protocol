#!/usr/bin/env python3
"""Render three kappa summaries from a versioned, explicitly sourced JSON asset.

``assets/judge_kappa_summary.json`` contains the existing inter-LLM binary audit
summary (296 Nemotron/MMLU-Pro rows) and role figures transcribed from the prior
paper PDF. Its provenance fields describe the sources and limitations: these
are not a new audit, a human-accuracy estimate, or design-weighted kappas. Raw
role audit records are unavailable, so those kappas cannot be recomputed here.
Only ``build(paths)`` reads data or registers fonts. Font assets are bundled with
Matplotlib, removing the former machine-specific system-font path.
"""
import json
from pathlib import Path
import matplotlib
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

FIGURES: Path
DATA: dict


def _register_fonts():
    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("KappaSans", str(font_dir / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("KappaBold", str(font_dir / "DejaVuSans-Bold.ttf")))


def draw(name, width, height, rows, title, subtitle, roles=False):
    c = Canvas(str(FIGURES / name), pagesize=(width, height), invariant=1)
    c.setTitle(title)
    c.setFillColor(HexColor("#243746"))
    c.setFont("KappaBold", 8)
    c.drawString(3, height - 12, title)
    c.setFont("KappaSans", 7)
    c.drawString(3, height - 23, subtitle)
    top = height - 47
    step = (height - 80) / (len(rows) - 1)
    left, right = width * 0.59, width - 24
    bottom = top - step * (len(rows) - 1)
    c.setStrokeColor(HexColor("#DCE2E7"))
    c.setLineWidth(.5)
    for x in (0, .5, 1):
        px = left + x * (right - left)
        c.line(px, bottom - 6, px, top + 7)
        c.setFont("KappaSans", 7)
        c.drawCentredString(px, bottom - 17, str(x).removesuffix('.0'))
    c.setFont("KappaSans", 7)
    c.drawRightString(width - 2, top + 15, "κ")
    for i, row in enumerate(rows):
        y = top - i * step
        label = row['label'].replace('grader code', 'grader hacking')
        label = label.replace('verification only', 'verification').replace('neutral mention', 'neutral')
        if roles:
            label = label.replace('all roles (5-class)', 'all roles')
        label += f" ({row['n']})"
        size = 7
        assert pdfmetrics.stringWidth(label, "KappaSans", size) < left - 5, label
        c.setFillColor(HexColor("#243746"))
        c.setFont("KappaBold" if i == len(rows) - 1 else "KappaSans", size)
        c.drawString(3, y - 2.5, label)
        c.setFillColor(HexColor("#0072B2"))
        c.rect(left, y - 3, row['kappa'] * (right - left), 6, stroke=0, fill=1)
        c.setFillColor(HexColor("#243746"))
        c.setFont("KappaSans", 7)
        c.drawRightString(width - 2, y - 2.5, f"{row['kappa']:.2f}")
    c.setFont("KappaSans", 7)
    c.drawString(3, 3, "Counts in parentheses")
    c.save()


FIGURE_NAMES = ("fig_judge_J1a.pdf", "fig_judge_J1b.pdf", "fig_judge_J1c.pdf")


def build(paths) -> list[Path]:
    """Write deterministic PDF renderings of the versioned audit summaries."""
    global FIGURES, DATA
    FIGURES = Path(paths.figures)
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA = json.loads((Path(__file__).parent / "assets" / "judge_kappa_summary.json").read_text())
    _register_fonts()
    _build()
    return [FIGURES / name for name in FIGURE_NAMES]


def _build():
    assert sum(x['n'] for x in DATA['cues'][:-1]) == 296
    draw("fig_judge_J1a.pdf", .36 * 5.5 * 72, 149, DATA['cues'],
         "Agreement by cue", "Nemotron · MMLU-Pro")
    draw("fig_judge_J1b.pdf", .49 * 5.5 * 72, 205, DATA['cues'],
         "Binary agreement by cue", "Primary judge versus arbiter")
    draw("fig_judge_J1c.pdf", .49 * 5.5 * 72, 205, DATA['roles'],
         "Role agreement", "One-vs-rest; all roles: five-class", roles=True)
