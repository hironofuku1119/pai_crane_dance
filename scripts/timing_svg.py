"""折れ線グラフを SVG で書く（標準ライブラリのみ・ROS なし）。scripts/analyze_timing.py が使う。

色は Okabe–Ito の配色（色覚の違いに強い 8 色の組）から選んだ。
"""
from __future__ import annotations

import math
from xml.sax.saxutils import escape

COLORS = {"choreo": "#999999", "plan": "#0072B2", "actual": "#E69F00", "ideal": "#999999",
          "j1": "#0072B2", "j2": "#E69F00", "j3": "#009E73", "j4": "#CC79A7", "hand": "#D55E00"}
FONT = "font-family='Hiragino Sans, Noto Sans CJK JP, sans-serif'"


def _ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 5, 10) if m * mag >= raw), default=raw)
    start = math.ceil(lo / step) * step
    out = []
    v = start
    while v <= hi + 1e-12:
        out.append(round(v, 10))
        v += step
    return out


def _fmt(v: float) -> str:
    return f"{v:g}"


def panels_svg(panels: list[dict], width: int = 900, panel_h: int = 170, title: str = "") -> str:
    """panels: [{"title", "xlabel", "ylabel", "series": [{"x", "y", "color", "dash", "label"}],
    "ylim": (lo, hi) か None, "xlim": (lo, hi) か None}]"""
    ml, mr, mt, mb = 70, 20, 40 if title else 16, 42
    gap = 28
    h = mt + len(panels) * (panel_h + gap) + mb - gap
    out = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{h}' viewBox='0 0 {width} {h}'>",
           f"<rect width='{width}' height='{h}' fill='white'/>"]
    if title:
        out.append(f"<text x='{ml}' y='24' {FONT} font-size='16' font-weight='bold'>{escape(title)}</text>")
    pw = width - ml - mr
    for i, p in enumerate(panels):
        top = mt + i * (panel_h + gap)
        xs = [x for s in p["series"] for x in s["x"]]
        ys = [y for s in p["series"] for y in s["y"]]
        if not xs:
            continue
        x0, x1 = p.get("xlim") or (min(xs), max(xs))
        y0, y1 = p.get("ylim") or (min(ys), max(ys))
        span = max(p.get("min_span", 0.0), 1e-9)
        if y1 - y0 < span:
            mid = (y0 + y1) / 2
            y0, y1 = mid - span / 2, mid + span / 2
        pad = (y1 - y0) * 0.06
        y0, y1 = y0 - pad, y1 + pad
        if x1 - x0 < 1e-9:
            x1 = x0 + 1

        def sx(v):
            return ml + (v - x0) / (x1 - x0) * pw

        def sy(v):
            return top + panel_h - (v - y0) / (y1 - y0) * panel_h

        out.append(f"<rect x='{ml}' y='{top}' width='{pw}' height='{panel_h}' fill='none' stroke='#cccccc'/>")
        for t in _ticks(y0, y1, 4):
            yy = sy(t)
            out.append(f"<line x1='{ml}' y1='{yy:.1f}' x2='{ml + pw}' y2='{yy:.1f}' stroke='#eeeeee'/>")
            out.append(f"<text x='{ml - 6}' y='{yy + 4:.1f}' {FONT} font-size='11' text-anchor='end'>{_fmt(t)}</text>")
        for t in _ticks(x0, x1, 8):
            xx = sx(t)
            out.append(f"<line x1='{xx:.1f}' y1='{top}' x2='{xx:.1f}' y2='{top + panel_h}' stroke='#f3f3f3'/>")
            out.append(f"<text x='{xx:.1f}' y='{top + panel_h + 14}' {FONT} font-size='11' text-anchor='middle'>{_fmt(t)}</text>")
        out.append(f"<text x='{ml + 6}' y='{top + 14}' {FONT} font-size='12' font-weight='bold'>{escape(p.get('title', ''))}</text>")
        if p.get("ylabel"):
            out.append(f"<text x='14' y='{top + panel_h / 2:.1f}' {FONT} font-size='11' "
                       f"transform='rotate(-90 14 {top + panel_h / 2:.1f})' text-anchor='middle'>{escape(p['ylabel'])}</text>")
        if p.get("xlabel"):
            out.append(f"<text x='{ml + pw / 2:.1f}' y='{top + panel_h + 30}' {FONT} font-size='11' "
                       f"text-anchor='middle'>{escape(p['xlabel'])}</text>")
        lx = ml + pw - 8
        for j, s in enumerate(p["series"]):
            pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(s["x"], s["y"]))
            dash = f" stroke-dasharray='{s['dash']}'" if s.get("dash") else ""
            out.append(f"<polyline points='{pts}' fill='none' stroke='{s['color']}' stroke-width='1.6'{dash}/>")
            if s.get("label"):
                ly = top + 16 + j * 15
                out.append(f"<line x1='{lx - 150}' y1='{ly - 4}' x2='{lx - 128}' y2='{ly - 4}' stroke='{s['color']}' stroke-width='2'{dash}/>")
                out.append(f"<text x='{lx - 122}' y='{ly}' {FONT} font-size='11'>{escape(s['label'])}</text>")
    out.append("</svg>")
    return "\n".join(out)
