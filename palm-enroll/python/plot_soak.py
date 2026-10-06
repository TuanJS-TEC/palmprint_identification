"""Draw the RSS curves of a palm_soak run as two SVG charts (no third-party
packages: the machine has no matplotlib).

  <out>_rss.svg    process RSS (VmRSS) and heap in use (malloc) over the frames
  <out>_delta.svg  RSS change against the first steady-state sample, zoomed

Usage:
  python plot_soak.py SOAK.csv OUT_PREFIX [--every 1000] [--steady-from 10000]
"""
import argparse
import csv
import math

W, H = 760, 340
ML, MR, MT, MB = 64, 128, 76, 48  # plot margins
# plain colours, no CSS variables: some SVG renderers (librsvg) do not know var()
STYLE = """<style>
  text { font-family:system-ui,-apple-system,"Segoe UI",sans-serif; }
  .bg { fill:#fcfcfb; } .title { fill:#0b0b0b; font-size:15px; font-weight:600; }
  .sub { fill:#52514e; font-size:12px; } .tick { fill:#898781; font-size:11px; font-variant-numeric:tabular-nums; }
  .lab { fill:#52514e; font-size:12px; } .grid { stroke:#e1e0d9; stroke-width:1; }
  .axis { stroke:#c3c2b7; stroke-width:1; } .mark { stroke:#898781; stroke-width:1; }
  .line { fill:none; stroke-width:2; stroke-linejoin:round; stroke-linecap:round; }
  .s1 { stroke:#2a78d6; } .s2 { stroke:#eb6834; } .f1 { fill:#2a78d6; } .f2 { fill:#eb6834; }
  .ring { stroke:#fcfcfb; stroke-width:2; } .hit { fill:transparent; }
  @media (prefers-color-scheme: dark) {
    .bg { fill:#1a1a19; } .title { fill:#ffffff; } .sub, .lab { fill:#c3c2b7; }
    .grid { stroke:#2c2c2a; } .axis { stroke:#383835; }
    .s1 { stroke:#3987e5; } .s2 { stroke:#d95926; } .f1 { fill:#3987e5; } .f2 { fill:#d95926; }
    .ring { stroke:#1a1a19; }
  }
</style>"""


def vn(x, digits=0):
    """Vietnamese number format: 100.000 / 541,2"""
    s = f"{abs(x):,.{digits}f}".replace(",", "_").replace(".", ",").replace("_", ".")
    return "\u2212" + s if x < 0 and s.strip("0,.") else s


def nice_step(span, n=5):
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    return next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)


def chart(rows, series, title, subtitle, ymin, ymax, ystep, ydigits, steady_from, path, legend=True):
    xmax = max(r["frame"] for r in rows)
    pw, ph = W - ML - MR, H - MT - MB
    sx = lambda x: ML + pw * x / xmax
    sy = lambda y: MT + ph * (1 - (y - ymin) / (ymax - ymin))
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
         f'aria-label="{title}">', STYLE, f'<rect class="bg" width="{W}" height="{H}" rx="8"/>',
         f'<text class="title" x="{ML}" y="24">{title}</text>', f'<text class="sub" x="{ML}" y="42">{subtitle}</text>']
    if legend:
        x = ML
        for cls, name, _ in series:
            o.append(f'<line class="line {cls}" x1="{x}" y1="60" x2="{x + 18}" y2="60"/>')
            o.append(f'<text class="lab" x="{x + 24}" y="64">{name}</text>')
            x += 24 + 7.2 * len(name) + 24
    k = 0
    while ymin + k * ystep <= ymax + 1e-9:  # y grid
        y = ymin + k * ystep
        cls = "axis" if abs(y) < 1e-9 or k == 0 else "grid"
        o.append(f'<line class="{cls}" x1="{ML}" y1="{sy(y):.1f}" x2="{ML + pw}" y2="{sy(y):.1f}"/>')
        o.append(f'<text class="tick" x="{ML - 8}" y="{sy(y) + 4:.1f}" text-anchor="end">{vn(y, ydigits)}</text>')
        k += 1
    for x in range(0, xmax + 2, 20000):  # x ticks (xmax = n - 1)
        o.append(f'<text class="tick" x="{sx(x):.1f}" y="{MT + ph + 18}" text-anchor="middle">{vn(x)}</text>')
    o.append(f'<text class="tick" x="{ML + pw}" y="{MT + ph + 36}" text-anchor="end">frame</text>')
    xs = sx(steady_from)  # start of the steady state
    o.append(f'<line class="mark" x1="{xs:.1f}" y1="{MT}" x2="{xs:.1f}" y2="{MT + ph}"/>')
    o.append(f'<text class="tick" x="{xs + 4:.1f}" y="{MT + 12}">ổn định từ frame {vn(steady_from)}</text>')
    for cls, name, key in series:
        pts = " ".join(f"{sx(r['frame']):.1f},{sy(r[key]):.1f}" for r in rows)
        o.append(f'<polyline class="line {cls}" points="{pts}"/>')
        for r in rows:  # native tooltips when the SVG is opened on its own
            o.append(f'<circle class="hit" cx="{sx(r["frame"]):.1f}" cy="{sy(r[key]):.1f}" r="6">'
                     f'<title>{name}: frame {vn(r["frame"])}: {vn(r[key], 2)} MB</title></circle>')
        last = rows[-1]
        cx, cy = sx(last["frame"]), sy(last[key])
        o.append(f'<circle class="ring f{cls[1]}" cx="{cx:.1f}" cy="{cy:.1f}" r="4"/>')
        o.append(f'<text class="lab" x="{cx + 10:.1f}" y="{cy + 4:.1f}">{vn(last[key], ydigits + 1)} MB</text>')
    o.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(o) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("out")
    ap.add_argument("--every", type=int, default=1000)
    ap.add_argument("--steady-from", type=int, default=10000)
    a = ap.parse_args()

    rows = []
    for r in csv.DictReader(open(a.csv)):
        fr = int(r["frame"])
        rows.append({"frame": fr, "rss": float(r["rss_mb"]), "live": float(r["heap_live_mb"])})
    last = rows[-1]
    rows = [r for r in rows if r["frame"] % a.every == 0]
    if rows[-1]["frame"] != last["frame"]:
        rows.append(last)
    n = last["frame"] + 1

    top = max(r["rss"] for r in rows)
    step = nice_step(top * 1.15)
    chart(rows, [("s1", "RSS tiến trình (VmRSS)", "rss"), ("s2", "Heap đang dùng (malloc)", "live")],
          "RSS theo số frame, soak " + vn(n) + " frame", f"MB, mẫu mỗi {vn(a.every)} frame",
          0, math.ceil(top * 1.15 / step) * step, step, 0, a.steady_from, a.out + "_rss.svg")

    base = next(r for r in rows if r["frame"] >= a.steady_from)
    steady = [dict(r, d=r["rss"] - base["rss"]) for r in rows if r["frame"] >= a.steady_from]
    span = max(1.0, max(abs(r["d"]) for r in steady))
    lim = math.ceil(span / nice_step(span, 2)) * nice_step(span, 2)
    chart(steady, [("s1", "RSS so với frame " + vn(base["frame"]), "d")],
          "Thay đổi RSS so với frame " + vn(base["frame"]), "MB, giai đoạn ổn định",
          -lim, lim, lim / 2, 1, a.steady_from, a.out + "_delta.svg", legend=False)
    print(f"{a.out}_rss.svg, {a.out}_delta.svg: {len(rows)} samples; steady RSS change "
          f"{min(r['d'] for r in steady):+.2f}..{max(r['d'] for r in steady):+.2f} MB")


if __name__ == "__main__":
    main()
