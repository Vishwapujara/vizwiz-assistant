"""Load-test report: tables, throughput at the latency target, plots (Step 6.1-6.2).

    python -m vizwiz.bench.report --sweep results/bench/load-test/sweep.jsonl \\
        --out results/bench/load-test

Reads the committed sweep rows (aggregate numbers only) and writes ``summary.json``,
``summary.md`` and two plots. The latency target is p95 < 2 s (PROJECT_PLAN 12.3).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SLO_P95_MS = 2000
ORDER = ("e5-parallel_checks", "e5-sequential", "e5-speculative", "specialists-parallel_checks")
NAMES = {
    "e5-parallel_checks": "E5, parallel checks (served)",
    "e5-sequential": "E5, sequential",
    "e5-speculative": "E5, speculative",
    "specialists-parallel_checks": "3 specialists, parallel",
}
SHORT = {
    "e5-parallel_checks": "E5 parallel",
    "e5-sequential": "E5 sequential",
    "e5-speculative": "E5 speculative",
    "specialists-parallel_checks": "specialists",
}
# Categorical slots 1-4 of the reference palette (validated: dataviz validate_palette.js)
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
MARKERS = ("o", "s", "^", "D")  # secondary encoding: identity is never colour alone
INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def load(path: Path) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {}
    for line in path.open():
        r = json.loads(line)
        rows.setdefault(r["label"], []).append(r)
    return {k: sorted(v, key=lambda r: r["users"]) for k, v in rows.items()}


def at_slo(rows: list[dict], slo_ms: float = SLO_P95_MS) -> dict | None:
    """Highest-throughput level whose p95 latency is under the target."""
    ok = [r for r in rows if r.get("p95_ms") is not None and r["p95_ms"] < slo_ms]
    return max(ok, key=lambda r: r["rps"]) if ok else None


def summarise(runs: dict[str, list[dict]]) -> dict:
    out = {}
    for label, rows in runs.items():
        one = rows[0]
        slo = at_slo(rows)
        peak = max(rows, key=lambda r: r["rps"])
        out[label] = {
            "single_user": {k: one.get(k) for k in ("p50_ms", "p95_ms", "rps")},
            "at_slo": None
            if slo is None
            else {k: slo.get(k) for k in ("users", "rps", "p50_ms", "p95_ms")},
            "peak": {k: peak.get(k) for k in ("users", "rps", "p95_ms")},
            "failures": sum(r["failures"] for r in rows),
            "requests": sum(r["requests"] for r in rows),
            "kv_cache_peak_pct": max((r["kv_cache_peak_pct"] or 0) for r in rows),
        }
    return out


def to_markdown(runs: dict[str, list[dict]], summary: dict) -> str:
    labels = [x for x in ORDER if x in runs]
    lines = [
        "| Configuration | 1 user p50 / p95 (ms) | Best req/s with p95 < 2 s "
        "| Peak req/s | Failures |",
        "|---|---|---|---|---|",
    ]
    for lab in labels:
        s = summary[lab]
        slo = s["at_slo"]
        slo_txt = (
            "none"
            if slo is None
            else f"{slo['rps']:.2f} at {slo['users']} users (p95 {slo['p95_ms']:.0f})"
        )
        lines.append(
            f"| {NAMES[lab]} | {s['single_user']['p50_ms']:.0f} / {s['single_user']['p95_ms']:.0f} "
            f"| {slo_txt} | {s['peak']['rps']:.2f} at {s['peak']['users']} | "
            f"{s['failures']} of {s['requests']} |"
        )
    lines += [
        "",
        "| Users | " + " | ".join(NAMES[x] for x in labels) + " |",
        "|---|" + "---|" * len(labels),
    ]
    users = sorted({r["users"] for x in labels for r in runs[x]})
    for u in users:
        cells = []
        for lab in labels:
            r = next((r for r in runs[lab] if r["users"] == u), None)
            cells.append(
                ""
                if r is None
                else f"{r['rps']:.2f} req/s, p50 {r['p50_ms']:.0f}, p95 {r['p95_ms']:.0f}"
            )
        lines.append(f"| {u} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def plot(runs: dict[str, list[dict]], out: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [x for x in ORDER if x in runs]
    files = []
    specs = (
        ("throughput", "rps", "Requests per second", None),
        ("latency_p95", "p95_ms", "p95 latency (s)", SLO_P95_MS),
    )
    for name, key, ylabel, slo in specs:
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)
        ends = []
        for i, lab in enumerate(labels):
            xs = [r["users"] for r in runs[lab]]
            ys = [r[key] / (1000 if key.endswith("_ms") else 1) for r in runs[lab]]
            ax.plot(
                xs,
                ys,
                color=COLORS[i],
                lw=2,
                marker=MARKERS[i],
                ms=6,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                label=NAMES[lab],
            )
            ends.append((ys[-1], xs[-1], SHORT[lab]))
        gap, last = 0.06 * max(y for y, _, _ in ends), None
        for y, x, text in sorted(ends):  # bottom to top, pushed up to keep a gap
            y_txt = y if last is None else max(y, last + gap)
            ax.annotate(
                text, (x, y), xytext=(x * 1.12, y_txt), va="center", fontsize=7.5, color=INK
            )
            last = y_txt
        if slo:
            ax.axhline(slo / 1000, color=MUTED, lw=1, ls=(0, (4, 3)))
            ax.annotate(
                "target: p95 < 2 s",
                (1, slo / 1000),
                xytext=(2, 4),
                textcoords="offset points",
                fontsize=7.5,
                color=MUTED,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32], labels=["1", "2", "4", "8", "16", "32"])
        ax.set_xlim(0.85, 75)
        ax.set_ylim(0, None)
        ax.set_xlabel("Concurrent users (closed loop)", color=MUTED, fontsize=9)
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.6)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
        fig.tight_layout()
        path = out / "plots" / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, facecolor=SURFACE)
        plt.close(fig)
        files.append(path)
    return files


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Load-test report")
    p.add_argument("--sweep", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args(argv)
    runs = load(args.sweep)
    summary = summarise(runs)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "summary.md").write_text(to_markdown(runs, summary))
    if not args.no_plots:
        for f in plot(runs, args.out):
            print("wrote", f)
    print(to_markdown(runs, summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
