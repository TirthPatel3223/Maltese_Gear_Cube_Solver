"""
plot_training_metrics.py
========================
Parses a DeepCubeA-style training log and produces 5 publication-quality plots:

  Plot 1  — Train KL vs Update Cycle
  Plot 3a — % Solved Heatmap (all GBFS cycles × all back-steps)
  Plot 3b — % Solved Line chart (selected back-steps)
  Plot 4a — CTG Mean / Min / Max per Back-Step  (latest GBFS eval)
  Plot 4b — CTG Mean evolution over GBFS cycles (selected back-steps)

Usage:
    python plot_training_metrics.py training_log.txt
    python plot_training_metrics.py training_log.txt --outdir my_plots
"""

import re
import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless / server-safe; change to "TkAgg" for interactive
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ─────────────────────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "lines.linewidth": 1.8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# ─────────────────────────────────────────────────────────────────────────────
# PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_log(filepath: str) -> dict:
    """
    Parses the training log and returns a dict with keys:
        cycles          list[int]
        mean_kl         dict[cycle -> float]
        depth_kl        dict[cycle -> dict[depth -> float]]
        gbfs_cycles     list[int]
        gbfs_solve_pct  dict[cycle -> dict[back_step -> float]]
        gbfs_ctg_mean   dict[cycle -> dict[back_step -> float]]
        gbfs_ctg_min    dict[cycle -> dict[back_step -> float]]
        gbfs_ctg_max    dict[cycle -> dict[back_step -> float]]
    """
    data = {
        "cycles": [],
        "train_kl": {},
        "gbfs_cycles": [],
        "gbfs_solve_pct": {},
        "gbfs_ctg_mean": {},
        "gbfs_ctg_min": {},
        "gbfs_ctg_max": {},
    }

    current_cycle = None
    in_gbfs       = False

    with open(filepath, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            s = raw_line.strip()

            # ── Update Cycle header ──────────────────────────────────────────
            m = re.match(r"---\s*Update Cycle\s+(\d+)/\d+", s)
            if m:
                current_cycle = int(m.group(1))
                if current_cycle not in data["cycles"]:
                    data["cycles"].append(current_cycle)
                in_gbfs = False
                continue

            if current_cycle is None:
                continue

            # ── Train KL (epoch summary line) ────────────────────────────────
            m = re.match(r"Epoch\s+\d+/\d+\s+\|\s+Train KL:\s*([\d.eE+\-]+)", s)
            if m:
                data["train_kl"][current_cycle] = float(m.group(1))
                continue

            # ── GBFS evaluation start ────────────────────────────────────────
            if re.match(r"Running GBFS evaluation", s):
                in_gbfs = True
                if current_cycle not in data["gbfs_cycles"]:
                    data["gbfs_cycles"].append(current_cycle)
                for key in ("gbfs_solve_pct", "gbfs_ctg_mean", "gbfs_ctg_min", "gbfs_ctg_max"):
                    data[key].setdefault(current_cycle, {})
                continue

            # ── Individual GBFS back-step line ───────────────────────────────
            if in_gbfs:
                m = re.match(
                    r"Back Steps:\s*(\d+),\s*%Solved:\s*([\d.]+),.*?"
                    r"CTG Mean\(Std/Min/Max\):\s*([\d.]+)\(([\d.]+)/([\d.]+)/([\d.]+)\)",
                    s,
                )
                if m:
                    bs = int(m.group(1))
                    data["gbfs_solve_pct"][current_cycle][bs] = float(m.group(2))
                    data["gbfs_ctg_mean"][current_cycle][bs]  = float(m.group(3))
                    data["gbfs_ctg_min"][current_cycle][bs]   = float(m.group(5))
                    data["gbfs_ctg_max"][current_cycle][bs]   = float(m.group(6))
                    continue
                if "Summary" in s:
                    in_gbfs = False

    return data


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: str):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path}")


def _matrix(cycles, keys, lookup, fill=np.nan):
    """Build a 2-D numpy array  [cycles × keys]."""
    mat = np.full((len(cycles), len(keys)), fill)
    for ci, c in enumerate(cycles):
        for ki, k in enumerate(keys):
            val = lookup.get(c, {}).get(k, fill)
            mat[ci, ki] = val
    return mat


def _colorbar(ax, im, label=""):
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="3%", pad=0.08)
    plt.colorbar(im, cax=cax, label=label)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Train KL vs Update Cycle
# ─────────────────────────────────────────────────────────────────────────────

def plot_train_kl(data: dict, outdir: str):
    cycles = sorted(data["train_kl"].keys())
    if not cycles:
        print("  [skip] No train-KL data found.")
        return
    kls = [data["train_kl"][c] for c in cycles]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(cycles, kls, color="#2196F3", lw=1.6)
    ax.fill_between(cycles, kls, alpha=0.12, color="#2196F3")
    ax.set_xlabel("Update Cycle")
    ax.set_ylabel("Train KL Loss")
    ax.set_title("Train KL Loss vs Update Cycle")
    ax.set_xlim(cycles[0], cycles[-1])
    fig.tight_layout()
    _save(fig, os.path.join(outdir, "plot1_train_kl.png"))


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3a — % Solved per Back-Step Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_solve_pct_heatmap(data: dict, outdir: str):
    cycles = sorted(data["gbfs_cycles"])
    if not cycles:
        print("  [skip] No GBFS data found.")
        return
    all_bs = sorted({bs for c in cycles for bs in data["gbfs_solve_pct"].get(c, {})})
    mat = _matrix(cycles, all_bs, data["gbfs_solve_pct"])

    fig, ax = plt.subplots(figsize=(14, 6))
    extent = [cycles[0] - 0.5, cycles[-1] + 0.5,
              all_bs[0]  - 0.5, all_bs[-1]  + 0.5]
    im = ax.imshow(mat.T, aspect="auto", origin="lower",
                   cmap="RdYlGn", vmin=0, vmax=100, extent=extent)
    _colorbar(ax, im, "% Solved")
    ax.set_xlabel("Update Cycle")
    ax.set_ylabel("Back Steps (Scramble Depth)")
    ax.set_title("% Solved per Back-Step vs Update Cycle")
    ax.set_yticks([bs for bs in all_bs if bs % 5 == 0])
    fig.tight_layout()
    _save(fig, os.path.join(outdir, "plot3a_solve_pct_heatmap.png"))


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3b — % Solved Line Chart (selected back-steps)
# ─────────────────────────────────────────────────────────────────────────────

def plot_solve_pct_lines(data: dict, outdir: str):
    cycles = sorted(data["gbfs_cycles"])
    if not cycles:
        return
    all_bs = sorted({bs for c in cycles for bs in data["gbfs_solve_pct"].get(c, {})})
    step = max(1, len(all_bs) // 8)
    selected = all_bs[::step]
    if all_bs[-1] not in selected:
        selected.append(all_bs[-1])

    fig, ax = plt.subplots(figsize=(12, 5))
    palette = cm.tab10(np.linspace(0, 0.9, len(selected)))
    for color, bs in zip(palette, selected):
        pcts = [data["gbfs_solve_pct"].get(c, {}).get(bs, np.nan) for c in cycles]
        ax.plot(cycles, pcts, label=f"Back Step {bs}", color=color,
                marker="o", markersize=3)
    ax.set_xlabel("Update Cycle")
    ax.set_ylabel("% Solved")
    ax.set_title("% Solved vs Update Cycle (Selected Back-Steps)")
    ax.set_ylim(-2, 102)
    ax.set_xlim(cycles[0], cycles[-1])
    ax.legend(ncol=2, loc="upper right")
    fig.tight_layout()
    _save(fig, os.path.join(outdir, "plot3b_solve_pct_lines.png"))


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4a — CTG Mean / Min / Max per Back-Step (latest GBFS eval)
# ─────────────────────────────────────────────────────────────────────────────

def plot_ctg_snapshot(data: dict, outdir: str):
    cycles = sorted(data["gbfs_cycles"])
    if not cycles:
        return

    for cyc in cycles:          # produce one file per GBFS eval cycle
        bsteps = sorted(data["gbfs_ctg_mean"].get(cyc, {}).keys())
        if not bsteps:
            continue
        means = [data["gbfs_ctg_mean"][cyc][bs] for bs in bsteps]
        mins  = [data["gbfs_ctg_min"][cyc][bs]  for bs in bsteps]
        maxs  = [data["gbfs_ctg_max"][cyc][bs]  for bs in bsteps]

        fig, ax = plt.subplots(figsize=(13, 5))
        ax.fill_between(bsteps, mins, maxs, alpha=0.25, color="#2196F3",
                        label="Min–Max range")
        ax.plot(bsteps, means, color="#2196F3", lw=2.0, label="CTG Mean")
        ax.plot(bsteps, mins,  color="#4CAF50", lw=1.0,
                linestyle="--", alpha=0.8, label="CTG Min")
        ax.plot(bsteps, maxs,  color="#F44336", lw=1.0,
                linestyle="--", alpha=0.8, label="CTG Max")
        ax.plot(bsteps, bsteps, color="gray", lw=1.0,
                linestyle=":", alpha=0.6, label="Optimal (CTG = depth)")
        ax.set_xlabel("Back Steps (Scramble Depth)")
        ax.set_ylabel("Cost-to-Go (CTG)")
        ax.set_title(f"CTG per Back-Step — Update Cycle {cyc}")
        ax.set_xticks(bsteps)
        ax.legend(loc="upper left")
        fig.tight_layout()
        fname = f"plot4a_ctg_cycle_{cyc:05d}.png"
        _save(fig, os.path.join(outdir, fname))

    # Convenience: also save the latest snapshot as a fixed name
    latest = cycles[-1]
    src = os.path.join(outdir, f"plot4a_ctg_cycle_{latest:05d}.png")
    dst = os.path.join(outdir, "plot4a_ctg_latest.png")
    import shutil
    shutil.copy2(src, dst)
    print(f"  ✓  Latest CTG snapshot also saved as plot4a_ctg_latest.png")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4b — CTG Mean Evolution over GBFS Cycles (selected back-steps)
# ─────────────────────────────────────────────────────────────────────────────

def plot_ctg_evolution(data: dict, outdir: str):
    cycles = sorted(data["gbfs_cycles"])
    if len(cycles) < 2:
        print("  [skip] Not enough GBFS cycles for evolution plot.")
        return
    all_bs = sorted({bs for c in cycles for bs in data["gbfs_ctg_mean"].get(c, {})})
    step = max(1, len(all_bs) // 8)
    selected = all_bs[::step]
    if all_bs[-1] not in selected:
        selected.append(all_bs[-1])

    fig, ax = plt.subplots(figsize=(12, 5))
    palette = cm.tab10(np.linspace(0, 0.9, len(selected)))
    for color, bs in zip(palette, selected):
        means = [data["gbfs_ctg_mean"].get(c, {}).get(bs, np.nan) for c in cycles]
        ax.plot(cycles, means, label=f"Back Step {bs}", color=color,
                marker="o", markersize=3)
    ax.set_xlabel("Update Cycle")
    ax.set_ylabel("CTG Mean")
    ax.set_title("CTG Mean Evolution vs Update Cycle (Selected Back-Steps)")
    ax.set_xlim(cycles[0], cycles[-1])
    ax.legend(ncol=2, loc="upper left")
    fig.tight_layout()
    _save(fig, os.path.join(outdir, "plot4b_ctg_evolution.png"))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot DeepCubeA-style training metrics from a log file."
    )
    parser.add_argument("logfile", help="Path to the training log .txt file")
    parser.add_argument(
        "--outdir", default="plots",
        help="Directory to write PNG files (default: ./plots)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.logfile):
        sys.exit(f"[ERROR] Log file not found: {args.logfile}")

    os.makedirs(args.outdir, exist_ok=True)

    print(f"\nParsing: {args.logfile}")
    data = parse_log(args.logfile)

    n_cycles   = len(data["cycles"])
    n_kl       = len(data["train_kl"])
    n_gbfs     = len(data["gbfs_cycles"])
    print(f"  Update cycles parsed      : {n_cycles}")
    print(f"  Cycles with train KL      : {n_kl}")
    print(f"  GBFS evaluation cycles    : {n_gbfs}")

    if n_cycles == 0:
        sys.exit("[ERROR] No update cycles found. Check the log file format.")

    print(f"\nGenerating plots → {args.outdir}/")

    plot_train_kl(data, args.outdir)
    plot_solve_pct_heatmap(data, args.outdir)
    plot_solve_pct_lines(data, args.outdir)
    plot_ctg_snapshot(data, args.outdir)
    plot_ctg_evolution(data, args.outdir)

    print("\nDone. All plots saved.\n")


if __name__ == "__main__":
    main()
