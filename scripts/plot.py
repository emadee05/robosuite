"""Plots for a uniform-vs-SGS comparison.

    python scripts/plot.py runs/wide_uniform_s0 runs/wide_sgs_s0
    python scripts/plot.py runs/wide_*            # seeds are averaged per sampler

Produces two files in --out:
  curves.png   eval success and train success vs environment steps
  sampling.png per-configuration success rate over training (one panel per run)
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Categorical slots 1 and 2 of the reference palette (documented as all-pairs
# CVD-validated); series colour follows the sampler, never the plotting order.
SERIES = {"uniform": "#2a78d6", "sgs": "#eb6834"}
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#dcdbd6"


def style(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)


def load(run_dir: Path):
    df = pd.read_csv(run_dir / "log.csv")
    args = json.loads((run_dir / "args.json").read_text())
    return df, args


def agg(dfs, col):
    """Mean +/- across seeds on a common step grid."""
    grid = dfs[0]["global_step"].values
    for d in dfs[1:]:
        grid = grid[: len(d)]
    vals = np.stack([d[col].values[: len(grid)] for d in dfs])
    return grid, np.nanmean(vals, axis=0), np.nanstd(vals, axis=0), vals.shape[0]


def curves(groups, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), facecolor="#fcfcfb")
    panels = [
        ("eval_success", "Evaluation success rate",
         "Mean success over the whole configuration pool,\nuniform, deterministic actions - the headline number"),
        ("train_success", "Training success rate",
         "Success on the episodes actually sampled.\nNot comparable across samplers: SGS trains on hard tasks by design"),
    ]
    for ax, (col, title, sub) in zip(axes, panels):
        style(ax)
        for name, dfs in groups.items():
            c = SERIES.get(name.split("_")[0], "#6250d6")
            x, m, s, k = agg(dfs, col)
            ok = ~np.isnan(m)
            ax.plot(x[ok], m[ok], color=c, linewidth=2.0, label=f"{name} (n={k})",
                    solid_capstyle="round")
            if k > 1:
                ax.fill_between(x[ok], (m - s)[ok], (m + s)[ok], color=c, alpha=0.15,
                                linewidth=0)
            if ok.any():  # direct label at the end of the line
                ax.annotate(f"{m[ok][-1]:.2f}", (x[ok][-1], m[ok][-1]), color=c,
                            fontsize=9, fontweight="bold",
                            xytext=(5, 0), textcoords="offset points", va="center")
        ax.set_title(title, color=INK, fontsize=11, fontweight="bold", loc="left")
        ax.text(0, 1.02, sub, transform=ax.transAxes, color=INK_2, fontsize=8,
                va="bottom", ha="left")
        ax.set_xlabel("environment steps", color=INK_2, fontsize=9)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    fig.subplots_adjust(top=0.78, wspace=0.25)
    fig.savefig(out / "curves.png", dpi=160, bbox_inches="tight",
                facecolor="#fcfcfb")
    print("wrote", out / "curves.png")


def sampling(runs, out):
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
    fig, axes = plt.subplots(1, len(runs), figsize=(5.5 * len(runs), 4.0),
                             squeeze=False, facecolor="#fcfcfb")
    for ax, rd in zip(axes[0], runs):
        p = np.load(rd / "p_history.npy")              # (updates, n_configs)
        d = np.load(rd / "configs.npz")["difficulty"]  # cube distance from centre
        order = np.argsort(d)                          # easy at top, hard at bottom
        im = ax.imshow(p[:, order].T, aspect="auto", origin="upper", cmap=cmap,
                       vmin=0, vmax=1, interpolation="nearest")
        style(ax)
        ax.grid(False)
        ax.set_title(rd.name, color=INK, fontsize=11, fontweight="bold", loc="left")
        ax.set_xlabel("PPO update", color=INK_2, fontsize=9)
        ax.set_ylabel("configuration (sorted easy -> hard)", color=INK_2, fontsize=9)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("success rate over last H episodes", color=INK_2, fontsize=8)
        cb.ax.tick_params(colors=INK_2, labelsize=8)
    fig.suptitle("Per-configuration success rate, p_i", color=INK, fontsize=12,
                 fontweight="bold", x=0.01, ha="left")
    fig.savefig(out / "sampling.png", dpi=160, bbox_inches="tight",
                facecolor="#fcfcfb")
    print("wrote", out / "sampling.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("runs"))
    args = ap.parse_args()
    runs = [r for r in args.runs if (r / "log.csv").exists()]
    if not runs:
        raise SystemExit("no runs with log.csv found")
    groups = {}
    for rd in runs:
        df, a = load(rd)
        groups.setdefault(a["sampler"], []).append(df)
    args.out.mkdir(parents=True, exist_ok=True)
    curves(groups, args.out)
    sampling(runs, args.out)


if __name__ == "__main__":
    main()
