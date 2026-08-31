from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "jit_apfrl_mplconfig")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "figure_data"
FIG_DIR = (
    Path(os.environ.get("JIT_APFRL_FIGURE_OUTPUT", str(ROOT / "generated_figures")))
    / "fig10"
)

METHODS = ["Async-FedDRL", "JIT-APFRL"]
SEEDS_5 = [3407, 4518, 5629, 6740, 7851]
COLORS = {
    "FedPPO": "#646C70",
    "DP-FedRL": "#4C97C9",
    "Async-FedDRL": "#8C78C6",
    "JIT-APFRL": "#E96B6C",
    "risk": "#B85C38",
    "async_response": "#D07A35",
    "preferred": "#5F9363",
    "text": "#222222",
    "axis": "#555555",
    "grid_major": "#E4E4E4",
    "grid_minor": "#F0F0F0",
    "ref_line": "#888888",
    "background": "#FFFFFF",
}
FILES = {
    "A": ("fig10A_high_straggler_reward.csv", "mean_reward"),
    "B": ("fig10B_high_straggler_p95_jit.csv", "p95_jit_violation_ms"),
    "C": (
        "fig10C_cross_seed_relative_update_dispersion.csv",
        "relative_deviation_percent",
    ),
    "D": ("fig10D_high_straggler_approx_kl.csv", "mean_approx_kl"),
    "E": ("fig10E_high_straggler_policy_entropy.csv", "mean_entropy"),
    "F": ("fig10F_high_straggler_resource_allocation.csv", "mean_allocated_resource"),
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "mathtext.fontset": "stix",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.edgecolor": COLORS["axis"],
        "axes.labelcolor": COLORS["text"],
        "xtick.color": COLORS["axis"],
        "ytick.color": COLORS["axis"],
        "legend.frameon": False,
        "figure.facecolor": COLORS["background"],
        "savefig.facecolor": COLORS["background"],
    }
)


def read_csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(PKG / rel)


def seed_offset(seed, seeds, width):
    positions = np.linspace(-width, width, len(seeds))
    return float(positions[seeds.index(int(seed))])


def prepare_phase(panel: str) -> pd.DataFrame:
    file = {
        "D": "fig10D_fixed_100_round_phase.csv",
        "F": "fig10F_fixed_100_round_phase.csv",
    }[panel]
    return read_csv(f"fig10/{file}")


def style_ax(ax, letter: str, title: str, xlabel: str, ylabel: str):
    ax.text(
        -0.12,
        1.09,
        f"({letter.lower()})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["text"],
    )
    ax.set_title(
        title, loc="left", fontsize=9.1, fontweight="bold", pad=5, color=COLORS["text"]
    )
    ax.set_xlabel(xlabel, fontsize=8.7)
    ax.set_ylabel(ylabel, fontsize=8.7)
    ax.tick_params(labelsize=7.8, length=2.5, width=0.6)
    ax.grid(axis="y", color=COLORS["grid_major"], lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(COLORS["axis"])
    ax.spines["bottom"].set_color(COLORS["axis"])
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)


def draw_trajectory(ax, panel: str):
    name, metric = FILES[panel]
    df = read_csv(f"fig10/{name}")
    x = np.arange(1, 401)
    for m in METHODS:
        arr = (
            df[df.method == m]
            .sort_values(["seed", "round"])
            .pivot(index="seed", columns="round", values=metric)
            .to_numpy(float)
        )
        for y in arr:
            ax.plot(x, y, color=COLORS[m], lw=0.72, alpha=0.24, zorder=1)
        mu, sd = arr.mean(axis=0), arr.std(axis=0, ddof=1)
        if panel != "C":
            ax.fill_between(
                x, mu - sd, mu + sd, color=COLORS[m], alpha=0.10, lw=0, zorder=2
            )
        ax.plot(x, mu, color=COLORS[m], lw=1.82, zorder=3)
    ax.set_xlim(1, 400)
    ax.set_xticks([1, 100, 200, 300, 400])


def draw_phase(ax, panel: str):
    df = prepare_phase(panel)
    x = np.arange(1, 5)
    offsets = {METHODS[0]: -0.13, METHODS[1]: 0.13}
    for m in METHODS:
        for phase in range(1, 5):
            points = df[(df.method == m) & (df.phase == phase)].copy()
            y = points["phase_mean"].astype(float).to_numpy()
            jitter = np.array(
                [seed_offset(seed, SEEDS_5, 0.045) for seed in points["seed"]]
            )
            xx = np.full(len(points), phase) + offsets[m] + jitter
            ax.scatter(
                xx, y, s=15, color=COLORS[m], alpha=0.82, edgecolors="none", zorder=3
            )
            mean, sd = y.mean(), y.std(ddof=1)
            ax.errorbar(
                phase + offsets[m],
                mean,
                yerr=sd,
                fmt="D",
                ms=4.8,
                mfc="white",
                mec=COLORS[m],
                mew=0.85,
                ecolor=COLORS[m],
                elinewidth=0.8,
                capsize=2,
                zorder=4,
            )
    ax.set_xlim(0.45, 4.55)
    ax.set_xticks(x, ["1–100", "101–200", "201–300", "301–400"])
    if panel == "D" and (df.phase_mean < 0).any():
        ax.axhline(0, color=COLORS["ref_line"], lw=0.7, zorder=1)


def render() -> dict:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 3, figsize=(7.4, 4.9), constrained_layout=False)
    specs = [
        ("A", "Reward dynamics", "Reward", "Training round"),
        ("B", "Tail JIT risk", "P95 JIT violation (ms)", "Training round"),
        (
            "C",
            "Client-update dispersion",
            "Relative update deviation (%)",
            "Training round",
        ),
        (
            "D",
            "PPO approximate-KL estimate",
            "PPO approximate-KL estimate",
            "Training phase",
        ),
        ("E", "Policy entropy", "Policy entropy", "Training round"),
        (
            "F",
            "Resource-allocation behavior",
            "Mean allocated resource",
            "Training phase",
        ),
    ]
    for ax, (p, title, ylabel, xlabel) in zip(axs.flat, specs):
        if p in "ABCE":
            draw_trajectory(ax, p)
        else:
            draw_phase(ax, p)
        style_ax(ax, p, title, xlabel, ylabel)
    handles = [Line2D([0], [0], color=COLORS[m], lw=1.8, label=m) for m in METHODS]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        fontsize=7.8,
        handlelength=2.2,
        columnspacing=1.6,
    )
    fig.subplots_adjust(
        left=0.095, right=0.985, bottom=0.105, top=0.86, wspace=0.34, hspace=0.47
    )
    base = FIG_DIR / "Fig10_final"
    fig.savefig(
        base.with_suffix(".pdf"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="#FFFFFF",
    )
    fig.savefig(
        base.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="#FFFFFF",
    )
    fig.savefig(
        base.with_suffix(".svg"),
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="#FFFFFF",
    )
    plt.close(fig)


if __name__ == "__main__":
    render()
