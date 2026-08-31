from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "jit_apfrl_mplconfig")
)
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = [
    "Times New Roman",
    "Times",
    "Liberation Serif",
    "DejaVu Serif",
]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update(
    {
        "font.size": 8.0,
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#505050",
        "axes.labelcolor": "#242424",
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "figure_data"
OUT = Path(os.environ.get("JIT_APFRL_FIGURE_OUTPUT", str(ROOT / "generated_figures")))
METHODS = ["FedPPO", "DP-FedRL", "Async-FedDRL", "JIT-APFRL"]
METHOD_COLORS = {
    "FedPPO": "#646C70",
    "DP-FedRL": "#4C97C9",
    "Async-FedDRL": "#8C78C6",
    "JIT-APFRL": "#E96B6C",
}
METHOD_STYLE = {
    "FedPPO": {"color": "#646C70", "linestyle": "-", "marker": "o"},
    "DP-FedRL": {"color": "#4C97C9", "linestyle": "--", "marker": "s"},
    "Async-FedDRL": {"color": "#8C78C6", "linestyle": "-.", "marker": "^"},
    "JIT-APFRL": {"color": "#E96B6C", "linestyle": "-", "marker": "D"},
}
VARIANT_COLORS = {
    "No MAD": "#5D6570",
    "Local only": "#4C97C9",
    "No adaptive privacy": "#D07A35",
    "No async": "#8C78C6",
}
COMP_COLORS = {
    "weighted_jit": "#B85C38",
    "weighted_staleness": "#D07A35",
    "weighted_concentration": "#4C97C9",
}
SEEDS_3 = [3407, 4518, 5629]
SEEDS_5 = [3407, 4518, 5629, 6740, 7851]


def read(rel):
    with (DATA / rel).open(newline="") as f:
        return list(csv.DictReader(f))


def val(r, k):
    return float(r[k])


def seed_offset(seed, seeds, width):
    positions = np.linspace(-width, width, len(seeds))
    return float(positions[seeds.index(int(seed))])


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / name
    fig.savefig(
        str(base) + ".pdf",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="#FFFFFF",
    )
    fig.savefig(
        str(base) + ".svg", bbox_inches="tight", pad_inches=0.02, facecolor="#FFFFFF"
    )
    fig.savefig(
        str(base) + ".png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="#FFFFFF",
    )
    plt.close(fig)


def label(ax, letter, title):
    ax.text(
        -0.13,
        1.08,
        f"({letter})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=9,
    )
    ax.set_title(title, loc="left", fontsize=8.3, fontweight="bold", pad=5)
    ax.grid(axis="y", color="#E7E7E7", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6.8, length=2.5, width=0.55)


def finish_axes(ax):
    ax.margins(x=0.02)
    for s in ax.spines.values():
        s.set_linewidth(0.7)


def group_rounds(rows, group="method", key="mean_reward"):
    out = {}
    for r in rows:
        out.setdefault((r[group], int(r["seed"])), {})[int(r["round"])] = val(r, key)
    return out


def trajectory(
    ax, rows, key, group, order, colors, xlabel="Training round", ylabel=None
):
    grouped = group_rounds(rows, group, key)
    for g in order:
        series = [
            np.array([d[i] for i in range(1, 401)])
            for (gg, _), d in grouped.items()
            if gg == g
        ]
        arr = np.vstack(series)
        c = colors[g]
        for y in arr:
            ax.plot(np.arange(1, 401), y, color=c, lw=0.45, alpha=0.16, zorder=1)
        mu, sd = arr.mean(0), arr.std(0)
        ax.fill_between(
            np.arange(1, 401), mu - sd, mu + sd, color=c, alpha=0.14, lw=0, zorder=2
        )
        ax.plot(
            np.arange(1, 401),
            mu,
            color=c,
            lw=1.75,
            linestyle=METHOD_STYLE.get(g, {}).get("linestyle", "-"),
            zorder=3,
            label=g,
        )
    ax.set_xlim(1, 400)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    finish_axes(ax)


def shared_legend(fig, labels, colors, y=1.01, ncol=None):
    hs = [Line2D([0], [0], color=colors[x], lw=1.7, label=x) for x in labels]
    fig.legend(
        handles=hs,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol or len(labels),
        fontsize=6.8,
        handlelength=2.2,
        columnspacing=1.2,
    )


def dot_interval(
    ax, rows, key, order, colors, ylabel, title, letter, xlabels=None, seeds=SEEDS_5
):
    xs = np.arange(len(order))
    for i, g in enumerate(order):
        points = [r for r in rows if r.get("method", r.get("variant")) == g]
        ys = [val(r, key) for r in points]
        jitter = np.array([seed_offset(r["seed"], seeds, 0.075) for r in points])
        ax.scatter(
            np.full(len(points), i) + jitter,
            ys,
            s=17,
            color=colors[g],
            alpha=0.82,
            zorder=3,
        )
        ax.errorbar(
            i,
            np.mean(ys),
            yerr=np.std(ys, ddof=1),
            fmt="D",
            ms=4.8,
            mfc="white",
            mec=colors[g],
            mew=0.8,
            ecolor=colors[g],
            elinewidth=0.8,
            capsize=2,
            zorder=4,
        )
    ax.set_xticks(xs, xlabels or order, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    label(ax, letter, title)
    finish_axes(ax)


def fig6():
    fig, axs = plt.subplots(2, 2, figsize=(7.25, 5.15), constrained_layout=True)
    rows = read("fig6/fig6A_reward_convergence.csv")
    trajectory(
        axs[0, 0],
        rows,
        "mean_reward",
        "method",
        METHODS,
        METHOD_COLORS,
        ylabel="Mean reward",
    )
    label(axs[0, 0], "A", "Training reward convergence")
    dot_interval(
        axs[0, 1],
        read("fig6/fig6B_terminal_mean_jit.csv"),
        "terminal_mean_jit_ms",
        METHODS,
        METHOD_COLORS,
        "Mean JIT violation (ms)",
        "Terminal mean JIT risk",
        "B",
    )
    dot_interval(
        axs[1, 0],
        read("fig6/fig6C_terminal_p95_jit.csv"),
        "terminal_p95_jit_ms",
        METHODS,
        METHOD_COLORS,
        "P95 JIT violation (ms)",
        "Terminal tail JIT risk",
        "C",
    )
    dot_interval(
        axs[1, 1],
        read("fig6/fig6D_deadline_reliability.csv"),
        "terminal_deadline_miss_percent",
        METHODS,
        METHOD_COLORS,
        "Deadline miss rate (%)",
        "Deadline reliability",
        "D",
    )
    shared_legend(fig, METHODS, METHOD_COLORS, 1.04, 4)
    save(fig, "Fig6_final")


def fig7():
    fig, axs = plt.subplots(2, 2, figsize=(7.25, 5.15), constrained_layout=True)
    ax = axs[0, 0]
    rows = read("fig7/fig7A_ablation_paired_effect.csv")
    variants = ["no-MAD-pure", "local-only", "no-adaptive-privacy", "no-async"]
    display = {
        "no-MAD-pure": "No MAD",
        "local-only": "Local only",
        "no-adaptive-privacy": "No adaptive privacy",
        "no-async": "No async",
    }
    metrics = [
        ("Reward", "reward_change_percent"),
        ("Mean JIT", "mean_jit_change_percent"),
        ("P95 JIT", "p95_change_percent"),
        ("Deadline miss", "miss_change_percent"),
    ]
    x = np.arange(4)
    offsets = np.linspace(-0.24, 0.24, 4)
    colors = {v: VARIANT_COLORS[display[v]] for v in variants}
    for j, v in enumerate(variants):
        for i, (_, k) in enumerate(metrics):
            points = [r for r in rows if r["variant"] == v]
            ys = [val(r, k) for r in points]
            jitter = np.array([seed_offset(r["seed"], SEEDS_3, 0.035) for r in points])
            ax.scatter(
                np.full(len(points), i) + offsets[j] + jitter,
                ys,
                s=11,
                color=colors[v],
                alpha=0.82,
                zorder=3,
            )
            ax.errorbar(
                i + offsets[j],
                np.mean(ys),
                yerr=np.std(ys, ddof=1),
                fmt="D",
                ms=3.8,
                mfc="white",
                mec=colors[v],
                mew=0.7,
                ecolor=colors[v],
                elinewidth=0.65,
                capsize=1.7,
                zorder=4,
            )
    ax.axhline(0, color="#777777", lw=0.7, ls="--")
    ax.set_xticks(x, [m[0] for m in metrics], rotation=20, ha="right")
    ax.set_ylabel("Matched change (%)")
    label(ax, "A", "Component-ablation effects")
    finish_axes(ax)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=colors[v],
            markeredgecolor=colors[v],
            ms=4,
            label=display[v],
        )
        for v in variants
    ]
    ax.legend(handles=handles, fontsize=6, loc="best")
    specs = [
        (
            "fig7/fig7B_graph_gradient_dynamics.csv",
            "graph_grad_norm",
            "Graph-gradient norm",
            "B",
            "Graph-gradient dynamics",
            {"Full": "#E96B6C", "no-MAD-pure": "#5D6570"},
        ),
        (
            "fig7/fig7C_attention_entropy_dynamics.csv",
            "mean_attention_entropy",
            "Mean attention entropy",
            "C",
            "Attention-entropy dynamics",
            {"Full": "#E96B6C", "no-MAD-pure": "#5D6570"},
        ),
        (
            "fig7/fig7D_advantage_dispersion.csv",
            "mean_advantage_std",
            "Advantage-estimate dispersion",
            "D",
            "Advantage-estimate dispersion",
            {"Full": "#E96B6C", "no-async": "#5D6570"},
        ),
    ]
    for ax, (rel, k, yl, let, t, cs) in zip([axs[0, 1], axs[1, 0], axs[1, 1]], specs):
        rs = read(rel)
        trajectory(ax, rs, k, "variant", list(cs), cs, ylabel=yl)
        label(ax, let, t)
        ax.legend(fontsize=6, loc="best")
    save(fig, "Fig7_final")


def fig8():
    fig, axs = plt.subplots(2, 2, figsize=(7.25, 5.25), constrained_layout=True)
    ax = axs[0, 0]
    rows = read("fig8/fig8A_joint_risk_quality_events.csv")
    hb = ax.hexbin(
        [val(r, "normalized_jit") for r in rows],
        [val(r, "normalized_staleness") for r in rows],
        C=[val(r, "quality_q_u") for r in rows],
        gridsize=30,
        reduce_C_function=np.mean,
        mincnt=1,
        cmap="viridis",
        linewidths=0.08,
    )
    fig.colorbar(hb, ax=ax, pad=0.02, fraction=0.046, label="Update quality $Q_u$")
    ax.set_xlabel("Normalized JIT risk")
    ax.set_ylabel("Normalized staleness")
    label(ax, "A", "Joint JIT–staleness quality response")
    finish_axes(ax)
    ax = axs[0, 1]
    rows = read("fig8/fig8B_q_coefficient_sensitivity.csv")
    pcols = {"alpha": "#646C70", "beta": "#D07A35", "chi": "#4C97C9"}
    for p in ["alpha", "beta", "chi"]:
        sub = [r for r in rows if r["coefficient"] == p]
        base = [r for r in rows if r["setting_id"] == "base"]
        xs = [0.5, 1.0, 1.5]
        means = []
        sds = []
        for x in xs:
            q = [val(r, "terminal_mean_q_u") for r in sub if val(r, "multiplier") == x]
            if x == 1.0:
                q = [val(r, "terminal_mean_q_u") for r in base]
            means.append(np.mean(q))
            sds.append(np.std(q, ddof=1) if len(q) > 1 else 0)
            ax.scatter([x] * len(q), q, s=10, color=pcols[p], alpha=0.65, zorder=3)
        ax.plot(xs, means, color=pcols[p], lw=1.2, marker="o", ms=3.5, label=p)
        ax.errorbar(
            xs, means, yerr=sds, color=pcols[p], fmt="none", elinewidth=0.7, capsize=2
        )
    ax.set_xlabel("Coefficient multiplier")
    ax.set_ylabel("Terminal mean $Q_u$")
    ax.set_xticks([0.5, 1, 1.5])
    label(ax, "B", "Q-weight coefficient sensitivity")
    ax.legend(fontsize=6, title="coefficient", title_fontsize=6)
    finish_axes(ax)
    ax = axs[1, 0]
    rows = read("fig8/fig8C_q_component_summary.csv")
    settings = [
        "alpha_0p50x",
        "base",
        "alpha_1p50x",
        "beta_0p50x",
        "beta_1p50x",
        "chi_0p50x",
        "chi_1p50x",
    ]
    comps = [
        ("weighted_jit", "JIT"),
        ("weighted_staleness", "Staleness"),
        ("weighted_concentration", "Concentration"),
    ]
    x = np.arange(7)
    off = np.linspace(-0.2, 0.2, 3)
    for j, (comp, name) in enumerate(comps):
        for i, s in enumerate(settings):
            points = [
                r for r in rows if r["setting_id"] == s and r["component"] == comp
            ]
            ys = [val(r, "component_mean") for r in points]
            jitter = np.array([seed_offset(r["seed"], SEEDS_3, 0.025) for r in points])
            ax.scatter(
                np.full(len(points), i) + off[j] + jitter,
                ys,
                s=9,
                color=COMP_COLORS[comp],
                alpha=0.75,
                zorder=3,
            )
            ax.errorbar(
                i + off[j],
                np.mean(ys),
                yerr=np.std(ys, ddof=1),
                fmt="D",
                ms=3.5,
                mfc="white",
                mec=COMP_COLORS[comp],
                mew=0.6,
                ecolor=COMP_COLORS[comp],
                elinewidth=0.65,
                capsize=1.8,
                zorder=4,
            )
        ax.plot([], [], color=COMP_COLORS[comp], marker="o", lw=1, label=name)
    ax.set_xticks(
        x,
        ["α0.5", "Base", "α1.5", "β0.5", "β1.5", "χ0.5", "χ1.5"],
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("Weighted denominator component")
    label(ax, "C", "Q-denominator component response")
    ax.legend(fontsize=6)
    finish_axes(ax)
    ax = axs[1, 1]
    ax.axis("off")
    rows = read("fig8/fig8D_privacy_terminal_raw.csv")
    metrics = [
        ("Reward", "terminal_reward"),
        ("Mean JIT", "terminal_mean_jit_ms"),
        ("P95 JIT", "terminal_p95_jit_ms"),
        ("Deadline miss", "terminal_deadline_miss_rate"),
    ]
    budgets = [2.5, 5, 10]
    label(ax, "D", "Privacy-budget robustness")
    for ii, (name, k) in enumerate(metrics):
        iax = ax.inset_axes(
            [0.08 + (ii % 2) * 0.48, 0.10 + (1 - ii // 2) * 0.43, 0.37, 0.30]
        )
        for j, b in enumerate(budgets):
            points = [r for r in rows if val(r, "E_total") == b]
            ys = [val(r, k) for r in points]
            jitter = np.array([seed_offset(r["seed"], SEEDS_3, 0.05) for r in points])
            iax.scatter(
                np.full(len(points), j) + jitter, ys, s=7, color="#4C97C9", alpha=0.75
            )
            iax.errorbar(
                j,
                np.mean(ys),
                yerr=np.std(ys, ddof=1),
                fmt="D",
                ms=3,
                mfc="white",
                mec="#4C97C9",
                ecolor="#4C97C9",
                elinewidth=0.55,
                capsize=1.5,
            )
        iax.set_xticks(range(3), ["2.5", "5", "10"])
        iax.set_title(name, fontsize=6.2, loc="left")
        iax.tick_params(labelsize=5.5, length=1.8)
        iax.grid(axis="y", color="#E7E7E7", lw=0.4)
        iax.spines["top"].set_visible(False)
        iax.spines["right"].set_visible(False)
        iax.set_ylabel(
            {
                "terminal_reward": "reward",
                "terminal_mean_jit_ms": "ms",
                "terminal_p95_jit_ms": "ms",
                "terminal_deadline_miss_rate": "fraction",
            }[k],
            fontsize=5.5,
        )
    save(fig, "Fig8_final")


def fig9():
    fig, axs = plt.subplots(2, 2, figsize=(7.25, 5.15), constrained_layout=True)
    specs = [
        (
            "fig9/fig9A_reward_scale_change.csv",
            "reward_delta",
            "Reward change",
            "A",
            "Reward scale retention",
        ),
        (
            "fig9/fig9B_mean_jit_scale_change.csv",
            "mean_jit_delta",
            "Mean JIT change (ms)",
            "B",
            "Mean-JIT scale retention",
        ),
        (
            "fig9/fig9C_p95_jit_scale_change.csv",
            "p95_delta",
            "P95 JIT change (ms)",
            "C",
            "Tail-JIT scale retention",
        ),
        (
            "fig9/fig9D_deadline_scale_change.csv",
            "miss_delta_percentage_points",
            "Deadline miss change (percentage points)",
            "D",
            "Deadline-reliability retention",
        ),
    ]
    shapes = {30: "o", 80: "^"}
    for ax, (rel, k, yl, let, t) in zip(axs.flat, specs):
        rows = read(rel)
        x = np.arange(4)
        off = {30: -0.12, 80: 0.12}
        for m in METHODS:
            for n in [30, 80]:
                points = [
                    r for r in rows if r["method"] == m and int(r["target_nodes"]) == n
                ]
                ys = [val(r, k) for r in points]
                jitter = np.array(
                    [seed_offset(r["seed"], SEEDS_3, 0.025) for r in points]
                )
                xpos = np.full(len(points), x[METHODS.index(m)] + off[n])
                ax.scatter(
                    xpos + jitter,
                    ys,
                    s=10,
                    color=METHOD_COLORS[m],
                    alpha=0.78,
                    marker=shapes[n],
                    zorder=3,
                )
                ax.errorbar(
                    x[METHODS.index(m)] + off[n],
                    np.mean(ys),
                    yerr=np.std(ys, ddof=1),
                    fmt=shapes[n],
                    ms=4,
                    mfc="white",
                    mec=METHOD_COLORS[m],
                    ecolor=METHOD_COLORS[m],
                    elinewidth=0.7,
                    capsize=1.8,
                )
        ax.axhline(0, color="#777777", lw=0.7, ls="--")
        ax.set_xticks(x, METHODS, rotation=22, ha="right")
        ax.set_ylabel(yl)
        label(ax, let, t)
        finish_axes(ax)
    mh = [Line2D([0], [0], color=METHOD_COLORS[m], lw=1.6, label=m) for m in METHODS]
    sh = [
        Line2D([0], [0], color="#555", marker=shapes[n], lw=0, ms=4, label=f"{n} nodes")
        for n in [30, 80]
    ]
    fig.legend(
        handles=mh + sh,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.055),
        ncol=6,
        fontsize=6.4,
        handlelength=1.9,
        columnspacing=1.0,
    )
    save(fig, "Fig9_final")


if __name__ == "__main__":
    fig6()
    fig7()
    fig8()
    fig9()
