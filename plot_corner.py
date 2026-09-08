#!/usr/bin/env python3
"""
Plot Sh_Dh(Da) curves in the small-Da "weak-exchange corner" for the
circular tube, together with the Pe->0 / Da->0 non-commuting-limit
asymptotics, plus two collapse diagnostics that isolate each asymptotic
term separately.

Expected input files in --data-dir (default: data), as saved by
tube_poiseuille_parallel_fast.py's "tube_corner" sweep:

    Da_tube_corner.txt
    Pe_tube_corner.txt
    Sh_tube_corner.txt      # shape (len(Da), len(Pe))
    fail_tube_corner.txt    # optional, same shape, 1 = solver failure

Limits and asymptotics:

    Sh -> 48/11        as Da -> 0 at any fixed Pe > 0
    Sh -> 6            as Pe -> 0 taken first, then Da -> 0
    Composite:  Sh ~ 24(1+s)/(7+4s) - (9/16) Da,  s = sqrt(1 + 8 Da/Pe^2)
    Overlap (Pe^2 << Da << 1):  Sh ~ 6 - (9/(4 sqrt2)) Pe/sqrt(Da) - (9/16) Da
    Local max:  Da_max ~ 2^(1/3) Pe^(2/3),  Sh_max ~ 6 - (27/16) 2^(1/3) Pe^(2/3)

Four figures are produced:

    <output-base>.png             Sh_Dh(Da) per Pe, with the composite curve
                                   and the predicted maxima overlaid on the
                                   raw (Pe, Da) axes.

    <output-base>_transition.png  Sh + (9/16) Da  vs.  Da/Pe^2.
                                   Removing the (9/16) Da correction kills
                                   the local maximum and collapses all Pe
                                   onto the single-variable transition curve
                                   24(1+s)/(7+4s), s = sqrt(1+8 eta) -- i.e. it
                                   isolates the 48/11 -> 6 crossover itself.

    <output-base>_maximum.png     (6 - Sh)/Pe^(2/3)  vs.  xi = Da/Pe^(2/3).
                                   As Pe -> 0 these curves should converge
                                   onto 9/(4 sqrt(2 xi)) + (9/16) xi, whose
                                   minimum at xi = 2^(1/3) (value ~2.126)
                                   pins down both the location and the
                                   height of the Sh maximum at once.

    <output-base>_composite.png   All three panels combined: the profile
                                   plot spans the top row, with the maximum-
                                   and transition-collapse panels side by
                                   side underneath.

Example:

    python plot_corner.py
    python plot_corner.py --data-dir data --output-dir figures
"""

from __future__ import annotations

import argparse
from pathlib import Path
from matplotlib.lines import Line2D

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# =========================================================
# Style
# =========================================================
font = {
    "family": "Times New Roman",
    "weight": "normal",
    "size": 24,
}

matplotlib.rc("font", **font)
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["mathtext.fontset"] = "stix"

# Colorblind-friendly Okabe-Ito palette. Distinct markers provide a
# color-independent identification of the Pe curves as well.
CURVE_COLORS = ["#0072B2", "#E69F00", "#CC79A7", "#009E73", "#56B4E9"]
CURVE_MARKERS = ["o", "s", "^", "D", "v"]

# Sub-panel text sizes for the composite figure, matching the grid-panel
# convention used elsewhere in this repo (see plot_full_diagrams.py's
# per-panel title/label/tick sizes and tube_solver.py's legend.fontsize).
PANEL_TITLE_FONTSIZE = 34
PANEL_LABEL_FONTSIZE = 30
PANEL_TICK_LABELSIZE = 26
PANEL_LEGEND_FONTSIZE = 18

SH_DA0_PE0 = 6.0
SH_DA0_PEPOS = 48.0 / 11.0
XI_MAX_STAR = 2.0 ** (1.0 / 3.0)


def _pe_label(Pe: float) -> str:
    """Compact mathematical label for a Péclet number."""
    if Pe > 0:
        exponent = np.log10(Pe)
        rounded = int(round(exponent))
        if np.isclose(exponent, rounded, atol=1e-12):
            if rounded == 0:
                return r"$\mathrm{Pe}=1$"
            return fr"$\mathrm{{Pe}}=10^{{{rounded}}}$"
    return fr"$\mathrm{{Pe}}={Pe:g}$"


def _pe_handles(Pe_tab):
    """Legend handles encoding only the Pe identity (color + marker)."""
    handles = []
    for j, Pe in enumerate(Pe_tab):
        handles.append(
            Line2D(
                [0], [0],
                color=CURVE_COLORS[j % len(CURVE_COLORS)],
                lw=1.8,
                marker=CURVE_MARKERS[j % len(CURVE_MARKERS)],
                markersize=6,
                markerfacecolor="none",
                markeredgewidth=1.2,
                label=_pe_label(Pe),
            )
        )
    return handles


def _direct_limit_label(ax, y, text, *, color="black", x=0.97, va="bottom"):
    """Place a compact label directly on a horizontal reference level."""
    fontsize = 0.82 * matplotlib.rcParams["font.size"]
    ax.text(
        x,
        y,
        text,
        transform=ax.get_yaxis_transform(),
        ha="right",
        va=va,
        color=color,
        fontsize=fontsize,
        #bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
        zorder=6,
    )


# =========================================================
# Asymptotic formulas
# =========================================================
def sh_composite(Da: np.ndarray, Pe: float) -> np.ndarray:
    """Composite approximation connecting the two weak-exchange limits."""
    s = np.sqrt(1.0 + 8.0 * Da / Pe**2)
    return 24.0 * (1.0 + s) / (7.0 + 4.0 * s) - (9.0 / 16.0) * Da


def sh_composite_leading(eta: np.ndarray) -> np.ndarray:
    """Leading distinguished-limit crossover as a function of eta = Da/Pe^2."""
    s = np.sqrt(1.0 + 8.0 * eta)
    return 24.0 * (1.0 + s) / (7.0 + 4.0 * s)


def maximum_envelope(xi: np.ndarray) -> np.ndarray:
    """Overlap prediction for (6-Sh)/Pe^(2/3) vs. xi = Da/Pe^(2/3)."""
    return 9.0 / (4.0 * np.sqrt(2.0 * xi)) + (9.0 / 16.0) * xi


def corner_maximum(Pe: float) -> tuple[float, float]:
    """Predicted (Da_max, Sh_max) of the local maximum from the overlap expansion."""
    Da_max = XI_MAX_STAR * Pe ** (2.0 / 3.0)
    Sh_max = SH_DA0_PE0 - (27.0 / 16.0) * XI_MAX_STAR * Pe ** (2.0 / 3.0)
    return Da_max, Sh_max


# =========================================================
# CLI
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Sh_Dh(Da) tube-corner curves with weak-exchange asymptotics."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing the tube_corner input files. Default: data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for output figures. Default: current directory.",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default="tube_corner_asymptotics",
        help="Output basename. Default: tube_corner_asymptotics.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="tube_corner",
        help="Input file prefix, i.e. Sh_<prefix>.txt etc. Default: tube_corner.",
    )
    parser.add_argument(
        "--skip-profile",
        action="store_true",
        help="Skip the Sh_Dh(Da) profile figure.",
    )
    parser.add_argument(
        "--skip-transition",
        action="store_true",
        help="Skip the Sh + (9/16)Da vs Da/Pe^2 transition-collapse figure.",
    )
    parser.add_argument(
        "--skip-maximum",
        action="store_true",
        help="Skip the (6-Sh)/Pe^(2/3) vs Da/Pe^(2/3) maximum-collapse figure.",
    )
    parser.add_argument(
        "--skip-composite",
        action="store_true",
        help="Skip the combined profile+maximum+transition figure.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figures interactively after saving if the backend allows it.",
    )

    return parser.parse_args()


# =========================================================
# Loading helpers
# =========================================================
def load_vector(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return np.asarray(np.loadtxt(path, dtype=float), dtype=float).ravel()


def load_table(path: Path, n_da: int, n_pe: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    Z = np.asarray(np.loadtxt(path, dtype=float), dtype=float)
    if Z.shape == (n_da, n_pe):
        return Z
    if Z.shape == (n_pe, n_da):
        return Z.T
    raise ValueError(f"Unexpected shape for {path}: {Z.shape}, expected ({n_da}, {n_pe}).")


def load_corner_data(data_dir: Path, prefix: str):
    Da_raw = load_vector(data_dir / f"Da_{prefix}.txt")
    Pe_raw = load_vector(data_dir / f"Pe_{prefix}.txt")

    i_da = np.argsort(Da_raw)
    Da = Da_raw[i_da]
    i_pe = np.argsort(Pe_raw)
    Pe_tab = Pe_raw[i_pe]

    Sh = load_table(data_dir / f"Sh_{prefix}.txt", len(Da_raw), len(Pe_raw))
    Sh = Sh[np.ix_(i_da, i_pe)]

    fail_path = data_dir / f"fail_{prefix}.txt"
    if fail_path.exists():
        fail = load_table(fail_path, len(Da_raw), len(Pe_raw))
        fail = fail[np.ix_(i_da, i_pe)]
        Sh = np.where(fail > 0, np.nan, Sh)

    return Da, Pe_tab, Sh


def _savefig(fig, output_base: Path):
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


# =========================================================
# Figure 1: Sh_Dh(Da) profile, raw axes
# =========================================================
PROFILE_TITLE = r"Near-corner Sherwood-number profiles"
TRANSITION_TITLE = r"Distinguished-limit crossover"
MAXIMUM_TITLE = r"Scaling near the local maximum"


def draw_profile(
    ax,
    Da,
    Pe_tab,
    Sh,
    legend_fontsize=11,
    title_fontsize=None,
    label_fontsize=None,
    tick_labelsize=None,
    title=PROFILE_TITLE,
):
    Da_dense = np.logspace(np.log10(Da.min()), np.log10(Da.max()), 400)

    for j, Pe in enumerate(Pe_tab):
        color = CURVE_COLORS[j % len(CURVE_COLORS)]
        marker = CURVE_MARKERS[j % len(CURVE_MARKERS)]

        ax.semilogx(
            Da,
            Sh[:, j],
            color=color,
            lw=0,
            marker=marker,
            markersize=6,
            markerfacecolor="none",
            markeredgewidth=1.3,
        )

        ax.semilogx(
            Da_dense,
            sh_composite(Da_dense, Pe),
            color=color,
            lw=2.8,
            ls="--",
        )

        Da_max, Sh_max = corner_maximum(Pe)
        if Da.min() <= Da_max <= Da.max():
            ax.plot(
                Da_max,
                Sh_max,
                marker="*",
                markersize=15,
                color=color,
                markeredgecolor="black",
                markeredgewidth=0.6,
                zorder=5,
            )

    # Reference levels are annotated directly instead of occupying legend rows.
    ax.axhline(SH_DA0_PE0, color="black", lw=1.1, ls="-.", zorder=1)
    ax.axhline(SH_DA0_PEPOS, color="0.45", lw=1.1, ls="-.", zorder=1)

    ax.set_ylim(4.25,6.15)
    ax.set_xlabel(r"$\mathrm{Da}$", fontsize=label_fontsize)
    ax.set_ylabel(r"$\mathrm{Sh}$", fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize)
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.10)
    if tick_labelsize is not None:
        ax.tick_params(labelsize=tick_labelsize)

    _direct_limit_label(ax, SH_DA0_PE0, r"$\mathrm{Sh}=6$", color="black")
    _direct_limit_label(ax, SH_DA0_PEPOS, r"$\mathrm{Sh}=48/11$", color="black")

    handles = _pe_handles(Pe_tab)
    handles.extend(
        [
            Line2D(
                [0], [0], marker="o", color="black", linestyle="None",
                markerfacecolor="none", markersize=6, label="numerical",
            ),
            Line2D(
                [0], [0], color="black", linestyle="--", lw=2.8,
                label="composite",
            ),
            Line2D(
                [0], [0], marker="*", color="black", linestyle="None",
                markersize=15, label="asymptotic maximum",
            ),
        ]
    )
    ax.legend(
        handles=handles,
        frameon=True,
        fontsize=legend_fontsize,
        ncol=2,
        loc="best",
        columnspacing=1.1,
        handletextpad=0.6,
    )


def plot_profile(Da, Pe_tab, Sh, output_base: Path):
    fig, ax = plt.subplots(figsize=(11.5, 8.0))
    draw_profile(ax, Da, Pe_tab, Sh)
    _savefig(fig, output_base)


# =========================================================
# Figure 2: transition collapse, Sh + (9/16)Da vs Da/Pe^2
# =========================================================
def draw_transition(
    ax,
    Da,
    Pe_tab,
    Sh,
    legend_fontsize=11,
    title_fontsize=None,
    label_fontsize=None,
    tick_labelsize=None,
    title=TRANSITION_TITLE,
):
    eta_all_min, eta_all_max = np.inf, -np.inf

    for j, Pe in enumerate(Pe_tab):
        color = CURVE_COLORS[j % len(CURVE_COLORS)]
        marker = CURVE_MARKERS[j % len(CURVE_MARKERS)]
        eta = Da / Pe**2
        y = Sh[:, j] + (9.0 / 16.0) * Da

        eta_all_min = min(eta_all_min, np.nanmin(eta))
        eta_all_max = max(eta_all_max, np.nanmax(eta))

        ax.semilogx(
            eta,
            y,
            color=color,
            lw=0,
            marker=marker,
            markersize=6,
            markerfacecolor="none",
            markeredgewidth=1.3,
            markevery=2
        )

    eta_dense = np.logspace(np.log10(eta_all_min), np.log10(eta_all_max), 400)
    ax.semilogx(
        eta_dense,
        sh_composite_leading(eta_dense),
        color="black",
        lw=2.8,
        ls="--",
        zorder=4,
    )

    ax.axhline(SH_DA0_PEPOS, color="0.45", lw=0.9, ls=":", zorder=1)
    ax.axhline(SH_DA0_PE0, color="black", lw=0.9, ls=":", zorder=1)

    ax.set_xlim(10**(-7), 10**9)
    ax.set_xlabel(r"$\eta = \mathrm{Da}/\mathrm{Pe}^2$", fontsize=label_fontsize)
    ax.set_ylabel(r"$\mathrm{Sh} + \frac{9}{16}\mathrm{Da}$", fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize)
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.10)
    if tick_labelsize is not None:
        ax.tick_params(labelsize=tick_labelsize)

    _direct_limit_label(ax, SH_DA0_PEPOS, r"$\mathrm{Sh}=48/11$", color="black")
    _direct_limit_label(ax, SH_DA0_PE0, r"$\mathrm{Sh}=6$", color="black")

    handles = _pe_handles(Pe_tab)
    handles.append(
        Line2D(
            [0], [0], color="black", linestyle="--", lw=2.8,
            label=r"$F(\eta)$",
        )
    )
    ax.legend(
        handles=handles,
        frameon=True,
        fontsize=legend_fontsize,
        ncol=1,
        loc="best",
    )


def plot_transition(Da, Pe_tab, Sh, output_base: Path):
    fig, ax = plt.subplots(figsize=(11.5, 8.0))
    draw_transition(ax, Da, Pe_tab, Sh)
    _savefig(fig, output_base)


# =========================================================
# Figure 3: maximum collapse, (6-Sh)/Pe^(2/3) vs Da/Pe^(2/3)
# =========================================================
def draw_maximum(
    ax,
    Da,
    Pe_tab,
    Sh,
    legend_fontsize=11,
    title_fontsize=None,
    label_fontsize=None,
    tick_labelsize=None,
    title=MAXIMUM_TITLE,
):
    xi_all_min, xi_all_max = np.inf, -np.inf

    for j, Pe in enumerate(Pe_tab):
        color = CURVE_COLORS[j % len(CURVE_COLORS)]
        marker = CURVE_MARKERS[j % len(CURVE_MARKERS)]
        xi = Da / Pe ** (2.0 / 3.0)
        y = (SH_DA0_PE0 - Sh[:, j]) / Pe ** (2.0 / 3.0)

        xi_all_min = min(xi_all_min, np.nanmin(xi))
        xi_all_max = max(xi_all_max, np.nanmax(xi))

        ax.loglog(
            xi,
            y,
            color=color,
            lw=0,
            marker=marker,
            markersize=6,
            markerfacecolor="none",
            markeredgewidth=1.3,
            markevery=2
        )

    xi_dense = np.logspace(np.log10(xi_all_min), np.log10(xi_all_max), 400)
    ax.loglog(
        xi_dense,
        maximum_envelope(xi_dense),
        color="black",
        lw=2.8,
        ls="--",
        zorder=4,
    )

    g_min = maximum_envelope(XI_MAX_STAR)
    ax.plot(
        XI_MAX_STAR,
        g_min,
        marker="*",
        markersize=15,
        color="black",
        zorder=5,
    )
    # ax.axvline(XI_MAX_STAR, color="black", lw=0.8, ls=":", alpha=0.6)
    # ax.axhline(g_min, color="black", lw=0.8, ls=":", alpha=0.6)
    # annotation = (
    #     fr"$\xi_*=2^{{1/3}}\approx{XI_MAX_STAR:.3g}$"
    #     + "\n"
    #     + fr"$g_*=\frac{{27}}{{16}}2^{{1/3}}\approx{g_min:.3g}$"
    # )
    # ax.annotate(
    #     annotation,
    #     xy=(XI_MAX_STAR, g_min),
    #     xytext=(0.53, 0.23),
    #     textcoords="axes fraction",
    #     fontsize=0.72 * (label_fontsize or matplotlib.rcParams["font.size"]),
    #     arrowprops=dict(arrowstyle="-", lw=0.8, color="black"),
    #     bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
    # )

    ax.set_xlabel(r"$\xi = \mathrm{Da}/\mathrm{Pe}^{2/3}$", fontsize=label_fontsize)
    #ax.set_ylabel(r"$g_{\mathrm{Pe}}(\xi)=(6 - \mathrm{Sh})/\mathrm{Pe}^{2/3}$", fontsize=label_fontsize)
    ax.set_ylabel(r"$(6 - \mathrm{Sh})/\mathrm{Pe}^{2/3}$", fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize)
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.10)
    if tick_labelsize is not None:
        ax.tick_params(labelsize=tick_labelsize)

    handles = _pe_handles(Pe_tab)
    handles.extend(
        [
            Line2D(
                [0], [0], color="black", linestyle="--", lw=2.8,
                label=r"$g(\xi)$",
            ),
            Line2D(
                [0], [0], marker="*", color="black", linestyle="None",
                markersize=15, label="asymptotic minimum",
            ),
        ]
    )
    ax.legend(
        handles=handles,
        frameon=True,
        fontsize=legend_fontsize,
        ncol=1,
        loc="best",
    )


def plot_maximum(Da, Pe_tab, Sh, output_base: Path):
    fig, ax = plt.subplots(figsize=(11.5, 8.0))
    draw_maximum(ax, Da, Pe_tab, Sh)
    _savefig(fig, output_base)


# =========================================================
# Figure 4: composite -- profile on top, maximum/transition below
# =========================================================
def plot_composite(Da, Pe_tab, Sh, output_base: Path):
    fig = plt.figure(figsize=(20.0, 17.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.4, wspace=0.32)

    ax_top = fig.add_subplot(gs[0, :])
    ax_bl = fig.add_subplot(gs[1, 0])
    ax_br = fig.add_subplot(gs[1, 1])

    panel_kwargs = dict(
        title_fontsize=PANEL_TITLE_FONTSIZE,
        label_fontsize=PANEL_LABEL_FONTSIZE,
        tick_labelsize=PANEL_TICK_LABELSIZE,
        legend_fontsize=PANEL_LEGEND_FONTSIZE,
    )
    draw_profile(ax_top, Da, Pe_tab, Sh, title="(a) Near-corner Sherwood-number profiles", **panel_kwargs)
    draw_maximum(ax_bl, Da, Pe_tab, Sh, title="(b) Scaling near the local maximum", **panel_kwargs)
    draw_transition(ax_br, Da, Pe_tab, Sh, title="(c) Distinguished-limit crossover", **panel_kwargs)

    _savefig(fig, output_base)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_base = args.output_dir / args.output_base

    Da, Pe_tab, Sh = load_corner_data(args.data_dir, args.prefix)

    if not args.skip_profile:
        plot_profile(
            Da, Pe_tab, Sh,
            output_base=output_base,
        )

    if not args.skip_transition:
        plot_transition(
            Da, Pe_tab, Sh,
            output_base=Path(f"{output_base}_transition"),
        )

    if not args.skip_maximum:
        plot_maximum(
            Da, Pe_tab, Sh,
            output_base=Path(f"{output_base}_maximum"),
        )

    if not args.skip_composite:
        plot_composite(
            Da, Pe_tab, Sh,
            output_base=Path(f"{output_base}_composite"),
        )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
