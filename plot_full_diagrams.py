#!/usr/bin/env python3
"""
Plot the final full maps for one geometry.

Expected input files in --data-dir (default: data):

    Pe.txt
    Da.txt
    Sh_<geometry>_pois.txt
    chi_<geometry>_pois.txt
    Le_<geometry>_pois.txt

where <geometry> is one of:

    tube
    plates
    oneplate

Example:

    python plot_full_diagrams.py tube
    python plot_full_diagrams.py plates --data-dir data --output-dir figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


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


# =========================================================
# Geometry configuration
# =========================================================
GEOMETRIES = {
    "tube": {
        "label": "Circular tube, Poiseuille flow",
        "cmap": "Reds",
    },
    "plates": {
        "label": "Parallel plates, Poiseuille flow",
        "cmap": "Blues",
    },
    "oneplate": {
        "label": "Parallel plates, one reactive surface",
        "cmap": "Greens",
    },
}


# =========================================================
# CLI
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Sh(Pe,Da), chi(Pe,Da), and Le(Pe,Da) maps."
    )

    parser.add_argument(
        "geometry",
        choices=["tube", "plates", "oneplate"],
        help="Geometry: tube, plates, or oneplate.",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing input txt files. Default: data.",
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
        default=None,
        help="Override output basename. Default: sh_pois_<geometry>.",
    )

    parser.add_argument(
        "--no-contours",
        action="store_true",
        help="Disable contours.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively after saving if backend allows it.",
    )

    return parser.parse_args()


# =========================================================
# Loading helpers
# =========================================================
def load_vector(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    x = np.loadtxt(path, dtype=float)
    return np.asarray(x, dtype=float).ravel()


def load_table(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return np.asarray(np.loadtxt(path, dtype=float), dtype=float)


def load_named_table(data_dir: Path, stem: str, geometry: str, flow: str = "pois") -> np.ndarray:
    """
    Load e.g. Sh_plates_pois.txt.

    For chi we also allow Chi_... and CHI_... as fallbacks.
    For Le we also allow le_... and LE_... as fallbacks.
    """
    candidates = [data_dir / f"{stem}_{geometry}_{flow}.txt"]

    if stem == "chi":
        candidates += [
            data_dir / f"Chi_{geometry}_{flow}.txt",
            data_dir / f"CHI_{geometry}_{flow}.txt",
        ]
    elif stem == "Le":
        candidates += [
            data_dir / f"le_{geometry}_{flow}.txt",
            data_dir / f"LE_{geometry}_{flow}.txt",
        ]

    checked = []
    for path in candidates:
        checked.append(str(path))
        if path.exists():
            return load_table(path)

    raise FileNotFoundError("None of these files were found:\n  " + "\n  ".join(checked))


def orient_table(Z: np.ndarray, Da: np.ndarray, Pe: np.ndarray, name: str) -> np.ndarray:
    Z = np.asarray(Z, dtype=float)

    if Z.ndim != 2:
        raise ValueError(f"{name} must be a 2D table, got shape {Z.shape}.")

    expected = (len(Da), len(Pe))
    transposed = (len(Pe), len(Da))

    if Z.shape == expected:
        return Z
    if Z.shape == transposed:
        return Z.T

    raise ValueError(
        f"Unexpected shape for {name}: {Z.shape}. Expected {expected} "
        f"or {transposed}."
    )


def load_full_dataset(data_dir: Path, geometry: str):
    Pe = load_vector(data_dir / "Pe.txt")
    Da = load_vector(data_dir / "Da.txt")

    Sh = load_named_table(data_dir, "Sh", geometry, "pois")
    chi = load_named_table(data_dir, "chi", geometry, "pois")
    Le = load_named_table(data_dir, "Le", geometry, "pois")

    Sh = orient_table(Sh, Da, Pe, "Sh")
    chi = orient_table(chi, Da, Pe, "chi")
    Le = orient_table(Le, Da, Pe, "Le")

    i_da = np.argsort(Da)
    i_pe = np.argsort(Pe)

    Da = Da[i_da]
    Pe = Pe[i_pe]
    Sh = Sh[np.ix_(i_da, i_pe)]
    chi = chi[np.ix_(i_da, i_pe)]
    Le = Le[np.ix_(i_da, i_pe)]

    return Pe, Da, Sh, chi, Le


# =========================================================
# Plot helpers
# =========================================================
def log_edges(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        raise ValueError("Need at least two points to construct log edges.")
    if np.any(x <= 0):
        raise ValueError("Log-spaced axes require positive values.")

    lx = np.log10(x)
    edges = np.empty(len(x) + 1)
    edges[1:-1] = 10.0 ** (0.5 * (lx[:-1] + lx[1:]))
    edges[0] = 10.0 ** (lx[0] - 0.5 * (lx[1] - lx[0]))
    edges[-1] = 10.0 ** (lx[-1] + 0.5 * (lx[-1] - lx[-2]))
    return edges


def auto_contour_levels(Z, n=8, mode="quantile", qmin=0.05, qmax=0.95):
    vals = np.asarray(Z, dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return None

    if mode == "log":
        vals = vals[vals > 0]
        if vals.size == 0:
            return None
        lo, hi = np.nanquantile(vals, [qmin, qmax])
        if lo <= 0 or hi <= 0 or lo == hi:
            return None
        levels = np.geomspace(lo, hi, n)
    elif mode == "linear":
        lo, hi = np.nanquantile(vals, [qmin, qmax])
        if lo == hi:
            return None
        levels = np.linspace(lo, hi, n)
    else:
        qs = np.linspace(qmin, qmax, n)
        levels = np.nanquantile(vals, qs)

    levels = np.unique(np.round(levels, 8))
    if len(levels) < 2:
        return None
    return levels


def plot_map_on_ax(
    fig,
    ax,
    Pe_tab,
    Da_tab,
    Z,
    title,
    cbar_label,
    cmap="Blues",
    log_color=False,
    symmetric=False,
    title_fontsize=34,
    tick_fontsize=24,
    label_fontsize=28,
    cbar_ticksize=20,
    cbar_labelsize=24,
    cbar_size="4.5%",
    cbar_pad=0.12,
    show_ylabel=True,
    show_xlabel=True,
    box_aspect=1,
    contours=True,
    contour_levels=None,
    contour_color="0.25",
    contour_linewidths=0.45,
    contour_alpha=0.55,
    contour_labels=False,
    contour_label_fontsize=13,
    contour_label_fmt="%.2g",
):
    Pe_edges = log_edges(Pe_tab)
    Da_edges = log_edges(Da_tab)

    if symmetric:
        vmax = np.nanmax(np.abs(Z))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        Zplot = np.array(Z, dtype=float)
        pcm = ax.pcolormesh(
            Pe_edges, Da_edges, Zplot,
            shading="flat", cmap=cmap, norm=norm,
        )
    elif log_color:
        Zplot = np.array(Z, dtype=float)
        Zplot[~np.isfinite(Zplot)] = np.nan
        Zplot = np.where(Zplot > 0, Zplot, np.nan)
        pcm = ax.pcolormesh(
            Pe_edges, Da_edges, Zplot,
            shading="flat", cmap=cmap,
            norm=LogNorm(vmin=np.nanmin(Zplot), vmax=np.nanmax(Zplot)),
        )
    else:
        Zplot = np.array(Z, dtype=float)
        pcm = ax.pcolormesh(
            Pe_edges, Da_edges, Zplot,
            shading="flat", cmap=cmap,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(Pe_tab[0], Pe_tab[-1])
    ax.set_ylim(Da_tab[0], Da_tab[-1])

    if contours:
        Pe_grid, Da_grid = np.meshgrid(Pe_tab, Da_tab)
        Zc = np.asarray(Z, dtype=float)
        Zc = np.ma.masked_invalid(Zc)

        if contour_levels is None:
            if log_color:
                contour_levels = auto_contour_levels(Z, n=7, mode="log")
            else:
                contour_levels = auto_contour_levels(Z, n=8, mode="quantile")

        if contour_levels is not None:
            cs = ax.contour(
                Pe_grid, Da_grid, Zc,
                levels=contour_levels,
                colors=contour_color,
                linewidths=contour_linewidths,
                alpha=contour_alpha,
            )
            if contour_labels:
                ax.clabel(
                    cs,
                    inline=True,
                    fontsize=contour_label_fontsize,
                    fmt=contour_label_fmt,
                )

    ax.set_xlabel("Pe" if show_xlabel else "", fontsize=label_fontsize)
    ax.set_ylabel("Da" if show_ylabel else "", fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize, pad=12)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.set_box_aspect(box_aspect)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=cbar_size, pad=cbar_pad)
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.set_label(cbar_label, fontsize=cbar_labelsize, labelpad=10)
    cbar.ax.tick_params(labelsize=cbar_ticksize)

    return pcm, cbar


def plot_full_maps(Pe_tab, Da_tab, Sh_tab, chi_tab, Le_tab, output_base: Path, suptitle: str, cmap: str, contours=True):
    fig = plt.figure(figsize=(23, 31))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[2.05, 1.0],
        hspace=0.24,
        wspace=0.26,
    )

    ax_top = fig.add_subplot(gs[0, :])
    ax_bl = fig.add_subplot(gs[1, 0], sharex=ax_top, sharey=ax_top)
    ax_br = fig.add_subplot(gs[1, 1], sharex=ax_top, sharey=ax_top)

    plot_map_on_ax(
        fig, ax_top, Pe_tab, Da_tab, Sh_tab,
        title=r"a) Sherwood number $\mathrm{Sh}$",
        cbar_label=r"$\mathrm{Sh}$",
        cmap=cmap,
        log_color=False,
        title_fontsize=38,
        tick_fontsize=30,
        label_fontsize=38,
        cbar_ticksize=30,
        cbar_labelsize=38,
        cbar_size="3.8%",
        cbar_pad=0.14,
        show_ylabel=True,
        show_xlabel=True,
        box_aspect=1,
        contours=contours,
    )

    plot_map_on_ax(
        fig, ax_bl, Pe_tab, Da_tab, chi_tab,
        title=r"b) Averaging factor $\chi$",
        cbar_label=r"$\chi$",
        cmap=cmap,
        log_color=False,
        title_fontsize=38,
        tick_fontsize=26,
        label_fontsize=30,
        cbar_ticksize=26,
        cbar_labelsize=30,
        cbar_size="4.3%",
        cbar_pad=0.12,
        show_ylabel=True,
        show_xlabel=True,
        box_aspect=1,
        contours=contours,
    )

    plot_map_on_ax(
        fig, ax_br, Pe_tab, Da_tab, Le_tab,
        title=r"c) Entrance length $L_{\mathrm{e}}$",
        cbar_label=r"$L_{\mathrm{e}}$",
        cmap=cmap,
        log_color=True,
        title_fontsize=38,
        tick_fontsize=26,
        label_fontsize=30,
        cbar_ticksize=26,
        cbar_labelsize=30,
        cbar_size="4.3%",
        cbar_pad=0.12,
        show_ylabel=False,
        show_xlabel=True,
        box_aspect=1,
        contours=contours,
    )

    fig.suptitle(suptitle, fontsize=46, y=0.975)
    fig.subplots_adjust(left=0.07, right=0.965, bottom=0.06, top=0.93)

    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = GEOMETRIES[args.geometry]
    Pe, Da, Sh, chi, Le = load_full_dataset(args.data_dir, args.geometry)

    output_base = args.output_dir / (args.output_base or f"sh_pois_{args.geometry}")
    plot_full_maps(
        Pe,
        Da,
        Sh,
        chi,
        Le,
        output_base=output_base,
        suptitle=config["label"],
        cmap=config["cmap"],
        contours=not args.no_contours,
    )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
