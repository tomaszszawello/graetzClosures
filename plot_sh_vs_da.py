#!/usr/bin/env python3
"""
Plot Sherwood number Sh(Da) curves from final data files.

Expected input files in --data-dir (default: data):

    Da.txt
    Pe.txt                         # needed for Poiseuille tables
    Sh_tube_plug.txt
    Sh_tube_pois.txt
    Sh_plates_plug.txt
    Sh_plates_pois.txt

The plug files can be either vectors of length len(Da), or 2D tables with one
axis equal to len(Da). The Poiseuille files are normally 2D tables with rows in
Da and columns in Pe. The script plots the plug curve, plus the lowest- and
highest-Pe Poiseuille curves.

Example:

    python plot_sh_vs_da.py
    python plot_sh_vs_da.py --data-dir data --output-dir figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

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


# =========================================================
# Geometry configuration
# =========================================================
GEOMETRIES = {
    "plates": {
        "label": "Parallel plates",
        "color": "tab:blue",
    },
    "tube": {
        "label": "Circular tube",
        "color": "tab:red",
    },
}


# =========================================================
# CLI
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Sh(Da) comparison curves."
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
        default="sh_vs_da",
        help="Output basename. Default: sh_vs_da.",
    )

    parser.add_argument(
        "--geometries",
        nargs="+",
        choices=["plates", "tube"],
        default=["plates", "tube"],
        help="Geometries to include. Default: all.",
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


def load_optional_table(path: Path):
    if not path.exists():
        return None
    return np.asarray(np.loadtxt(path, dtype=float), dtype=float)


def orient_table(Z: np.ndarray, Da: np.ndarray, Pe: np.ndarray | None, name: str) -> np.ndarray:
    Z = np.asarray(Z, dtype=float)

    if Z.ndim != 2:
        raise ValueError(f"{name} must be a 2D table, got shape {Z.shape}.")

    if Pe is not None:
        expected = (len(Da), len(Pe))
        transposed = (len(Pe), len(Da))

        if Z.shape == expected:
            return Z
        if Z.shape == transposed:
            return Z.T

    # Useful fallback if a plug file was saved as len(Da) x n or n x len(Da).
    if Z.shape[0] == len(Da):
        return Z
    if Z.shape[1] == len(Da):
        return Z.T

    raise ValueError(
        f"Unexpected shape for {name}: {Z.shape}. Could not match Da length {len(Da)}"
        + (f" and Pe length {len(Pe)}." if Pe is not None else ".")
    )


def extract_curve_from_vector_or_table(Z: np.ndarray, Da: np.ndarray, Pe: np.ndarray | None, col: int = 0) -> np.ndarray:
    Z = np.asarray(Z, dtype=float)

    if Z.ndim == 1:
        if len(Z) != len(Da):
            raise ValueError(f"Vector length {len(Z)} does not match len(Da)={len(Da)}.")
        return Z

    T = orient_table(Z, Da, Pe, "Sh")
    if T.shape[1] == 0:
        raise ValueError("Empty Sh table.")

    if col < 0:
        col = T.shape[1] + col
    col = min(max(col, 0), T.shape[1] - 1)
    return T[:, col]


def load_sh_file(data_dir: Path, geometry: str, flow: str):
    path = data_dir / f"Sh_{geometry}_{flow}.txt"
    return load_optional_table(path), path


# =========================================================
# Main plot
# =========================================================
def plot_sh_vs_da(data_dir: Path, output_base: Path, geometries: list[str]):
    Da_raw = load_vector(data_dir / "Da.txt")

    Pe_path = data_dir / "Pe.txt"
    Pe_raw = load_vector(Pe_path) if Pe_path.exists() else None

    i_da = np.argsort(Da_raw)
    Da = Da_raw[i_da]

    fig, ax = plt.subplots(figsize=(11.5, 8.0))

    any_curve = False

    for geometry in geometries:
        config = GEOMETRIES[geometry]
        label = config["label"]
        color = config["color"]

        # Plug flow: Sh_<geometry>_plug.txt
        Sh_plug_raw, plug_path = load_sh_file(data_dir, geometry, "plug")
        if Sh_plug_raw is not None:
            Sh_plug = extract_curve_from_vector_or_table(
                Sh_plug_raw, Da=Da_raw, Pe=Pe_raw, col=0
            )
            Sh_plug = Sh_plug[i_da]
            ax.semilogx(
                Da,
                Sh_plug,
                color=color,
                lw=2.8,
                ls="-",
                label=f"{label}: plug",
            )
            any_curve = True
            print(f"Loaded plug curve: {plug_path}")
        else:
            print(f"Skipping {geometry} plug: missing {plug_path}")

        # Poiseuille flow: Sh_<geometry>_pois.txt
        Sh_pois_raw, pois_path = load_sh_file(data_dir, geometry, "pois")
        if Sh_pois_raw is not None:
            if Pe_raw is None:
                raise FileNotFoundError(
                    f"Need {Pe_path} to plot low/high Pe curves from {pois_path}."
                )

            T = orient_table(Sh_pois_raw, Da_raw, Pe_raw, "Sh")
            # Sort both axes.
            i_pe = np.argsort(Pe_raw)
            Pe_sorted = Pe_raw[i_pe]
            T = T[np.ix_(i_da, i_pe)]

            ax.semilogx(
                Da,
                T[:, 0],
                color=color,
                lw=2.8,
                ls="--",
                label=fr"{label}: Poiseuille, low $\mathrm{{Pe}}$",
            )
            ax.semilogx(
                Da,
                T[:, -1],
                color=color,
                lw=2.8,
                ls=":",
                label=fr"{label}: Poiseuille, high $\mathrm{{Pe}}$",
            )
            any_curve = True
            print(
                f"Loaded Poiseuille curves: {pois_path} "
                f"(low Pe={Pe_sorted[0]:.6g}, high Pe={Pe_sorted[-1]:.6g})"
            )
        else:
            print(f"Skipping {geometry} Poiseuille: missing {pois_path}")

    if not any_curve:
        raise FileNotFoundError(
            f"No Sh_<geometry>_<flow>.txt files found in {data_dir}."
        )

    ax.set_xlabel(r"$\mathrm{Da}$")
    ax.set_ylabel(r"$\mathrm{Sh}$")
    ax.set_title(r"Sherwood number $\mathrm{Sh}$")
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.10)
    ax.legend(frameon=True, fontsize=10, ncol=1)

    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_base = args.output_dir / args.output_base

    plot_sh_vs_da(
        data_dir=args.data_dir,
        output_base=output_base,
        geometries=args.geometries,
    )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
