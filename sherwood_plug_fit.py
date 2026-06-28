#!/usr/bin/env python3
"""
Fit the plug-flow Sherwood-number correlation used in the manuscript.

The plug-flow transverse eigenvalue problem is independent of Pe, so the
fully developed hydraulic-diameter Sherwood number is fitted as a function of
Da only:

    Sh_plug(Da) = Sh_DaInf + (Sh_Da0 - Sh_DaInf) / (1 + Da / Da_c)

where the two limiting Sherwood numbers are fixed to their exact values and
only the crossover Damkohler number Da_c is fitted.

Supported geometries
--------------------
tube     : circular tube, one reactive cylindrical wall
plates   : parallel plates with two reactive surfaces
oneplate : parallel plates with one inert wall and one reactive wall

Typical usage
-------------
    python sherwood_plug_fit.py tube
    python sherwood_plug_fit.py plates
    python sherwood_plug_fit.py oneplate
    python sherwood_plug_fit.py all

By default the script reads data/Sh_<geometry>_plug.txt and, if available,
data/Da_plug_<geometry>.txt.  It falls back to data/Da.txt only if its length
matches the Sherwood table.  For oneplate, if Sh_oneplate_plug.txt is absent
but Sh_plates_plug.txt is present, the script can use Sh_oneplate =
Sh_plates/2, which is the exact plug-flow relation for the hydraulic-diameter
normalization used in the manuscript.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import least_squares
from scipy.special import jn_zeros


# =========================================================
# Geometry-specific constants
# =========================================================
J01 = float(jn_zeros(0, 1)[0])


@dataclass(frozen=True)
class GeometryConfig:
    name: str
    label: str
    sh_file: str
    da_file: str
    output_prefix: str
    sh_da0_exact: float
    sh_dainf_exact: float
    fallback_from_plates_factor: float | None = None


GEOMETRY_CONFIGS: dict[str, GeometryConfig] = {
    "tube": GeometryConfig(
        name="tube",
        label="Circular tube, plug flow",
        sh_file="Sh_tube_plug.txt",
        da_file="Da_plug_tube.txt",
        output_prefix="tube_plug",
        # hydraulic-diameter Sherwood, d_h = 2R
        sh_da0_exact=8.0,
        sh_dainf_exact=J01**2,
    ),
    "plates": GeometryConfig(
        name="plates",
        label="Parallel plates, plug flow, two reactive surfaces",
        sh_file="Sh_plates_plug.txt",
        da_file="Da_plug_plates.txt",
        output_prefix="plates_plug",
        # hydraulic-diameter Sherwood, d_h = 4a
        sh_da0_exact=12.0,
        sh_dainf_exact=np.pi**2,
    ),
    "oneplate": GeometryConfig(
        name="oneplate",
        label="Parallel plates, plug flow, one reactive surface",
        sh_file="Sh_oneplate_plug.txt",
        da_file="Da_plug_oneplate.txt",
        output_prefix="oneplate_plug",
        # hydraulic-diameter Sherwood, d_h = 2b
        sh_da0_exact=6.0,
        sh_dainf_exact=0.5 * np.pi**2,
        # For plug flow, Sh_oneplate,plug = Sh_plates,plug / 2 exactly.
        fallback_from_plates_factor=0.5,
    ),
}

GEOMETRIES = tuple(GEOMETRY_CONFIGS.keys())


# =========================================================
# CLI
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the one-parameter plug-flow Sherwood correlation "
            "Sh(Da) = Sh_inf + (Sh_0 - Sh_inf)/(1 + Da/Da_c)."
        )
    )
    parser.add_argument(
        "geometry",
        choices=(*GEOMETRIES, "all"),
        help="Geometry to fit: tube, plates, oneplate, or all.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing plug-flow Sh and Da text files. Default: data",
    )
    parser.add_argument(
        "--output-dir",
        default="fits",
        help="Directory for fit summary files. Default: fits",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output file path. Only allowed when fitting a single geometry; "
            "overrides --output-dir."
        ),
    )
    parser.add_argument(
        "--sh-file",
        default=None,
        help="Optional Sherwood data file name/path for a single-geometry fit.",
    )
    parser.add_argument(
        "--da-file",
        default=None,
        help="Optional Da grid file name/path for a single-geometry fit.",
    )
    parser.add_argument(
        "--log-residuals",
        action="store_true",
        help="Fit using log(Sh_fit)-log(Sh_data) instead of relative residuals.",
    )
    parser.add_argument(
        "--no-oneplate-fallback",
        action="store_true",
        help=(
            "Disable the exact plug-flow fallback Sh_oneplate = Sh_plates/2 "
            "when Sh_oneplate_plug.txt is missing."
        ),
    )
    return parser.parse_args()


# =========================================================
# Utilities
# =========================================================
def round4(x: float) -> float:
    return float(np.round(x, 4))


def _resolve_path(path_or_name: str | None, data_dir: Path) -> Path | None:
    if path_or_name is None:
        return None
    p = Path(path_or_name)
    if p.is_absolute() or p.parent != Path("."):
        return p
    return data_dir / p


def _load_1d(path: Path, name: str) -> np.ndarray:
    try:
        arr = np.loadtxt(path)
    except OSError as exc:
        raise FileNotFoundError(f"Could not read {name} file: {path}") from exc

    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def _squeeze_or_average_sh(Sh_raw: np.ndarray, n_da: int | None = None) -> np.ndarray:
    """Return a 1D plug-flow Sh(Da) curve from a vector or table."""
    Sh = np.asarray(Sh_raw, dtype=float)

    if Sh.ndim == 1:
        return Sh.ravel()

    if Sh.ndim != 2:
        raise ValueError(f"Plug-flow Sh data must be 1D or 2D, got shape {Sh.shape}.")

    # Accept a row/column vector.
    if 1 in Sh.shape:
        return Sh.ravel()

    # If a redundant Pe dimension is present, average over it.
    if n_da is not None:
        if Sh.shape[0] == n_da:
            return np.nanmean(Sh, axis=1)
        if Sh.shape[1] == n_da:
            return np.nanmean(Sh, axis=0)

    raise ValueError(
        f"Could not infer Da axis in plug-flow Sh table with shape {Sh.shape}. "
        "Use a 1D file or pass a matching --da-file."
    )


def _load_da_grid(config: GeometryConfig, data_dir: Path, da_file_arg: str | None) -> tuple[np.ndarray, Path]:
    candidates: list[Path] = []

    if da_file_arg is not None:
        candidates.append(_resolve_path(da_file_arg, data_dir))
    else:
        candidates.append(data_dir / config.da_file)
        # The one-reactive-wall plug-flow problem uses the same transverse
        # spectrum as the two-wall plate problem, so its generated table often
        # shares the two-wall plate Da grid when Sh_oneplate is reconstructed
        # from Sh_plates/2.
        if config.name == "oneplate":
            candidates.append(data_dir / "Da_plug_plates.txt")
        candidates.append(data_dir / "Da.txt")

    for path in candidates:
        if path is not None and path.exists():
            Da = _load_1d(path, "Da").ravel()
            return Da, path

    raise FileNotFoundError(
        "Could not find a Da grid. Tried: "
        + ", ".join(str(p) for p in candidates if p is not None)
    )


def _load_sh_data(
    config: GeometryConfig,
    data_dir: Path,
    sh_file_arg: str | None,
    allow_oneplate_fallback: bool,
) -> tuple[np.ndarray, Path, str]:
    if sh_file_arg is not None:
        path = _resolve_path(sh_file_arg, data_dir)
        if path is None:
            raise FileNotFoundError("Internal error: unresolved --sh-file path.")
        Sh = _load_1d(path, "Sh")
        return Sh, path, "direct"

    path = data_dir / config.sh_file
    if path.exists():
        Sh = _load_1d(path, "Sh")
        return Sh, path, "direct"

    if (
        config.name == "oneplate"
        and allow_oneplate_fallback
        and config.fallback_from_plates_factor is not None
    ):
        plates_path = data_dir / "Sh_plates_plug.txt"
        if plates_path.exists():
            Sh_plates = _load_1d(plates_path, "Sh_plates_plug")
            return (
                config.fallback_from_plates_factor * Sh_plates,
                plates_path,
                "exact fallback: Sh_oneplate = Sh_plates/2",
            )

    raise FileNotFoundError(
        f"Could not find Sherwood data file {path}. "
        "Generate it with sherwood_plug.py or pass --sh-file."
    )


def load_data(
    config: GeometryConfig,
    data_dir: str = "data",
    sh_file_arg: str | None = None,
    da_file_arg: str | None = None,
    allow_oneplate_fallback: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    data_path = Path(data_dir)

    Da_raw, da_path = _load_da_grid(config, data_path, da_file_arg)
    Sh_raw, sh_path, sh_source = _load_sh_data(
        config,
        data_path,
        sh_file_arg,
        allow_oneplate_fallback=allow_oneplate_fallback,
    )

    # First try to pair Sh with the loaded Da file.
    Sh = _squeeze_or_average_sh(Sh_raw, n_da=len(Da_raw))
    Da = np.asarray(Da_raw, dtype=float).ravel()

    if len(Sh) != len(Da):
        # If the default generic Da.txt was selected but the geometry-specific
        # file exists, try that before failing.  This catches repositories that
        # have both Poiseuille and plug grids.
        geom_da_path = data_path / config.da_file
        if da_file_arg is None and da_path.name == "Da.txt" and geom_da_path.exists():
            Da2 = _load_1d(geom_da_path, "Da").ravel()
            Sh2 = _squeeze_or_average_sh(Sh_raw, n_da=len(Da2))
            if len(Sh2) == len(Da2):
                Da, Sh, da_path = Da2, Sh2, geom_da_path

    if len(Sh) != len(Da):
        raise ValueError(
            f"Length mismatch: Da file {da_path} has {len(Da)} points, "
            f"but Sh file/source {sh_path} gives {len(Sh)} points. "
            "Use --da-file to select the matching plug-flow Da grid."
        )

    mask = np.isfinite(Da) & np.isfinite(Sh) & (Da > 0.0) & (Sh > 0.0)
    if np.count_nonzero(mask) < 2:
        raise RuntimeError("Need at least two finite positive (Da, Sh) points to fit Da_c.")

    Da = Da[mask]
    Sh = Sh[mask]

    idx = np.argsort(Da)
    Da = Da[idx]
    Sh = Sh[idx]

    metadata = {
        "da_file": str(da_path),
        "sh_file": str(sh_path),
        "sh_source": sh_source,
    }
    return Da, Sh, metadata


# =========================================================
# Model and fit
# =========================================================
def sh_plug_model_physical(Da: np.ndarray, Da_c: float, Sh_da0: float, Sh_dainf: float) -> np.ndarray:
    Da = np.asarray(Da, dtype=float)
    return Sh_dainf + (Sh_da0 - Sh_dainf) / (1.0 + Da / Da_c)


def sh_plug_model_log(logDa_c: np.ndarray, Da: np.ndarray, config: GeometryConfig) -> np.ndarray:
    Da_c = 10.0 ** float(np.atleast_1d(logDa_c)[0])
    return sh_plug_model_physical(
        Da,
        Da_c=Da_c,
        Sh_da0=config.sh_da0_exact,
        Sh_dainf=config.sh_dainf_exact,
    )


def residuals_relative(logDa_c: np.ndarray, Da: np.ndarray, Sh_data: np.ndarray, config: GeometryConfig) -> np.ndarray:
    Sh_fit = sh_plug_model_log(logDa_c, Da, config)
    if np.any(~np.isfinite(Sh_fit)) or np.any(Sh_fit <= 0.0):
        return 1e6 * np.ones_like(Sh_data)
    return (Sh_fit - Sh_data) / Sh_data


def residuals_log(logDa_c: np.ndarray, Da: np.ndarray, Sh_data: np.ndarray, config: GeometryConfig) -> np.ndarray:
    Sh_fit = sh_plug_model_log(logDa_c, Da, config)
    if np.any(~np.isfinite(Sh_fit)) or np.any(Sh_fit <= 0.0):
        return 1e6 * np.ones_like(Sh_data)
    return np.log(Sh_fit) - np.log(Sh_data)


def fit_da_c(Da: np.ndarray, Sh: np.ndarray, config: GeometryConfig, use_log_residuals: bool = False):
    residual_fun = residuals_log if use_log_residuals else residuals_relative

    lda_min = float(np.log10(np.min(Da)))
    lda_max = float(np.log10(np.max(Da)))
    lda_mid = 0.5 * (lda_min + lda_max)

    guesses = np.array([[lda_mid], [lda_min], [lda_max], [0.0]], dtype=float)
    lower = np.array([lda_min - 6.0])
    upper = np.array([lda_max + 6.0])

    best = None
    best_cost = np.inf

    for p0 in guesses:
        p0 = np.clip(p0, lower + 1e-12, upper - 1e-12)
        res = least_squares(
            residual_fun,
            p0,
            args=(Da, Sh, config),
            bounds=(lower, upper),
            method="trf",
            loss="soft_l1",
            f_scale=0.05,
            max_nfev=50000,
        )
        if res.cost < best_cost:
            best = res
            best_cost = res.cost

    if best is None:
        raise RuntimeError("Plug-flow least-squares fit failed to start.")
    return best


# =========================================================
# Diagnostics and summary
# =========================================================
def fit_quality(Sh_true: np.ndarray, Sh_fit: np.ndarray) -> dict[str, float]:
    rel_err = (Sh_fit - Sh_true) / Sh_true
    abs_rel_err = np.abs(rel_err)
    return {
        "mean_abs_rel": float(np.mean(abs_rel_err)),
        "median_abs_rel": float(np.median(abs_rel_err)),
        "max_abs_rel": float(np.max(abs_rel_err)),
        "rms_rel": float(np.sqrt(np.mean(rel_err**2))),
    }


def worst_point(Da: np.ndarray, Sh_true: np.ndarray, Sh_fit: np.ndarray) -> dict[str, float]:
    rel_err = (Sh_fit - Sh_true) / Sh_true
    abs_rel_err = np.abs(rel_err)
    imax = int(np.argmax(abs_rel_err))
    return {
        "Da": float(Da[imax]),
        "Sh_true": float(Sh_true[imax]),
        "Sh_fit": float(Sh_fit[imax]),
        "rel_err": float(rel_err[imax]),
        "abs_rel_err": float(abs_rel_err[imax]),
    }


def corner_limits(Da_c: float, config: GeometryConfig) -> dict[str, float]:
    return {
        "Da_to_0": float(sh_plug_model_physical(1e-30, Da_c, config.sh_da0_exact, config.sh_dainf_exact)),
        "Da_to_inf": float(sh_plug_model_physical(1e30, Da_c, config.sh_da0_exact, config.sh_dainf_exact)),
    }


def add_key_values(lines: list[str], title: str, values: dict[str, object], float_fmt: str = ".12g") -> None:
    lines.append("")
    lines.append(title)
    lines.append("-" * len(title))
    for key, val in values.items():
        if isinstance(val, (float, np.floating)):
            lines.append(f"{key:20s} = {val:{float_fmt}}")
        else:
            lines.append(f"{key:20s} = {val}")


def build_summary_text(
    config: GeometryConfig,
    args: argparse.Namespace,
    metadata: dict[str, str],
    Da_c_full: float,
    Da_c_round: float,
    quality_full: dict[str, float],
    quality_round: dict[str, float],
    worst_full: dict[str, float],
    worst_round: dict[str, float],
    corners_round: dict[str, float],
    n_valid_points: int,
) -> str:
    lines: list[str] = []

    lines.append("Plug-flow Sherwood correlation fit summary")
    lines.append("===========================================")
    lines.append(f"geometry              = {config.name}")
    lines.append(f"label                 = {config.label}")
    lines.append(f"Sh source             = {metadata['sh_file']}")
    lines.append(f"Sh source mode        = {metadata['sh_source']}")
    lines.append(f"Da file               = {metadata['da_file']}")
    lines.append(f"valid fitted points   = {n_valid_points}")
    lines.append(f"residuals             = {'logarithmic' if args.log_residuals else 'relative'}")

    lines.append("")
    lines.append("Fitted structure")
    lines.append("----------------")
    lines.append("Sh_plug(Da) = Sh_DaInf + (Sh_Da0 - Sh_DaInf)/(1 + Da/Da_c)")
    lines.append("Only Da_c is fitted; both limiting Sherwood numbers are fixed.")

    add_key_values(
        lines,
        "Exact constants",
        {
            "Sh_Da0": config.sh_da0_exact,
            "Sh_DaInf": config.sh_dainf_exact,
        },
    )

    add_key_values(lines, "Full-precision fitted parameter", {"Da_c": Da_c_full})

    add_key_values(
        lines,
        "Rounded constants and fitted parameter",
        {
            "Sh_Da0": round4(config.sh_da0_exact),
            "Sh_DaInf": round4(config.sh_dainf_exact),
            "Da_c": Da_c_round,
        },
        float_fmt=".4f",
    )

    formula = (
        f"Sh_{config.name},plug(Da) = {round4(config.sh_dainf_exact):.4f} + "
        f"({round4(config.sh_da0_exact):.4f} - {round4(config.sh_dainf_exact):.4f})"
        f"/(1 + Da/{Da_c_round:.4f})"
    )
    lines.append("")
    lines.append("Rounded formula")
    lines.append("---------------")
    lines.append(formula)

    add_key_values(lines, "Corner limits from rounded formula", corners_round)
    add_key_values(lines, "Fit quality: full-precision formula", quality_full)
    add_key_values(lines, "Worst point: full-precision formula", worst_full)
    add_key_values(lines, "Fit quality: rounded formula", quality_round)
    add_key_values(lines, "Worst point: rounded formula", worst_round)

    return "\n".join(lines) + "\n"


# =========================================================
# Single/all geometry drivers
# =========================================================
def fit_one_geometry(
    geometry: str,
    args: argparse.Namespace,
    output_override: Path | None = None,
) -> Path:
    config = GEOMETRY_CONFIGS[geometry]

    Da, Sh_data, metadata = load_data(
        config,
        data_dir=args.data_dir,
        sh_file_arg=args.sh_file,
        da_file_arg=args.da_file,
        allow_oneplate_fallback=not args.no_oneplate_fallback,
    )

    res = fit_da_c(Da, Sh_data, config, use_log_residuals=args.log_residuals)
    Da_c_full = 10.0 ** float(res.x[0])
    Da_c_round = round4(Da_c_full)

    Sh_fit_full = sh_plug_model_physical(
        Da,
        Da_c=Da_c_full,
        Sh_da0=config.sh_da0_exact,
        Sh_dainf=config.sh_dainf_exact,
    )
    Sh_fit_round = sh_plug_model_physical(
        Da,
        Da_c=Da_c_round,
        Sh_da0=round4(config.sh_da0_exact),
        Sh_dainf=round4(config.sh_dainf_exact),
    )

    quality_full = fit_quality(Sh_data, Sh_fit_full)
    quality_round = fit_quality(Sh_data, Sh_fit_round)
    worst_full = worst_point(Da, Sh_data, Sh_fit_full)
    worst_round = worst_point(Da, Sh_data, Sh_fit_round)
    corners_round = corner_limits(Da_c_round, config)

    summary = build_summary_text(
        config=config,
        args=args,
        metadata=metadata,
        Da_c_full=Da_c_full,
        Da_c_round=Da_c_round,
        quality_full=quality_full,
        quality_round=quality_round,
        worst_full=worst_full,
        worst_round=worst_round,
        corners_round=corners_round,
        n_valid_points=len(Da),
    )

    if output_override is not None:
        output_file = output_override
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{config.output_prefix}_fit_summary.txt"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(summary, encoding="utf-8")
    print(f"Saved plug-flow fit summary to {output_file}")
    return output_file


def main() -> None:
    args = parse_args()

    if args.geometry == "all":
        if args.output is not None:
            raise ValueError("--output can only be used for a single geometry, not geometry='all'.")
        if args.sh_file is not None or args.da_file is not None:
            raise ValueError("--sh-file and --da-file can only be used for a single geometry.")
        for geometry in GEOMETRIES:
            fit_one_geometry(geometry, args)
        return

    output_override = Path(args.output) if args.output is not None else None
    fit_one_geometry(args.geometry, args, output_override=output_override)


if __name__ == "__main__":
    main()
