#!/usr/bin/env python3
"""
Plug-flow Sherwood-number solver for the extended Graetz problem.

Supported geometries
--------------------
    tube      : circular tube, one reactive cylindrical wall
    plates    : parallel plates with two reactive surfaces, symmetric half-gap
    oneplate  : parallel plates with one inert wall and one reactive wall

The plug-flow eigenvalue problem is independent of Pe.  Therefore Sh(Da) and
chi(Da) are computed from the transverse eigenvalue only.  If requested, the
script also saves plug-flow entrance-length estimates based on the first two
transverse eigenvalues:

    Le(Pe -> 0, Da)  = 1 / (kappa2 - kappa1)
    Le(Pe >> 1, Da) ~ Pe / (kappa2^2 - kappa1^2)

The exact plug-flow entrance length for a supplied Pe is also available because
    beta_n = (-Pe + sqrt(Pe^2 + 4 kappa_n^2)) / 2.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import root_scalar
from scipy.special import j0, j1, jn_zeros


# =========================================================
# Geometry definitions
# =========================================================
GEOM_TUBE = "tube"
GEOM_PLATES = "plates"
GEOM_ONEPLATE = "oneplate"
GEOMETRIES = (GEOM_TUBE, GEOM_PLATES, GEOM_ONEPLATE)


@dataclass(frozen=True)
class GeometryConfig:
    name: str
    label: str
    hydraulic_multiplier: float
    transverse_problem: str


GEOMETRY_CONFIGS = {
    GEOM_TUBE: GeometryConfig(
        name=GEOM_TUBE,
        label="Circular tube, plug flow",
        hydraulic_multiplier=2.0,          # d_h / R = 2
        transverse_problem="tube",
    ),
    GEOM_PLATES: GeometryConfig(
        name=GEOM_PLATES,
        label="Parallel plates, plug flow, two reactive surfaces",
        hydraulic_multiplier=4.0,          # d_h / a = 4, half-gap formulation
        transverse_problem="plates",
    ),
    GEOM_ONEPLATE: GeometryConfig(
        name=GEOM_ONEPLATE,
        label="Parallel plates, plug flow, one reactive surface",
        hydraulic_multiplier=2.0,          # d_h / a = 2, full-gap one-wall formulation
        transverse_problem="plates",       # same eigenvalue equation as plates
    ),
}


# =========================================================
# Transverse residuals
# =========================================================
def residual_kappa_tube(kappa, Da):
    """
    Tube plug flow:
        phi(rho) = J0(kappa rho)
        kappa J1(kappa) = Da J0(kappa)
    """
    return kappa * j1(kappa) - Da * j0(kappa)


def residual_kappa_plates(kappa, Da):
    """
    Parallel-plate plug flow, including the one-reactive-wall geometry:
        phi(rho) = cos(kappa rho)
        kappa sin(kappa) = Da cos(kappa)
    """
    return kappa * np.sin(kappa) - Da * np.cos(kappa)


def residual_kappa(kappa, Da, geometry):
    config = GEOMETRY_CONFIGS[geometry]
    if config.transverse_problem == "tube":
        return residual_kappa_tube(kappa, Da)
    if config.transverse_problem == "plates":
        return residual_kappa_plates(kappa, Da)
    raise ValueError(f"Unknown transverse problem: {config.transverse_problem}")


# =========================================================
# Root bracketing
# =========================================================
def _brent_root(fun, bracket, args=(), xtol=1e-13, rtol=1e-13):
    a, b = bracket
    fa = fun(a, *args)
    fb = fun(b, *args)

    if not (np.isfinite(fa) and np.isfinite(fb)):
        raise RuntimeError(f"Non-finite bracket residuals: {fa}, {fb}")
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0:
        raise RuntimeError(
            f"Root is not bracketed on {bracket}: f(a)={fa}, f(b)={fb}"
        )

    res = root_scalar(
        fun,
        args=args,
        bracket=bracket,
        method="brentq",
        xtol=xtol,
        rtol=rtol,
    )
    if not res.converged:
        raise RuntimeError(f"Root solve did not converge on bracket {bracket}")
    return res.root


def find_kappa1_kappa2(Da, geometry):
    """
    Return the first two positive transverse eigenvalues.

    For plates/oneplate, roots are bracketed in
        kappa1 in (0, pi/2), kappa2 in (pi, 3pi/2).

    For tube, roots are bracketed in
        kappa1 in (0, j0,1), kappa2 in (j0,1, j0,2),
    where j0,n are zeros of J0.
    """
    if geometry not in GEOMETRIES:
        raise ValueError(f"Unknown geometry: {geometry}")
    if Da < 0:
        raise ValueError("Da must be non-negative")

    # Exact Da = 0 limits.  These are useful for diagnostics, although the
    # production grids normally use Da > 0.
    if Da == 0.0:
        if GEOMETRY_CONFIGS[geometry].transverse_problem == "tube":
            return 0.0, float(jn_zeros(1, 1)[0])
        return 0.0, np.pi

    eps = 1e-12
    config = GEOMETRY_CONFIGS[geometry]

    if config.transverse_problem == "plates":
        kappa1 = _brent_root(
            residual_kappa_plates,
            (eps, 0.5 * np.pi - eps),
            args=(Da,),
        )
        kappa2 = _brent_root(
            residual_kappa_plates,
            (np.pi + eps, 1.5 * np.pi - eps),
            args=(Da,),
        )
        return kappa1, kappa2

    if config.transverse_problem == "tube":
        j0_zeros = jn_zeros(0, 2)
        j01, j02 = float(j0_zeros[0]), float(j0_zeros[1])

        kappa1 = _brent_root(
            residual_kappa_tube,
            (eps, j01 - eps),
            args=(Da,),
        )
        kappa2 = _brent_root(
            residual_kappa_tube,
            (j01 + eps, j02 - eps),
            args=(Da,),
        )
        return kappa1, kappa2

    raise ValueError(f"Unknown transverse problem: {config.transverse_problem}")


# =========================================================
# Analytic plug-flow averages and Sherwood number
# =========================================================
def _denom_tan_over_kappa_minus_one(kappa):
    """tan(kappa)/kappa - 1, with a small-kappa series fallback."""
    k = float(kappa)
    if abs(k) < 1e-4:
        k2 = k * k
        return k2 / 3.0 + 2.0 * k2 * k2 / 15.0 + 17.0 * k2**3 / 315.0
    return np.tan(k) / k - 1.0


def _denom_tube_avg_minus_one(kappa):
    """2 J1(kappa)/(kappa J0(kappa)) - 1, with a small-kappa series fallback."""
    k = float(kappa)
    if abs(k) < 1e-4:
        k2 = k * k
        return k2 / 8.0 + k2 * k2 / 48.0 + 11.0 * k2**3 / 3072.0
    return 2.0 * j1(k) / (k * j0(k)) - 1.0


def plug_ratios(kappa, geometry):
    """
    Return Cm/Cw, Cavg/Cw, and Cavg/Cm.

    Since plug flow has U = 1, Cm = Cavg and chi = Cavg/Cm = 1 exactly.
    """
    config = GEOMETRY_CONFIGS[geometry]

    if config.transverse_problem == "plates":
        denom = _denom_tan_over_kappa_minus_one(kappa)
    elif config.transverse_problem == "tube":
        denom = _denom_tube_avg_minus_one(kappa)
    else:
        raise ValueError(f"Unknown transverse problem: {config.transverse_problem}")

    Cm_over_Cw = 1.0 + denom
    return Cm_over_Cw, Cm_over_Cw, 1.0


def sherwood_characteristic(Da, kappa, geometry):
    """
    Sherwood number based on the transverse length scale ell.

    Uses analytic plug-flow expressions rather than numerical quadrature:
        Sh_ell = Da / (Cm/Cw - 1).
    """
    if Da == 0.0:
        return weak_da_limit_characteristic(geometry)

    config = GEOMETRY_CONFIGS[geometry]

    if config.transverse_problem == "plates":
        denom = _denom_tan_over_kappa_minus_one(kappa)
    elif config.transverse_problem == "tube":
        denom = _denom_tube_avg_minus_one(kappa)
    else:
        raise ValueError(f"Unknown transverse problem: {config.transverse_problem}")

    if denom <= 0 or not np.isfinite(denom):
        raise RuntimeError(f"Invalid Sherwood denominator: {denom}")

    return Da / denom


def weak_da_limit_characteristic(geometry):
    """Da -> 0 limit of Sh based on ell."""
    if GEOMETRY_CONFIGS[geometry].transverse_problem == "tube":
        return 4.0
    return 3.0


def dirichlet_limit_hydraulic(geometry):
    """Da -> infinity limit of hydraulic-diameter Sh."""
    config = GEOMETRY_CONFIGS[geometry]
    if config.transverse_problem == "tube":
        j01 = float(jn_zeros(0, 1)[0])
        return config.hydraulic_multiplier * (j01**2 / 2.0)
    # plates and oneplate have same Sh_ell limit but different multiplier
    return config.hydraulic_multiplier * (np.pi**2 / 4.0)


def weak_da_limit_hydraulic(geometry):
    """Da -> 0 limit of hydraulic-diameter Sh."""
    return GEOMETRY_CONFIGS[geometry].hydraulic_multiplier * weak_da_limit_characteristic(geometry)


# =========================================================
# Entrance-length formulas for plug flow
# =========================================================
def beta_from_kappa(kappa, Pe):
    """beta from kappa^2 = beta^2 + Pe beta."""
    kappa = np.asarray(kappa, dtype=float)
    Pe = np.asarray(Pe, dtype=float)
    return 0.5 * (-Pe + np.sqrt(Pe**2 + 4.0 * kappa**2))


def entrance_length_exact(Pe, kappa1, kappa2):
    """Exact plug-flow modal entrance length 1/(beta2 - beta1)."""
    beta1 = beta_from_kappa(kappa1, Pe)
    beta2 = beta_from_kappa(kappa2, Pe)
    return 1.0 / (beta2 - beta1)


def entrance_length_low_pe(kappa1, kappa2):
    """Pe -> 0 limit of entrance length."""
    return 1.0 / (kappa2 - kappa1)


def entrance_length_high_pe_slope(kappa1, kappa2):
    """High-Pe prefactor b(Da) in Le ~ b(Da) Pe."""
    return 1.0 / (kappa2**2 - kappa1**2)


# =========================================================
# Main point solver
# =========================================================
def solve_plug_flow(Da, geometry, Pe=None):
    """
    Solve the plug-flow eigenvalue problem for one Da and geometry.

    Sh_Dh and chi depend only on Da.  If Pe is supplied, beta1, beta2, and the
    exact plug-flow entrance length are also returned.
    """
    if geometry not in GEOMETRIES:
        raise ValueError(f"Unknown geometry: {geometry}. Choose from {GEOMETRIES}.")

    kappa1, kappa2 = find_kappa1_kappa2(Da, geometry)
    Cm_over_Cw, Cavg_over_Cw, Cavg_over_Cm = plug_ratios(kappa1, geometry)

    Sh_ell = sherwood_characteristic(Da, kappa1, geometry)
    Sh_Dh = GEOMETRY_CONFIGS[geometry].hydraulic_multiplier * Sh_ell

    out = {
        "geometry": geometry,
        "Da": Da,
        "kappa1": kappa1,
        "kappa2": kappa2,
        "Cm_over_Cw": Cm_over_Cw,
        "Cavg_over_Cw": Cavg_over_Cw,
        "Cavg_over_Cm": Cavg_over_Cm,
        "chi": 1.0,
        "Sh_characteristic": Sh_ell,
        "Sh_Dh": Sh_Dh,
        "Le_low_Pe": entrance_length_low_pe(kappa1, kappa2),
        "Le_high_Pe_slope": entrance_length_high_pe_slope(kappa1, kappa2),
        "wall_bc_residual": residual_kappa(kappa1, Da, geometry),
    }

    if Pe is not None:
        beta1 = beta_from_kappa(kappa1, Pe)
        beta2 = beta_from_kappa(kappa2, Pe)
        out.update(
            {
                "Pe": Pe,
                "beta1": beta1,
                "beta2": beta2,
                "entrance_length": 1.0 / (beta2 - beta1),
            }
        )

    return out


# =========================================================
# Table generation
# =========================================================
def build_da_table(Da_tab, geometry):
    rows = []
    for Da in Da_tab:
        res = solve_plug_flow(Da=Da, geometry=geometry)
        rows.append(
            [
                res["Da"],
                res["kappa1"],
                res["kappa2"],
                res["Sh_Dh"],
                res["Sh_characteristic"],
                res["Cm_over_Cw"],
                res["chi"],
                res["Le_low_Pe"],
                res["Le_high_Pe_slope"],
                res["wall_bc_residual"],
            ]
        )
    return np.asarray(rows, dtype=float)


def build_entrance_map(Da_tab, Pe_tab, geometry):
    Le = np.full((len(Da_tab), len(Pe_tab)), np.nan, dtype=float)
    beta1 = np.full_like(Le, np.nan)
    beta2 = np.full_like(Le, np.nan)

    for i, Da in enumerate(Da_tab):
        res = solve_plug_flow(Da=Da, geometry=geometry)
        k1 = res["kappa1"]
        k2 = res["kappa2"]
        b1 = beta_from_kappa(k1, Pe_tab)
        b2 = beta_from_kappa(k2, Pe_tab)

        beta1[i, :] = b1
        beta2[i, :] = b2
        Le[i, :] = 1.0 / (b2 - b1)

    return Le, beta1, beta2


# =========================================================
# CLI
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute plug-flow Sh(Da) and optional entrance-length estimates."
    )
    parser.add_argument(
        "geometry",
        choices=GEOMETRIES,
        help="Geometry: tube, plates, or oneplate.",
    )
    parser.add_argument("--da-min", type=float, default=1e-3)
    parser.add_argument("--da-max", type=float, default=1e3)
    parser.add_argument("--n-da", type=int, default=1000)
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory for output text files. Default: data",
    )
    parser.add_argument(
        "--with-entrance",
        action="store_true",
        help=(
            "Save entrance-length estimates versus Da: Pe->0 limit and "
            "high-Pe prefactor in Le ~ b Pe."
        ),
    )
    parser.add_argument(
        "--entrance-map",
        action="store_true",
        help="Also save exact plug-flow Le(Pe,Da), beta1, and beta2 tables.",
    )
    parser.add_argument("--pe-min", type=float, default=1e-3)
    parser.add_argument("--pe-max", type=float, default=1e3)
    parser.add_argument("--n-pe", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    geometry = args.geometry
    config = GEOMETRY_CONFIGS[geometry]

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    Da_tab = np.logspace(np.log10(args.da_min), np.log10(args.da_max), args.n_da)
    table = build_da_table(Da_tab, geometry)

    prefix = f"plug_{geometry}"

    np.savetxt(outdir / f"Da_{prefix}.txt", Da_tab)
    np.savetxt(outdir / f"Sh_{geometry}_plug.txt", table[:, 3])
    np.savetxt(
        outdir / f"{prefix}_Da_table.txt",
        table,
        header=(
            "Da kappa1 kappa2 Sh_Dh Sh_characteristic Cm_over_Cw chi "
            "Le_low_Pe Le_high_Pe_slope wall_bc_residual"
        ),
    )

    if args.with_entrance or args.entrance_map:
        np.savetxt(
            outdir / f"Le_estimate_{prefix}.txt",
            table[:, [0, 1, 2, 7, 8]],
            header="Da kappa1 kappa2 Le_low_Pe Le_high_Pe_slope_for_Le_over_Pe",
        )

    if args.entrance_map:
        Pe_tab = np.logspace(np.log10(args.pe_min), np.log10(args.pe_max), args.n_pe)
        Le, beta1, beta2 = build_entrance_map(Da_tab, Pe_tab, geometry)

        np.savetxt(outdir / f"Pe_{prefix}.txt", Pe_tab)
        np.savetxt(outdir / f"Le_{prefix}.txt", Le)
        np.savetxt(outdir / f"beta1_{prefix}.txt", beta1)
        np.savetxt(outdir / f"beta2_{prefix}.txt", beta2)

    print(f"\n{config.label}")
    print(f"Saved: {outdir / f'{prefix}_Da_table.txt'}")
    print(f"Saved: {outdir / f'Sh_{geometry}_plug.txt'}")
    print(f"Weak-Da hydraulic limit:       {weak_da_limit_hydraulic(geometry):.12g}")
    print(f"Dirichlet hydraulic limit:     {dirichlet_limit_hydraulic(geometry):.12g}")
    print(f"Computed Sh at Da_min:         {table[0, 3]:.12g}")
    print(f"Computed Sh at Da_max:         {table[-1, 3]:.12g}")

    if args.with_entrance or args.entrance_map:
        Le0 = table[:, 7]
        b = table[:, 8]
        print(f"Low-Pe Le range:              {np.nanmin(Le0):.6g}--{np.nanmax(Le0):.6g}")
        print(f"High-Pe Le/Pe range:          {np.nanmin(b):.6g}--{np.nanmax(b):.6g}")
        print(f"Saved: {outdir / f'Le_estimate_{prefix}.txt'}")

    if args.entrance_map:
        print(f"Saved: {outdir / f'Le_{prefix}.txt'}")


if __name__ == "__main__":
    main()
