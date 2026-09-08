#!/usr/bin/env python3
"""Axisymmetric tube solver and reduced-model comparison, plug flow.

The plug-flow analog of tube_solver.py: solves the steady extended-Graetz
problem for a uniform (plug) velocity profile in a circular tube with a Robin
wall condition, evaluates the fitted plug-flow Sh(Da) correlation (chi = 1
exactly for plug flow, so there is nothing to fit there), and compares the
exact dominant axial decay rate with the decay rate predicted by the
correlated closure.

Unlike Poiseuille flow, the plug-flow transverse eigenvalue problem is
Pe-independent (see sherwood_plug.py): the dominant eigenvalue beta_1 is
available in closed form from the first root kappa_1 of
    kappa J1(kappa) = Da J0(kappa)
via beta_1 = (-Pe + sqrt(Pe^2 + 4 kappa_1^2)) / 2, so no ODE shooting or
root-scanning over beta is needed here.

Gamma is defined and fitted in analogy to tube_solver.py:
    C_m,full(Z) ~= Gamma * C_m,exact(Z)
as the geometric mean of C_m,full / C_m,exact over a user-selectable fully
developed window. Each run writes the same four data files (profiles in txt,
parameters/diagnostics in txt) and an optional concentration plot (PNG/PDF)
as tube_solver.py, so the same table columns
(beta_1, beta_corr, eps_corr, beta_const, eps_const, Gamma) can be assembled
for plug flow.

Default definitions:
    Pe = u_bar R / D
    Da = k R / D
    Z  = z / R
    Sh = 2 R j_w / [D (C_m - C_w)]

Example:
    python tube_solver_plug.py --Pe 10 --Da 100 --Lambda 5 --output-dir results
"""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import j0, j1

from sherwood_plug import beta_from_kappa, find_kappa1_kappa2, plug_ratios
from tube_solver import (
    CaseResult,
    ExactModeProperties,
    _positive_inputs,
    averaged_decay_lambdas,
    extract_sh_numbers,
    fit_downstream_gamma,
    save_concentration_plot,
    save_parameters_txt,
    save_profiles_txt,
    solve_averaged_model_semi_infinite,
    solve_axisym_ADR_steady,
    validate_args,
)


# =========================================================
# Plug-flow tube correlation
# =========================================================
# Rounded fit from fits/tube_plug_fit_summary.txt (sherwood_plug_fit.py):
#     Sh_plug(Da) = Sh_DaInf + (Sh_Da0 - Sh_DaInf) / (1 + Da / Da_c)
# chi = 1 exactly for plug flow (Cm = Ca since U = 1), so there is no
# chi correlation to fit, and -- unlike Poiseuille -- no Pe crossover: the
# plug-flow transverse eigenvalue problem is Pe-independent.
@dataclass(frozen=True)
class TubePlugCorrelationParameters:
    sh_da0: float = 8.0000
    sh_dainf: float = 5.7832
    da_c: float = 2.7784


TUBE_PLUG_CORR = TubePlugCorrelationParameters()


@dataclass(frozen=True)
class PlugCorrelationResult:
    sh: float
    chi: float


def validate_plug_correlation_window(Da: float) -> None:
    """Warn when the fitted plug-flow Sh(Da) correlation is being extrapolated."""
    if not (1.0e-3 <= Da <= 1.0e3):
        warnings.warn(
            "The plug-flow Sh correlation was fitted on 1e-3 <= Da <= 1e3; "
            "this case extrapolates beyond that window.",
            RuntimeWarning,
            stacklevel=2,
        )


def sh_fit_tube_plug(
    Da: float, p: TubePlugCorrelationParameters = TUBE_PLUG_CORR
) -> float:
    return p.sh_dainf + (p.sh_da0 - p.sh_dainf) / (1.0 + Da / p.da_c)


def evaluate_tube_plug_correlations(
    Da: float, p: TubePlugCorrelationParameters = TUBE_PLUG_CORR
) -> PlugCorrelationResult:
    validate_plug_correlation_window(Da)
    return PlugCorrelationResult(sh=sh_fit_tube_plug(Da, p), chi=1.0)


# =========================================================
# Exact plug-flow dominant mode (closed form)
# =========================================================
def exact_plug_mode_properties(Pe: float, Da: float) -> ExactModeProperties:
    """Closed-form Sh, chi, and shape integrals for the dominant plug-flow mode.

    kappa_1 is the first root of the transverse (Bessel) eigenproblem, found
    by sherwood_plug.find_kappa1_kappa2. phi(rho) = J0(kappa_1 rho), so the
    wall value, wall flux, and cross-sectional integrals below follow directly
    from J0/J1 identities rather than an ODE shoot.
    """
    kappa1, _ = find_kappa1_kappa2(Da, "tube")
    beta = float(beta_from_kappa(kappa1, Pe))

    cm_over_cw, _, _ = plug_ratios(kappa1, "tube")
    c_wall_shape = float(j0(kappa1))
    wall_flux_shape = float(kappa1 * j1(kappa1))
    cm_shape = float(cm_over_cw * c_wall_shape)
    c_area_shape = cm_shape  # chi = 1 exactly

    sh = 2.0 * wall_flux_shape / (cm_shape - c_wall_shape)
    chi = 1.0
    k_eff = 2.0 * Da * sh / (2.0 * Da + sh)
    identity_residual = chi * beta**2 + Pe * beta - k_eff

    return ExactModeProperties(
        beta=beta,
        sh=float(sh),
        chi=chi,
        cm_shape=cm_shape,
        c_area_shape=c_area_shape,
        c_wall_shape=c_wall_shape,
        wall_flux_shape=wall_flux_shape,
        identity_residual=float(identity_residual),
    )


def analytic_gamma_leading_mode(kappa1: float) -> float:
    """Closed-form leading-eigenmode amplitude for the classical (no axial
    diffusion) slug-flow Graetz series with a Robin wall condition:

        gamma_1 = 4 J1(kappa_1)^2 / { kappa_1^2 [J0(kappa_1)^2 + J1(kappa_1)^2] }

    This is the exact coefficient multiplying exp(-kappa_1^2 z/(R Pe)) in the
    mixing-cup concentration expansion for a uniform inlet and plug velocity
    profile, obtained from the standard eigenfunction-expansion projection of
    a uniform inlet condition and depending on Da only through kappa_1 (Pe
    drops out, same as Sh and chi here). It is the reference value the
    numerically fitted Gamma (see fit_downstream_gamma in tube_solver.py)
    approaches only once axial diffusion is negligible over the fit window
    AND the domain is long enough in the classical Graetz variable, i.e.
    Lambda/Pe >> 1/kappa_1^2 -- simply raising Pe at fixed Lambda instead
    shrinks that window relative to the (Pe-stretched) entrance length and
    can make the fitted Gamma diverge from this value rather than converge.
    """
    j0_k = float(j0(kappa1))
    j1_k = float(j1(kappa1))
    return 4.0 * j1_k**2 / (kappa1**2 * (j0_k**2 + j1_k**2))


# =========================================================
# High-accuracy Gamma verification (--gamma-high-accuracy)
# =========================================================
# Gamma = gamma_1 identically for the semi-infinite plug-flow problem: the
# 1/(1+beta_1/Pe) inlet-amplitude prefactor is common to the exact 2D mode
# and to the 1D single-mode (chi=1) averaged model, so it cancels out of
# their ratio, leaving Gamma = gamma_1 with no residual Pe dependence. Naive
# finite-domain fits can still disagree with gamma_1 by 1-3%, from two
# distinct, independently diagnosable numerical error sources rather than
# any missing physics:
#
#   1. Axial truncation error. For fast-decaying cases (large beta_1) a
#      fixed axial spacing under-resolves the sharp exponential profile.
#      The error is first-order in dz (confirmed by direct refinement:
#      halving dz roughly halves it), so two mesh levels + Richardson
#      extrapolation removes it.
#
#   2. Second-mode contamination. The fit window must sit many multiples of
#      the local spectral gap Le = 1/(beta_2 - beta_1) downstream of the
#      inlet for the second eigenmode to have decayed away; Le varies
#      severalfold across the (Pe, Da) grid (it is set by Da alone here,
#      since kappa_1, kappa_2 are Pe-independent), so a single fixed window
#      (e.g. a fixed fraction of a fixed Lambda) is too shallow for some
#      cases and wasteful for others. Placing the window at a fixed multiple
#      of Le for each case fixes this; a control test that instead only grew
#      Lambda while keeping the window near the inlet did not help, which
#      rules out outlet/backward-mode contamination as the dominant effect.
#
# Combining both -- a window scaled to Le, and two-mesh Richardson
# extrapolation in dz -- collapses the fitted Gamma onto gamma_1 to
# <=0.03% across Pe in [0.1, 10] and Da in [0.1, 100], with no residual
# trend against Pe, Da, or beta_1. This is opt-in (--gamma-high-accuracy)
# because it runs two 2D solves per case on grids roughly an order of
# magnitude finer than the defaults.
GAMMA_WINDOW_LE_START = 6.0
GAMMA_WINDOW_LE_END = 8.0
GAMMA_LAMBDA_MARGIN = 5.0
GAMMA_DZ_BETA_TARGET = 0.01
GAMMA_DZ_MAX = 0.003
GAMMA_NR_MIN = 100


def high_accuracy_gamma_grid(beta1: float, beta2: float) -> tuple[float, float, float, float, float]:
    """Return (Lambda, gamma_zmin_frac, gamma_zmax_frac, dz_fine, dz_coarse)
    for the high-accuracy Gamma fit, given the exact beta_1, beta_2.
    """
    Le = 1.0 / (beta2 - beta1)
    z_min = GAMMA_WINDOW_LE_START * Le
    z_max = GAMMA_WINDOW_LE_END * Le
    Lambda = z_max + GAMMA_LAMBDA_MARGIN

    dz_coarse = min(GAMMA_DZ_MAX, GAMMA_DZ_BETA_TARGET / max(beta1, 0.1))
    dz_fine = dz_coarse / 2.0

    return Lambda, z_min / Lambda, z_max / Lambda, dz_fine, dz_coarse


def fit_gamma_only(
    Pe: float,
    Da: float,
    exact_mode: ExactModeProperties,
    Lambda: float,
    gamma_zmin_frac: float,
    gamma_zmax_frac: float,
    dz: float,
    Nr: int,
    Cin: float,
    cm_min: float,
    inlet_velocity: str,
    advection_scheme: str,
) -> float:
    """Solve one 2D case and return only its fitted Gamma (vs. the exact-mode
    single-mode model), for the auxiliary coarse-grid point of a Richardson
    extrapolation. Cheaper than compute_case_plug: skips the correlated and
    constant-closure models, error breakdowns, and file-ready summary.
    """
    nz = int(math.ceil(Lambda / dz)) + 1
    z, r, C, u = solve_axisym_ADR_steady(
        Nz=nz,
        Nr=Nr,
        Lambda=Lambda,
        Pe=Pe,
        Da=Da,
        Cin=Cin,
        inlet_velocity=inlet_velocity,
        advection_scheme=advection_scheme,
        outlet_bc="mode",
        outlet_lambda=-exact_mode.beta,
        velocity_profile="plug",
    )
    cm_full, _, _, _, _, _, _ = extract_sh_numbers(z, r, C, u, Da)
    cm_exact_mode, _, _, _, _, _, _, _ = solve_averaged_model_semi_infinite(
        z=z, Pe=Pe, Da=Da, Sh=exact_mode.sh, chi=exact_mode.chi, Cin=Cin,
    )
    gamma_fit, _ = fit_downstream_gamma(
        z,
        cm_full,
        cm_exact_mode,
        z_min=gamma_zmin_frac * Lambda,
        z_max=gamma_zmax_frac * Lambda,
        cm_min=cm_min,
    )
    return gamma_fit.Gamma


# =========================================================
# Case computation
# =========================================================
def compute_case_plug(args: argparse.Namespace) -> CaseResult:
    """Solve the 2D plug-flow problem, evaluate the correlation, and fit Gamma.

    Pure computation: no printing, no file output. Mirrors tube_solver.py's
    compute_case(), with the ODE-based eigenvalue shoot replaced by the
    closed-form plug-flow root find.
    """
    _positive_inputs(args.Pe, args.Da)

    prefix = args.prefix or f"tube_plug_Pe{args.Pe:g}_Da{args.Da:g}"
    prefix = prefix.replace("/", "_").replace(" ", "_")

    corr = evaluate_tube_plug_correlations(args.Da)

    exact_mode = exact_plug_mode_properties(args.Pe, args.Da)
    beta_exact = exact_mode.beta

    kappa1, kappa2 = find_kappa1_kappa2(args.Da, "tube")
    gamma_analytic = analytic_gamma_leading_mode(kappa1)

    high_accuracy = bool(getattr(args, "gamma_high_accuracy", False))
    dz_coarse_ha = float("nan")
    if high_accuracy:
        beta2 = float(beta_from_kappa(kappa2, args.Pe))
        Le = 1.0 / (beta2 - beta_exact)
        Lambda_eff, gamma_zmin_frac_eff, gamma_zmax_frac_eff, dz_eff, dz_coarse_ha = (
            high_accuracy_gamma_grid(beta_exact, beta2)
        )
        nr_eff = max(args.Nr, GAMMA_NR_MIN)
    else:
        Le = float("nan")
        Lambda_eff = args.Lambda
        gamma_zmin_frac_eff = args.gamma_zmin_frac
        gamma_zmax_frac_eff = args.gamma_zmax_frac
        dz_eff = args.dz
        nr_eff = args.Nr

    if args.nz is not None and not high_accuracy:
        nz = args.nz
    else:
        nz = int(math.ceil(Lambda_eff / dz_eff)) + 1

    if nz < 3:
        raise ValueError("The axial grid must contain at least three points.")

    z, r, C, u = solve_axisym_ADR_steady(
        Nz=nz,
        Nr=nr_eff,
        Lambda=Lambda_eff,
        Pe=args.Pe,
        Da=args.Da,
        Cin=args.Cin,
        inlet_velocity=args.inlet_velocity,
        advection_scheme=args.advection_scheme,
        outlet_bc="mode",
        outlet_lambda=-beta_exact,
        velocity_profile="plug",
    )
    cm_full, c_area_full, chi_2d, c_wall_full, dcdr_wall, sh_eff, sh_film = (
        extract_sh_numbers(z, r, C, u, args.Da)
    )

    (
        cm_corr,
        c_area_corr,
        c_wall_corr,
        k_corr,
        lambda_minus_corr,
        lambda_plus_corr,
        amplitude_corr,
        _,
    ) = solve_averaged_model_semi_infinite(
        z=z,
        Pe=args.Pe,
        Da=args.Da,
        Sh=corr.sh,
        chi=corr.chi,
        Cin=args.Cin,
    )
    beta_corr = -lambda_minus_corr

    # Single-mode model built from the exact (closed-form) Sh/chi rather than
    # the fitted correlation, so its decay rate equals beta_exact exactly.
    # Gamma is fit against this analytic-decay model rather than cm_corr, so
    # it isolates the amplitude/inlet-region mismatch instead of also
    # absorbing the (small) decay-rate error of the fitted correlation --
    # otherwise Gamma would not be directly comparable to gamma_analytic.
    (
        cm_exact_mode,
        _,
        _,
        _,
        lambda_minus_exact,
        _,
        _,
        _,
    ) = solve_averaged_model_semi_infinite(
        z=z,
        Pe=args.Pe,
        Da=args.Da,
        Sh=exact_mode.sh,
        chi=exact_mode.chi,
        Cin=args.Cin,
    )

    (
        cm_const,
        _,
        _,
        k_const,
        lambda_minus_const,
        _,
        amplitude_const,
        _,
    ) = solve_averaged_model_semi_infinite(
        z=z,
        Pe=args.Pe,
        Da=args.Da,
        Sh=args.constant_sh,
        chi=args.constant_chi,
        Cin=args.Cin,
    )
    beta_const = -lambda_minus_const

    z_fit_min = gamma_zmin_frac_eff * Lambda_eff
    z_fit_max = gamma_zmax_frac_eff * Lambda_eff
    gamma_fit, gamma_mask = fit_downstream_gamma(
        z,
        cm_full,
        cm_exact_mode,
        z_min=z_fit_min,
        z_max=z_fit_max,
        cm_min=args.cm_min,
    )
    cm_corr_scaled = gamma_fit.Gamma * cm_corr

    gamma_analytic_rel_error = (
        (gamma_fit.Gamma - gamma_analytic) / gamma_analytic
        if gamma_analytic != 0.0
        else np.nan
    )

    # Two-mesh Richardson extrapolation: a second, cheaper solve at 2x the
    # fine-grid dz, extrapolating out the leading (first-order in dz) axial
    # truncation error. See the "High-accuracy Gamma verification" comment
    # block above analytic_gamma_leading_mode() for why this -- combined
    # with the Le-scaled window already used above -- collapses Gamma onto
    # gamma_analytic.
    gamma_richardson = float("nan")
    gamma_richardson_rel_error = float("nan")
    gamma_coarse = float("nan")
    if high_accuracy:
        gamma_coarse = fit_gamma_only(
            Pe=args.Pe,
            Da=args.Da,
            exact_mode=exact_mode,
            Lambda=Lambda_eff,
            gamma_zmin_frac=gamma_zmin_frac_eff,
            gamma_zmax_frac=gamma_zmax_frac_eff,
            dz=dz_coarse_ha,
            Nr=nr_eff,
            Cin=args.Cin,
            cm_min=args.cm_min,
            inlet_velocity=args.inlet_velocity,
            advection_scheme=args.advection_scheme,
        )
        gamma_richardson = 2.0 * gamma_fit.Gamma - gamma_coarse
        gamma_richardson_rel_error = (
            (gamma_richardson - gamma_analytic) / gamma_analytic
            if gamma_analytic != 0.0
            else np.nan
        )

    # Exact-vs-correlated decay diagnostics.
    beta_abs_error = beta_corr - beta_exact
    beta_rel_error = beta_abs_error / beta_exact if beta_exact != 0.0 else np.nan
    beta_profile_abs_error = gamma_fit.beta_full_profile_fit - beta_exact
    beta_profile_rel_error = (
        beta_profile_abs_error / beta_exact if beta_exact != 0.0 else np.nan
    )
    beta_const_rel_error = (
        (beta_const - beta_exact) / beta_exact if beta_exact != 0.0 else np.nan
    )
    identity_residual_at_exact_beta = (
        corr.chi * beta_exact**2 + args.Pe * beta_exact - k_corr
    )
    identity_scale = max(abs(k_corr), 1.0e-300)

    # Fully developed values sampled from the 2D solution in the Gamma window.
    sh_2d_fit = float(np.nanmean(sh_film[gamma_mask]))
    chi_2d_fit = float(np.nanmean(chi_2d[gamma_mask]))

    def relative_error(
        model: np.ndarray, truth: np.ndarray, mask: np.ndarray
    ) -> tuple[float, float]:
        rel = (model[mask] - truth[mask]) / truth[mask]
        return float(np.mean(np.abs(rel))), float(np.max(np.abs(rel)))

    full_positive = np.isfinite(cm_full) & np.isfinite(cm_corr) & (cm_full > args.cm_min)
    mean_corr_all, max_corr_all = relative_error(cm_corr, cm_full, full_positive)
    mean_scaled_all, max_scaled_all = relative_error(cm_corr_scaled, cm_full, full_positive)
    mean_const_all, max_const_all = relative_error(cm_const, cm_full, full_positive)

    decay_comparison = {
        "beta_exact_eigenvalue": beta_exact,
        "beta_correlated_closure": beta_corr,
        "absolute_error_correlated_vs_exact": beta_abs_error,
        "relative_error_correlated_vs_exact": beta_rel_error,
        "beta_full_profile_fit": gamma_fit.beta_full_profile_fit,
        "relative_error_profile_fit_vs_exact": beta_profile_rel_error,
        "beta_constant_closure": beta_const,
        "relative_error_constant_vs_exact": beta_const_rel_error,
        "identity_residual_using_exact_beta_and_correlated_coefficients": (
            identity_residual_at_exact_beta
        ),
        "normalized_identity_residual": identity_residual_at_exact_beta / identity_scale,
    }

    summary: dict[str, Any] = {
        "inputs": {
            "Pe": args.Pe,
            "Da": args.Da,
            "Lambda": Lambda_eff,
            "Cin": args.Cin,
            "inlet_velocity": args.inlet_velocity,
            "advection_scheme": args.advection_scheme,
            "velocity_profile": "plug",
        },
        "grid": {
            "Nz": nz,
            "Nr": nr_eff,
            "dz": float(z[1] - z[0]),
        },
        "correlation_parameters": asdict(TUBE_PLUG_CORR),
        "correlation_values": asdict(corr),
        "exact_dominant_mode": asdict(exact_mode),
        "decay_rate_comparison": decay_comparison,
        "Gamma_fit": asdict(gamma_fit),
        "Gamma_analytic": {
            "kappa1": kappa1,
            "Gamma_analytic": gamma_analytic,
            "Gamma_fit_vs_analytic_relative_error": gamma_analytic_rel_error,
        },
        "Gamma_high_accuracy": {
            "enabled": high_accuracy,
            "beta2": (float(beta_from_kappa(kappa2, args.Pe)) if high_accuracy else float("nan")),
            "Le_spectral_gap": Le,
            "window_z_min": z_fit_min,
            "window_z_max": z_fit_max,
            "Lambda": Lambda_eff,
            "dz_fine": dz_eff,
            "dz_coarse": dz_coarse_ha,
            "Gamma_fine": gamma_fit.Gamma,
            "Gamma_coarse": gamma_coarse,
            "Gamma_richardson": gamma_richardson,
            "Gamma_richardson_vs_analytic_relative_error": gamma_richardson_rel_error,
        },
        "averaged_correlated_model": {
            "K": k_corr,
            "lambda_minus": lambda_minus_corr,
            "lambda_plus": lambda_plus_corr,
            "inlet_amplitude": amplitude_corr,
        },
        "constant_model": {
            "Sh": args.constant_sh,
            "chi": args.constant_chi,
            "K": k_const,
            "lambda_minus": lambda_minus_const,
            "inlet_amplitude": amplitude_const,
        },
        "axisymmetric_downstream_values": {
            "Sh_film_mean_in_Gamma_window": sh_2d_fit,
            "chi_mean_in_Gamma_window": chi_2d_fit,
        },
        "correlation_errors_against_exact_mode": {
            "Sh_relative_error": (corr.sh - exact_mode.sh) / exact_mode.sh,
            "chi_absolute_error": corr.chi - exact_mode.chi,
            "chi_relative_error": (corr.chi - exact_mode.chi) / exact_mode.chi,
        },
        "concentration_errors": {
            "unscaled_correlated_full_domain_mean_abs_rel": mean_corr_all,
            "unscaled_correlated_full_domain_max_abs_rel": max_corr_all,
            "scaled_correlated_full_domain_mean_abs_rel": mean_scaled_all,
            "scaled_correlated_full_domain_max_abs_rel": max_scaled_all,
            "constant_full_domain_mean_abs_rel": mean_const_all,
            "constant_full_domain_max_abs_rel": max_const_all,
            "unscaled_correlated_Gamma_window_mean_abs_rel": (
                gamma_fit.mean_abs_rel_error_unscaled
            ),
            "scaled_correlated_Gamma_window_mean_abs_rel": (
                gamma_fit.mean_abs_rel_error_scaled
            ),
            "scaled_correlated_Gamma_window_max_abs_rel": (
                gamma_fit.max_abs_rel_error_scaled
            ),
        },
    }

    return CaseResult(
        args=args,
        prefix=prefix,
        nz=nz,
        corr=corr,
        exact_mode=exact_mode,
        beta_exact=beta_exact,
        beta_corr=beta_corr,
        beta_const=beta_const,
        beta_rel_error=beta_rel_error,
        beta_const_rel_error=beta_const_rel_error,
        z=z,
        cm_full=cm_full,
        cm_corr=cm_corr,
        cm_corr_scaled=cm_corr_scaled,
        cm_const=cm_const,
        gamma_fit=gamma_fit,
        gamma_mask=gamma_mask,
        decay_comparison=decay_comparison,
        summary=summary,
        gamma_analytic=gamma_analytic,
        gamma_richardson=gamma_richardson,
    )


def print_case_report_plug(result: CaseResult) -> None:
    """Print the correlation, decay-rate, and Gamma-fit summary for one case."""
    args = result.args
    print("\nPlug-flow tube correlation (chi = 1 exactly, Sh depends on Da only)")
    print(f"  Sh(Da)                        = {result.corr.sh:.10g}")
    print(f"  chi                           = {result.corr.chi:.10g}")
    print("\nDominant decay-rate comparison")
    print(f"  beta_exact (closed form)      = {result.beta_exact:.10g}")
    print(f"  beta_correlated               = {result.beta_corr:.10g}")
    print(f"  relative difference           = {result.beta_rel_error:.6e}")
    print(
        f"  beta fitted from 2D profile   = {result.gamma_fit.beta_full_profile_fit:.10g}"
    )
    print(
        f"  beta_averaged (Sh={args.constant_sh:g}, chi={args.constant_chi:g}) "
        f"= {result.beta_const:.10g}"
    )
    print(f"  relative difference           = {result.beta_const_rel_error:.6e}")
    print("\nDownstream amplitude fit")
    print(f"  Gamma (fitted, full 2D)       = {result.gamma_fit.Gamma:.10g}")
    print(
        f"  fit window                    = "
        f"[{result.gamma_fit.z_min:.6g}, {result.gamma_fit.z_max:.6g}]"
    )
    print(
        f"  scaled mean abs rel error     = "
        f"{result.gamma_fit.mean_abs_rel_error_scaled:.6e}"
    )
    print(
        f"  Gamma (analytic, leading mode) = {result.gamma_analytic:.10g}"
    )
    gamma_analytic_rel_error = (
        result.summary["Gamma_analytic"]["Gamma_fit_vs_analytic_relative_error"]
    )
    print(f"  relative difference (fit vs analytic) = {gamma_analytic_rel_error:.6e}")

    ha = result.summary["Gamma_high_accuracy"]
    if ha["enabled"]:
        print("\nHigh-accuracy Gamma (Le-scaled window + Richardson extrapolation)")
        print(f"  Le = 1/(beta2-beta1)           = {ha['Le_spectral_gap']:.6g}")
        print(f"  fit window                    = [{ha['window_z_min']:.6g}, {ha['window_z_max']:.6g}]")
        print(f"  Lambda                        = {ha['Lambda']:.6g}")
        print(f"  Gamma (dz={ha['dz_fine']:.6g})            = {ha['Gamma_fine']:.10g}")
        print(f"  Gamma (dz={ha['dz_coarse']:.6g})            = {ha['Gamma_coarse']:.10g}")
        print(f"  Gamma (Richardson-extrapolated) = {result.gamma_richardson:.10g}")
        print(
            f"  relative difference (Richardson vs analytic) = "
            f"{ha['Gamma_richardson_vs_analytic_relative_error']:.6e}"
        )


def run_case_plug(args: argparse.Namespace) -> dict[str, Any]:
    """Compute one case, save its outputs, and print its report (CLI path)."""
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    result = compute_case_plug(args)

    profile_columns = {
        "Z": result.z,
        "Cm_full": result.cm_full,
        "Cm_averaged_correlated": result.cm_corr,
        "Cm_averaged_constant": result.cm_const,
        "Cm_rescaled_correlated": result.cm_corr_scaled,
    }
    profiles_txt_path = save_profiles_txt(output_dir, result.prefix, profile_columns)
    parameters_txt_path = save_parameters_txt(output_dir, result.prefix, result.summary)

    png_path: Path | None = None
    pdf_path: Path | None = None
    if not args.no_plots:
        png_path, pdf_path = save_concentration_plot(
            output_dir=output_dir,
            prefix=result.prefix,
            z=result.z,
            cm_full=result.cm_full,
            cm_corr=result.cm_corr,
            cm_corr_scaled=result.cm_corr_scaled,
            cm_const=result.cm_const,
            constant_sh=args.constant_sh,
            constant_chi=args.constant_chi,
            y_min=args.plot_ymin,
            y_max=args.plot_ymax,
            dpi=args.plot_dpi,
            Pe=args.Pe,
            Da=args.Da,
        )

    print_case_report_plug(result)
    print("\nSaved files")
    print(f"  {profiles_txt_path}")
    print(f"  {parameters_txt_path}")
    if png_path is not None:
        print(f"  {png_path}")
        print(f"  {pdf_path}")

    return result.summary


# =========================================================
# CLI
# =========================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Solve the circular-tube extended-Graetz problem for plug flow, "
            "evaluate the fitted Sh(Da) correlation, fit Gamma, and save "
            "decay diagnostics."
        )
    )
    parser.add_argument("--Pe", type=float, default=10.0, help="Pe = u_bar R / D")
    parser.add_argument("--Da", type=float, default=100.0, help="Da = k R / D")
    parser.add_argument("--Lambda", type=float, default=5.0, help="L/R")
    parser.add_argument("--Cin", type=float, default=1.0, help="inlet concentration")
    parser.add_argument("--Nr", type=int, default=50, help="number of radial cells")
    parser.add_argument("--nz", type=int, default=None, help="number of axial nodes")
    parser.add_argument(
        "--dz",
        type=float,
        default=0.025,
        help="target axial spacing if --nz is omitted",
    )
    parser.add_argument(
        "--inlet-velocity",
        choices=("local", "mean"),
        default="local",
        help="velocity used in the Danckwerts inlet condition (equivalent for plug flow)",
    )
    parser.add_argument(
        "--advection-scheme",
        choices=("upwind1", "upwind2"),
        default="upwind2",
        help="axial advection discretization in the 2D solver",
    )
    parser.add_argument(
        "--gamma-zmin-frac",
        type=float,
        default=0.40,
        help="lower Gamma-fit bound as a fraction of Lambda",
    )
    parser.add_argument(
        "--gamma-zmax-frac",
        type=float,
        default=0.80,
        help="upper Gamma-fit bound as a fraction of Lambda",
    )
    parser.add_argument(
        "--cm-min",
        type=float,
        default=1.0e-10,
        help="minimum concentration retained in downstream fits",
    )
    parser.add_argument(
        "--gamma-high-accuracy",
        action="store_true",
        help=(
            "Verify Gamma = gamma_analytic to <=0.03%% instead of using "
            "--Lambda/--dz/--gamma-zmin-frac/--gamma-zmax-frac: place the "
            "Gamma-fit window at [6, 8] local spectral-gap lengths "
            "Le=1/(beta2-beta1) downstream, size dz to resolve beta_1, and "
            "Richardson-extrapolate two dz levels. Runs two 2D solves on "
            "grids roughly 10x finer than the defaults, so it is slower."
        ),
    )
    parser.add_argument(
        "--constant-sh",
        type=float,
        default=4.0,
        help="Sherwood number in the baseline constant closure",
    )
    parser.add_argument(
        "--constant-chi",
        type=float,
        default=1.0,
        help="averaging factor in the baseline constant closure",
    )
    parser.add_argument("--output-dir", default="tube_solver_plug_output")
    parser.add_argument("--prefix", default=None)
    parser.add_argument(
        "--plot-ymin",
        type=float,
        default=0.1,
        help="lower logarithmic y limit for the concentration plot",
    )
    parser.add_argument(
        "--plot-ymax",
        type=float,
        default=1.0,
        help="upper logarithmic y limit for the concentration plot",
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=128,
        help="PNG resolution; 128 gives a 1536 x 1024 image",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    run_case_plug(args)


if __name__ == "__main__":
    main()
