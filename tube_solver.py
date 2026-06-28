#!/usr/bin/env python3
"""Axisymmetric tube solver and reduced-model comparison.

The script solves the steady extended-Graetz problem for Poiseuille flow in a
circular tube with a Robin wall condition, evaluates the updated manuscript
correlations for Sh(Pe, Da) and chi(Pe, Da), and compares the exact dominant
axial decay rate with the decay rate predicted by the correlated closure.

It also fits the downstream multiplicative amplitude factor Gamma defined by

    C_m,full(Z) ~= Gamma * C_m,corr(Z)

in a user-selectable fully developed window. Gamma is obtained as the
geometric mean of C_m,full / C_m,corr in that window, i.e. by least squares in
log concentration. Each run writes four data files (profiles in CSV and
space-delimited txt, parameters/diagnostics in JSON and key=value txt)
and an optional concentration plot (PNG).

Default definitions:
    Pe = u_bar R / D
    Da = k R / D
    Z  = z / R
    Sh = 2 R j_w / [D (C_m - C_w)]

Example:
    python tube_solver.py --Pe 10 --Da 100 --Lambda 5 --output-dir results
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar


# =========================================================
# Plot style
# =========================================================
def configure_plot_style() -> None:
    """Configure the large serif style used by the manuscript figure."""
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 22,
            "axes.labelsize": 34,
            "axes.titlesize": 36,
            "xtick.labelsize": 27,
            "ytick.labelsize": 27,
            "legend.fontsize": 18,
            "axes.linewidth": 2.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 8,
            "ytick.major.size": 8,
            "xtick.minor.size": 5,
            "ytick.minor.size": 5,
            "xtick.major.width": 2.0,
            "ytick.major.width": 2.0,
            "xtick.minor.width": 1.4,
            "ytick.minor.width": 1.4,
        }
    )


# =========================================================
# Updated tube/Poiseuille correlations from Appendix B
# =========================================================
@dataclass(frozen=True)
class TubeCorrelationParameters:
    """Rounded coefficients appearing in the final manuscript."""

    # Low-Pe branch, Sh^0(Da)
    sh_0_da0: float = 6.0
    sh_0_dainf: float = 4.1807
    da_c_0_sh: float = 2.6886

    # High-Pe branch, Sh^inf(Da)
    sh_inf_da0: float = 4.3636  # 48/11 rounded as in Appendix B
    sh_inf_dainf: float = 3.6568
    da_c_inf_sh: float = 1.8848

    # Pe crossover for Sh
    pe_c_da0: float = 0.0108
    pe_c_dainf: float = 0.9728
    da_c_pe_sh: float = 0.5687
    sh_pe_exponent: float = 4.0 / 3.0
    pe_c_da_exponent: float = 2.0 / 3.0

    # Averaging factor chi = C_a/C_m
    chi_0_dainf: float = 0.7229
    da_c_0_chi: float = 2.8167
    chi_inf_dainf: float = 0.7095
    da_c_inf_chi: float = 2.1190
    pe_c_chi: float = 0.6874


TUBE_CORR = TubeCorrelationParameters()


@dataclass(frozen=True)
class TubeCorrelationResult:
    sh_low_pe: float
    sh_high_pe: float
    pe_c_sh: float
    sh: float
    chi_low_pe: float
    chi_high_pe: float
    chi: float


def _positive_inputs(Pe: float, Da: float) -> None:
    if Pe <= 0.0:
        raise ValueError("Pe must be positive.")
    if Da < 0.0:
        raise ValueError("Da must be non-negative.")


def validate_correlation_window(Pe: float, Da: float) -> None:
    """Warn when the manuscript correlation is being extrapolated."""
    if not (1.0e-3 <= Pe <= 1.0e3 and 1.0e-3 <= Da <= 1.0e3):
        warnings.warn(
            "The Sh and chi correlations were fitted on "
            "1e-3 <= Pe, Da <= 1e3; this case extrapolates beyond that window.",
            RuntimeWarning,
            stacklevel=2,
        )

    if Pe < 0.1 and Da < 0.01 * Pe**2:
        warnings.warn(
            "This case lies deep in the singular weak-exchange corner "
            "Da << Pe^2 << 1, which is not resolved by the compact Sh fit.",
            RuntimeWarning,
            stacklevel=2,
        )


def sh_low_pe_tube(Da: float, p: TubeCorrelationParameters = TUBE_CORR) -> float:
    return p.sh_0_dainf + (p.sh_0_da0 - p.sh_0_dainf) / (
        1.0 + Da / p.da_c_0_sh
    )


def sh_high_pe_tube(Da: float, p: TubeCorrelationParameters = TUBE_CORR) -> float:
    return p.sh_inf_dainf + (p.sh_inf_da0 - p.sh_inf_dainf) / (
        1.0 + Da / p.da_c_inf_sh
    )


def pe_c_sh_tube(Da: float, p: TubeCorrelationParameters = TUBE_CORR) -> float:
    return p.pe_c_dainf - (p.pe_c_dainf - p.pe_c_da0) / (
        1.0 + (Da / p.da_c_pe_sh) ** p.pe_c_da_exponent
    )


def sh_fit_tube(Pe: float, Da: float, p: TubeCorrelationParameters = TUBE_CORR) -> float:
    _positive_inputs(Pe, Da)
    sh_0 = sh_low_pe_tube(Da, p)
    sh_inf = sh_high_pe_tube(Da, p)
    pe_c = pe_c_sh_tube(Da, p)
    return sh_inf + (sh_0 - sh_inf) / (
        1.0 + (Pe / pe_c) ** p.sh_pe_exponent
    )


def chi_low_pe_tube(Da: float, p: TubeCorrelationParameters = TUBE_CORR) -> float:
    return p.chi_0_dainf + (1.0 - p.chi_0_dainf) / (
        1.0 + Da / p.da_c_0_chi
    )


def chi_high_pe_tube(Da: float, p: TubeCorrelationParameters = TUBE_CORR) -> float:
    return p.chi_inf_dainf + (1.0 - p.chi_inf_dainf) / (
        1.0 + Da / p.da_c_inf_chi
    )


def chi_fit_tube(Pe: float, Da: float, p: TubeCorrelationParameters = TUBE_CORR) -> float:
    _positive_inputs(Pe, Da)
    chi_0 = chi_low_pe_tube(Da, p)
    chi_inf = chi_high_pe_tube(Da, p)
    return chi_inf + (chi_0 - chi_inf) / (1.0 + Pe / p.pe_c_chi)


def evaluate_tube_correlations(
    Pe: float,
    Da: float,
    p: TubeCorrelationParameters = TUBE_CORR,
) -> TubeCorrelationResult:
    validate_correlation_window(Pe, Da)
    sh_0 = sh_low_pe_tube(Da, p)
    sh_inf = sh_high_pe_tube(Da, p)
    pe_c = pe_c_sh_tube(Da, p)
    sh = sh_inf + (sh_0 - sh_inf) / (
        1.0 + (Pe / pe_c) ** p.sh_pe_exponent
    )
    chi_0 = chi_low_pe_tube(Da, p)
    chi_inf = chi_high_pe_tube(Da, p)
    chi = chi_inf + (chi_0 - chi_inf) / (1.0 + Pe / p.pe_c_chi)
    return TubeCorrelationResult(
        sh_low_pe=sh_0,
        sh_high_pe=sh_inf,
        pe_c_sh=pe_c,
        sh=sh,
        chi_low_pe=chi_0,
        chi_high_pe=chi_inf,
        chi=chi,
    )


# =========================================================
# 2D axisymmetric steady ADR solver
# =========================================================
def solve_axisym_ADR_steady(
    Nz: int = 800,
    Nr: int = 40,
    Lambda: float = 20.0,
    Pe: float = 1.0,
    Da: float = 1.0,
    Cin: float = 1.0,
    inlet_velocity: str = "local",
    advection_scheme: str = "upwind2",
    outlet_bc: str = "mode",
    outlet_lambda: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve the steady axisymmetric advection-diffusion problem.

    Dimensionless PDE in Z=z/R and rho=r/R:
        (1/Pe) C_ZZ + (1/Pe) rho^{-1}(rho C_rho)_rho - u_hat C_Z = 0.

    The Robin wall condition is -C_rho(1,Z)=Da C(1,Z).  The default outlet
    condition imposes the exact dominant modal slope, C_Z=lambda_minus C.
    """
    _positive_inputs(Pe, Da)
    if Nz < 3 or Nr < 2:
        raise ValueError("Require Nz >= 3 and Nr >= 2.")
    if Lambda <= 0.0:
        raise ValueError("Lambda must be positive.")

    invPe = 1.0 / Pe

    z = np.linspace(0.0, Lambda, Nz)
    dz = z[1] - z[0]
    dz2 = dz * dz

    # Cell-centred radial finite-volume grid.
    rf = np.linspace(0.0, 1.0, Nr + 1)
    dr = rf[1] - rf[0]
    r = 0.5 * (rf[:-1] + rf[1:])

    u = 2.0 * (1.0 - r**2)  # mean-normalized Poiseuille profile

    inlet_velocity_lc = inlet_velocity.lower()
    if inlet_velocity_lc == "local":
        U_in = u.copy()
    elif inlet_velocity_lc == "mean":
        U_in = np.ones_like(u)
    else:
        raise ValueError("inlet_velocity must be 'local' or 'mean'.")

    advection_scheme_lc = advection_scheme.lower()
    if advection_scheme_lc not in {"upwind1", "upwind2"}:
        raise ValueError("advection_scheme must be 'upwind1' or 'upwind2'.")

    def idx(i: int, j: int) -> int:
        return j * Nr + i

    n_unknowns = Nr * Nz
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rhs = np.zeros(n_unknowns)

    # Radial operator coefficients.
    a_w = np.zeros(Nr)
    a_p = np.zeros(Nr)
    a_e = np.zeros(Nr)

    for i in range(Nr):
        rc = r[i]
        rw = rf[i]
        re = rf[i + 1]

        if i == 0:
            a_w[i] = 0.0
            a_e[i] = invPe * re / (rc * dr * dr)
            a_p[i] = -a_e[i]
        elif i == Nr - 1:
            a_w[i] = invPe * rw / (rc * dr * dr)
            alpha = Da * dr / 2.0
            wall_sink = invPe * (-2.0 * alpha / (1.0 + alpha)) / (
                rc * dr * dr
            )
            a_e[i] = 0.0
            a_p[i] = -a_w[i] + wall_sink
        else:
            a_w[i] = invPe * rw / (rc * dr * dr)
            a_e[i] = invPe * re / (rc * dr * dr)
            a_p[i] = -(a_w[i] + a_e[i])

    for j in range(Nz):
        for i in range(Nr):
            row = idx(i, j)

            if j == 0:
                # Danckwerts inlet: -(1/Pe) C_Z + U_in C = U_in C_in.
                rows.extend([row, row])
                cols.extend([idx(i, j), idx(i, j + 1)])
                vals.extend([invPe / dz + U_in[i], -invPe / dz])
                rhs[row] = U_in[i] * Cin
                continue

            if j == Nz - 1:
                outlet_bc_lc = outlet_bc.lower()
                if outlet_bc_lc == "neumann":
                    rows.extend([row, row])
                    cols.extend([idx(i, j), idx(i, j - 1)])
                    vals.extend([1.0, -1.0])
                elif outlet_bc_lc == "mode":
                    if outlet_lambda is None:
                        raise ValueError(
                            "outlet_lambda is required when outlet_bc='mode'."
                        )
                    rows.extend([row, row])
                    cols.extend([idx(i, j), idx(i, j - 1)])
                    vals.extend([1.0 / dz - outlet_lambda, -1.0 / dz])
                else:
                    raise ValueError("outlet_bc must be 'mode' or 'neumann'.")
                continue

            # Axial diffusion.
            rows.extend([row, row, row])
            cols.extend([idx(i, j - 1), idx(i, j), idx(i, j + 1)])
            vals.extend([invPe / dz2, -2.0 * invPe / dz2, invPe / dz2])

            # Positive-flow upwind advection.  Second-order backward
            # differencing strongly reduces the artificial axial diffusion in
            # the fitted downstream decay rate; the first interior plane uses
            # first-order upwinding because j-2 is unavailable.
            if advection_scheme_lc == "upwind2" and j >= 2:
                rows.extend([row, row, row])
                cols.extend([idx(i, j), idx(i, j - 1), idx(i, j - 2)])
                vals.extend(
                    [
                        -3.0 * u[i] / (2.0 * dz),
                        2.0 * u[i] / dz,
                        -u[i] / (2.0 * dz),
                    ]
                )
            else:
                rows.extend([row, row])
                cols.extend([idx(i, j), idx(i, j - 1)])
                vals.extend([-u[i] / dz, u[i] / dz])

            # Radial diffusion.
            if i > 0:
                rows.append(row)
                cols.append(idx(i - 1, j))
                vals.append(a_w[i])

            rows.append(row)
            cols.append(idx(i, j))
            vals.append(a_p[i])

            if i < Nr - 1:
                rows.append(row)
                cols.append(idx(i + 1, j))
                vals.append(a_e[i])

    matrix = sp.csr_matrix((vals, (rows, cols)), shape=(n_unknowns, n_unknowns))
    c_flat = spla.spsolve(matrix, rhs)
    c = c_flat.reshape(Nz, Nr).T

    return z, r, c, u


# =========================================================
# Averaged quantities from the 2D solution
# =========================================================
def extract_sh_numbers(
    z: np.ndarray,
    r: np.ndarray,
    C: np.ndarray,
    u: np.ndarray,
    Da: float,
) -> tuple[np.ndarray, ...]:
    """Return C_m, C_a, chi, C_w, wall gradient, and two Sh definitions."""
    del z  # included in the signature for compatibility/readability
    dr = r[1] - r[0]

    numerator = 2.0 * np.sum(u[:, None] * C * r[:, None] * dr, axis=0)
    denominator = 2.0 * np.sum(u * r * dr)
    cm = numerator / denominator

    c_area = 2.0 * np.sum(C * r[:, None] * dr, axis=0)
    chi = c_area / (cm + 1.0e-300)

    alpha = Da * dr / 2.0
    c_wall = C[-1, :] / (1.0 + alpha)
    dcdr_wall = (c_wall - C[-1, :]) / (dr / 2.0)

    # Diameter-based Sherwood numbers, d_h=2R.
    sh_eff = -2.0 * dcdr_wall / (cm + 1.0e-300)
    sh_film = np.full_like(cm, np.nan, dtype=float)
    driving = cm - c_wall
    valid = np.abs(driving) > 1.0e-12
    sh_film[valid] = -2.0 * dcdr_wall[valid] / driving[valid]

    return cm, c_area, chi, c_wall, dcdr_wall, sh_eff, sh_film


# =========================================================
# Averaged 1D model
# =========================================================
def averaged_decay_lambdas(
    Pe: float,
    Da: float,
    Sh: float,
    chi: float,
) -> tuple[float, float, float]:
    """Return K, lambda_minus, and lambda_plus for the averaged equation.

    In Z=z/R, the diameter-based Sh gives
        chi C'' - Pe C' - K C = 0,
        K = 2 Da Sh / (2 Da + Sh).
    The physical downstream decay rate is beta=-lambda_minus > 0.
    """
    _positive_inputs(Pe, Da)
    if Sh <= 0.0 or chi <= 0.0:
        raise ValueError("Sh and chi must be positive.")

    k_eff = 2.0 * Da * Sh / (2.0 * Da + Sh)
    if abs(k_eff) < 1.0e-300:
        return k_eff, 0.0, Pe / chi

    discriminant = math.sqrt(Pe**2 + 4.0 * chi * k_eff)
    lambda_plus = (Pe + discriminant) / (2.0 * chi)
    lambda_minus = (Pe - discriminant) / (2.0 * chi)
    return k_eff, lambda_minus, lambda_plus


def solve_averaged_model_semi_infinite(
    z: np.ndarray,
    Pe: float,
    Da: float,
    Sh: float,
    chi: float,
    Cin: float = 1.0,
) -> tuple[np.ndarray, ...]:
    """Solve the semi-infinite one-mode averaged model with Danckwerts inlet."""
    z = np.asarray(z, dtype=float)
    k_eff, lambda_minus, lambda_plus = averaged_decay_lambdas(Pe, Da, Sh, chi)

    if abs(k_eff) < 1.0e-300:
        cm = Cin * np.ones_like(z)
        c_wall = Sh / (2.0 * Da + Sh) * cm
        c_area = chi * cm
        return cm, c_area, c_wall, k_eff, 0.0, lambda_plus, Cin, 0.0

    denominator = 1.0 - (chi / Pe) * lambda_minus
    amplitude = Cin / denominator
    cm = amplitude * np.exp(lambda_minus * z)
    c_wall = Sh / (2.0 * Da + Sh) * cm
    c_area = chi * cm

    return (
        cm,
        c_area,
        c_wall,
        k_eff,
        lambda_minus,
        lambda_plus,
        amplitude,
        0.0,
    )


# =========================================================
# Exact dominant eigenvalue and eigenfunction
# =========================================================
def velocity_profile_tube(rho: np.ndarray | float) -> np.ndarray:
    rho_arr = np.asarray(rho, dtype=float)
    return 2.0 * (1.0 - rho_arr**2)


def tube_poiseuille_eigen_ode(
    rho: float,
    y: np.ndarray,
    beta: float,
    Pe: float,
) -> list[float]:
    phi, dphi = y
    coefficient = beta**2 + Pe * beta * float(velocity_profile_tube(rho))
    ddphi = -(dphi / rho) - coefficient * phi
    return [dphi, ddphi]


def tube_eigen_initial_conditions(beta: float, Pe: float, rho0: float) -> tuple[float, float]:
    a0 = beta**2 + 2.0 * Pe * beta
    phi0 = 1.0 - 0.25 * a0 * rho0**2
    dphi0 = -0.5 * a0 * rho0
    return phi0, dphi0


def tube_wall_residual(
    beta: float,
    Pe: float,
    Da: float,
    rho0: float = 1.0e-7,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-12,
) -> float:
    if beta <= 0.0:
        return np.nan

    phi0, dphi0 = tube_eigen_initial_conditions(beta, Pe, rho0)
    solution = solve_ivp(
        tube_poiseuille_eigen_ode,
        t_span=(rho0, 1.0),
        y0=[phi0, dphi0],
        args=(beta, Pe),
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"Eigenfunction ODE solve failed: {solution.message}")

    phi_wall = solution.y[0, -1]
    dphi_wall = solution.y[1, -1]
    return -dphi_wall - Da * phi_wall


def tube_eigenfunction(
    beta: float,
    Pe: float,
    n_points: int = 1000,
    rho0: float = 1.0e-7,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi0, dphi0 = tube_eigen_initial_conditions(beta, Pe, rho0)
    rho_eval = np.linspace(rho0, 1.0, n_points)
    solution = solve_ivp(
        tube_poiseuille_eigen_ode,
        t_span=(rho0, 1.0),
        y0=[phi0, dphi0],
        t_eval=rho_eval,
        args=(beta, Pe),
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"Eigenfunction ODE solve failed: {solution.message}")

    rho = np.concatenate(([0.0], solution.t))
    phi = np.concatenate(([1.0], solution.y[0]))
    dphi = np.concatenate(([0.0], solution.y[1]))
    return rho, phi, dphi


def tube_mode_is_physical(beta: float, Pe: float, Da: float, rho0: float = 1.0e-7) -> bool:
    rho, phi, _ = tube_eigenfunction(
        beta,
        Pe,
        n_points=500,
        rho0=rho0,
        rtol=1.0e-9,
        atol=1.0e-11,
    )
    if phi[-1] <= 0.0 or np.min(phi) <= 0.0:
        return False

    u = velocity_profile_tube(rho)
    cm_over_cw = 2.0 * np.trapezoid(rho * u * phi, rho) / phi[-1]
    return bool(cm_over_cw > 1.0)


def find_beta1_tube_poiseuille(
    Pe: float,
    Da: float,
    beta_start: float = 1.0e-12,
    beta_max: float = 50.0,
    growth: float = 1.12,
    rho0: float = 1.0e-7,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-12,
    verbose: bool = False,
) -> float:
    """Find the first positive physical decay rate beta_1."""
    _positive_inputs(Pe, Da)
    if Da == 0.0:
        return 0.0

    def residual(beta: float) -> float:
        return tube_wall_residual(
            beta,
            Pe,
            Da,
            rho0=rho0,
            rtol=rtol,
            atol=atol,
        )

    a = beta_start
    fa = residual(a)

    while a < beta_max:
        b = min(a * growth + 1.0e-15, beta_max)
        fb = residual(b)

        if verbose:
            print(f"scan beta [{a:.5e}, {b:.5e}]: R=[{fa:.5e}, {fb:.5e}]")

        if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0.0:
            root = root_scalar(
                residual,
                bracket=(a, b),
                method="brentq",
                xtol=1.0e-12,
                rtol=1.0e-12,
            )
            if root.converged and tube_mode_is_physical(root.root, Pe, Da, rho0=rho0):
                return float(root.root)

        a, fa = b, fb

    raise RuntimeError(
        "Could not find the physical beta_1. Increase beta_max or reduce growth."
    )


@dataclass(frozen=True)
class ExactModeProperties:
    beta: float
    sh: float
    chi: float
    cm_shape: float
    c_area_shape: float
    c_wall_shape: float
    wall_flux_shape: float
    identity_residual: float


def exact_mode_properties(Pe: float, Da: float, beta: float) -> ExactModeProperties:
    """Compute exact fully developed Sh and chi from the dominant eigenfunction."""
    rho, phi, dphi = tube_eigenfunction(beta, Pe, n_points=1600)
    u = velocity_profile_tube(rho)

    cm_shape = 2.0 * np.trapezoid(rho * u * phi, rho)
    c_area_shape = 2.0 * np.trapezoid(rho * phi, rho)
    c_wall_shape = float(phi[-1])
    wall_flux_shape = float(-dphi[-1])

    sh = 2.0 * wall_flux_shape / (cm_shape - c_wall_shape)
    chi = c_area_shape / cm_shape
    k_eff = 2.0 * Da * sh / (2.0 * Da + sh)
    identity_residual = chi * beta**2 + Pe * beta - k_eff

    return ExactModeProperties(
        beta=beta,
        sh=float(sh),
        chi=float(chi),
        cm_shape=float(cm_shape),
        c_area_shape=float(c_area_shape),
        c_wall_shape=c_wall_shape,
        wall_flux_shape=wall_flux_shape,
        identity_residual=float(identity_residual),
    )


# =========================================================
# Downstream fitting and diagnostics
# =========================================================
def local_decay_rate(z: np.ndarray, concentration: np.ndarray) -> np.ndarray:
    log_c = np.log(np.maximum(concentration, 1.0e-300))
    return -np.gradient(log_c, z, edge_order=2)


@dataclass(frozen=True)
class DownstreamFit:
    Gamma: float
    z_min: float
    z_max: float
    n_points: int
    beta_full_profile_fit: float
    amplitude_full_profile_fit: float
    beta_corr_profile_fit: float
    gamma_log_std: float
    log_rmse_unscaled: float
    log_rmse_scaled: float
    mean_abs_rel_error_unscaled: float
    mean_abs_rel_error_scaled: float
    max_abs_rel_error_unscaled: float
    max_abs_rel_error_scaled: float


def fit_downstream_gamma(
    z: np.ndarray,
    cm_full: np.ndarray,
    cm_corr: np.ndarray,
    z_min: float,
    z_max: float,
    cm_min: float = 1.0e-10,
) -> tuple[DownstreamFit, np.ndarray]:
    """Fit Gamma in log space and estimate the downstream slopes.

    Gamma = exp(mean(log(C_full/C_corr))) over the selected window.  This is
    the constant multiplier that minimizes squared log-concentration error.
    """
    if not (0.0 <= z_min < z_max <= float(z[-1]) + 1.0e-12):
        raise ValueError("Require 0 <= z_min < z_max <= Lambda.")

    mask = (
        np.isfinite(cm_full)
        & np.isfinite(cm_corr)
        & (cm_full > cm_min)
        & (cm_corr > cm_min)
        & (z >= z_min)
        & (z <= z_max)
    )
    if np.count_nonzero(mask) < 5:
        raise RuntimeError(
            "Too few valid points in the Gamma fitting window. "
            "Increase Lambda, relax cm_min, or change the fit window."
        )

    z_fit = z[mask]
    log_full = np.log(cm_full[mask])
    log_corr = np.log(cm_corr[mask])
    log_ratio = log_full - log_corr
    log_gamma = float(np.mean(log_ratio))
    gamma = math.exp(log_gamma)

    slope_full, intercept_full = np.polyfit(z_fit, log_full, deg=1)
    slope_corr, _ = np.polyfit(z_fit, log_corr, deg=1)

    scaled = gamma * cm_corr[mask]
    rel_unscaled = (cm_corr[mask] - cm_full[mask]) / cm_full[mask]
    rel_scaled = (scaled - cm_full[mask]) / cm_full[mask]

    fit = DownstreamFit(
        Gamma=gamma,
        z_min=float(z_min),
        z_max=float(z_max),
        n_points=int(np.count_nonzero(mask)),
        beta_full_profile_fit=float(-slope_full),
        amplitude_full_profile_fit=float(math.exp(intercept_full)),
        beta_corr_profile_fit=float(-slope_corr),
        gamma_log_std=float(np.std(log_ratio)),
        log_rmse_unscaled=float(np.sqrt(np.mean((log_corr - log_full) ** 2))),
        log_rmse_scaled=float(
            np.sqrt(np.mean((np.log(scaled) - log_full) ** 2))
        ),
        mean_abs_rel_error_unscaled=float(np.mean(np.abs(rel_unscaled))),
        mean_abs_rel_error_scaled=float(np.mean(np.abs(rel_scaled))),
        max_abs_rel_error_unscaled=float(np.max(np.abs(rel_unscaled))),
        max_abs_rel_error_scaled=float(np.max(np.abs(rel_scaled))),
    )
    return fit, mask


# =========================================================
# Saving helpers
# =========================================================
def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def save_parameters_errors(
    output_dir: Path,
    prefix: str,
    summary: dict[str, Any],
) -> Path:
    """Save all scalar parameters, fitted quantities, and errors in one JSON."""
    path = output_dir / f"{prefix}_parameters_errors.json"
    safe_summary = _json_safe(summary)
    path.write_text(json.dumps(safe_summary, indent=2) + "\n", encoding="utf-8")
    return path


def save_profiles(
    output_dir: Path,
    prefix: str,
    columns: dict[str, np.ndarray],
) -> Path:
    """Save all plotted axial concentration profiles in one CSV file."""
    path = output_dir / f"{prefix}_profiles.csv"
    names = list(columns.keys())
    arrays = [np.asarray(columns[name]) for name in names]
    n = len(arrays[0])
    if any(len(array) != n for array in arrays):
        raise ValueError("All profile arrays must have the same length.")

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        for row in zip(*arrays):
            writer.writerow([f"{float(value):.16g}" for value in row])
    return path


def save_profiles_txt(
    output_dir: Path,
    prefix: str,
    columns: dict[str, np.ndarray],
) -> Path:
    """Save all plotted axial concentration profiles as a space-delimited txt file."""
    path = output_dir / f"{prefix}_profiles.txt"
    names = list(columns.keys())
    arrays = [np.asarray(columns[name], dtype=float) for name in names]
    n = len(arrays[0])
    if any(len(array) != n for array in arrays):
        raise ValueError("All profile arrays must have the same length.")

    data = np.column_stack(arrays)
    header = "  ".join(f"{name:>24s}" for name in names)
    np.savetxt(path, data, fmt="%.16e", header=header, comments="# ")
    return path


def _flatten_dict(d: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    for key, value in d.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            items.extend(_flatten_dict(value, full_key))
        else:
            items.append((full_key, value))
    return items


def save_parameters_txt(
    output_dir: Path,
    prefix: str,
    summary: dict[str, Any],
) -> Path:
    """Save all scalar parameters, fitted quantities, and errors as a txt file."""
    path = output_dir / f"{prefix}_parameters.txt"
    safe_summary = _json_safe(summary)
    lines: list[str] = [f"# tube_solver parameters and diagnostics — {prefix}", ""]
    for key, value in _flatten_dict(safe_summary):
        if isinstance(value, list):
            lines.append(f"{key} = {value}")
        elif isinstance(value, float):
            lines.append(f"{key} = {value:.16g}")
        else:
            lines.append(f"{key} = {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# =========================================================
# Plotting
# =========================================================
def _reference_log_tick_label(value: float, _position: int) -> str:
    """Format the selected log ticks as in the reference manuscript figure."""
    if np.isclose(value, 1.0):
        return r"$10^{0}$"
    if np.isclose(value, 0.1):
        return r"$10^{-1}$"
    coefficient = value / 0.1
    if np.isclose(coefficient, round(coefficient)):
        coefficient_text = str(int(round(coefficient)))
    else:
        coefficient_text = f"{coefficient:g}"
    return rf"${coefficient_text}\times 10^{{-1}}$"


def save_concentration_plot(
    output_dir: Path,
    prefix: str,
    z: np.ndarray,
    cm_full: np.ndarray,
    cm_corr: np.ndarray,
    cm_corr_scaled: np.ndarray,
    cm_const: np.ndarray,
    constant_sh: float,
    constant_chi: float,
    y_min: float,
    y_max: float,
    dpi: int,
) -> Path:
    """Save the single publication-style concentration comparison plot."""
    configure_plot_style()

    # 12 x 8 inches at 128 dpi reproduces the 1536 x 1024 reference aspect.
    fig, ax = plt.subplots(figsize=(12.0, 8.0), dpi=dpi)
    fig.subplots_adjust(left=0.16, right=0.975, bottom=0.185, top=0.92)

    gray = "#7f7f7f"
    red = "#d62728"
    orange = "#ff9900"

    ax.semilogy(
        z,
        cm_full,
        color=gray,
        lw=5.5,
        ls="-",
        label="full",
        zorder=2,
    )
    ax.semilogy(
        z,
        cm_corr,
        color=red,
        lw=5.0,
        ls=(0, (4.0, 2.0)),
        label=r"averaged, correlated $\mathrm{Sh},\,\chi$",
        zorder=3,
    )
    ax.semilogy(
        z,
        cm_const,
        color=orange,
        lw=5.0,
        ls=":",
        label=(
            rf"averaged, $\mathrm{{Sh}}={constant_sh:.1f},"
            rf"\ \chi={constant_chi:.1f}$"
        ),
        zorder=3,
    )
    ax.semilogy(
        z,
        cm_corr_scaled,
        color=red,
        lw=5.0,
        ls=":",
        label=r"rescaled, correlated $\mathrm{Sh},\,\chi$",
        zorder=4,
    )

    ax.set_xlim(float(z[0]), float(z[-1]))
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"$Z$", labelpad=10)
    ax.set_ylabel(r"$C_{\mathrm{m}}$", labelpad=0)
    ax.set_title("Concentration", pad=8)

    # Match the labeled logarithmic ticks visible in the reference figure.
    labeled_ticks = [0.1, 0.2, 0.3, 0.4, 0.6, 1.0]
    visible_ticks = [tick for tick in labeled_ticks if y_min <= tick <= y_max]
    if visible_ticks:
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(visible_ticks))
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(_reference_log_tick_label)
        )
    ax.yaxis.set_minor_locator(
        matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)
    )
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax.grid(True, which="major", color="#bfbfbf", alpha=0.38, linewidth=1.3)
    ax.grid(True, which="minor", color="#cfcfcf", alpha=0.30, linewidth=1.0)
    ax.legend(
        loc="upper right",
        frameon=False,
        handlelength=2.2,
        handletextpad=0.75,
        borderaxespad=0.55,
        labelspacing=0.55,
    )

    path = output_dir / f"{prefix}.png"
    fig.savefig(path, dpi=dpi, facecolor="white")
    path = output_dir / f"{prefix}.pdf"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


# =========================================================
# Case runner
# =========================================================
def run_case(args: argparse.Namespace) -> dict[str, Any]:
    _positive_inputs(args.Pe, args.Da)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.nz is None:
        nz = int(math.ceil(args.Lambda / args.dz)) + 1
    else:
        nz = args.nz

    if nz < 3:
        raise ValueError("The axial grid must contain at least three points.")

    prefix = args.prefix or f"tube_Pe{args.Pe:g}_Da{args.Da:g}"
    prefix = prefix.replace("/", "_").replace(" ", "_")

    corr = evaluate_tube_correlations(args.Pe, args.Da)

    beta_exact = find_beta1_tube_poiseuille(
        Pe=args.Pe,
        Da=args.Da,
        beta_start=args.beta_start,
        beta_max=args.beta_max,
        growth=args.beta_growth,
        verbose=args.verbose_eigen,
    )
    exact_mode = exact_mode_properties(args.Pe, args.Da, beta_exact)

    z, r, C, u = solve_axisym_ADR_steady(
        Nz=nz,
        Nr=args.Nr,
        Lambda=args.Lambda,
        Pe=args.Pe,
        Da=args.Da,
        Cin=args.Cin,
        inlet_velocity=args.inlet_velocity,
        advection_scheme=args.advection_scheme,
        outlet_bc="mode",
        outlet_lambda=-beta_exact,
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

    z_fit_min = args.gamma_zmin_frac * args.Lambda
    z_fit_max = args.gamma_zmax_frac * args.Lambda
    gamma_fit, gamma_mask = fit_downstream_gamma(
        z,
        cm_full,
        cm_corr,
        z_min=z_fit_min,
        z_max=z_fit_max,
        cm_min=args.cm_min,
    )
    cm_corr_scaled = gamma_fit.Gamma * cm_corr

    beta_full_local = local_decay_rate(z, cm_full)
    beta_corr_local = local_decay_rate(z, cm_corr)
    beta_const_local = local_decay_rate(z, cm_const)

    # Exact-vs-correlated decay diagnostics.
    beta_abs_error = beta_corr - beta_exact
    beta_rel_error = beta_abs_error / beta_exact if beta_exact != 0.0 else np.nan
    beta_profile_abs_error = gamma_fit.beta_full_profile_fit - beta_exact
    beta_profile_rel_error = (
        beta_profile_abs_error / beta_exact if beta_exact != 0.0 else np.nan
    )
    identity_residual_at_exact_beta = (
        corr.chi * beta_exact**2 + args.Pe * beta_exact - k_corr
    )
    identity_scale = max(abs(k_corr), 1.0e-300)

    # Fully developed values sampled from the 2D solution in the Gamma window.
    sh_2d_fit = float(np.nanmean(sh_film[gamma_mask]))
    chi_2d_fit = float(np.nanmean(chi_2d[gamma_mask]))

    def relative_error(model: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
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
        "relative_error_constant_vs_exact": (
            (beta_const - beta_exact) / beta_exact if beta_exact != 0.0 else np.nan
        ),
        "identity_residual_using_exact_beta_and_correlated_coefficients": (
            identity_residual_at_exact_beta
        ),
        "normalized_identity_residual": identity_residual_at_exact_beta / identity_scale,
    }

    summary: dict[str, Any] = {
        "inputs": {
            "Pe": args.Pe,
            "Da": args.Da,
            "Lambda": args.Lambda,
            "Cin": args.Cin,
            "inlet_velocity": args.inlet_velocity,
            "advection_scheme": args.advection_scheme,
        },
        "grid": {
            "Nz": nz,
            "Nr": args.Nr,
            "dz": float(z[1] - z[0]),
        },
        "correlation_parameters": asdict(TUBE_CORR),
        "correlation_values": asdict(corr),
        "exact_dominant_mode": asdict(exact_mode),
        "decay_rate_comparison": decay_comparison,
        "Gamma_fit": asdict(gamma_fit),
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

    #parameters_path = save_parameters_errors(output_dir, prefix, summary)
    profile_columns = {
        "Z": z,
        "Cm_full": cm_full,
        "Cm_averaged_correlated": cm_corr,
        "Cm_averaged_constant": cm_const,
        "Cm_rescaled_correlated": cm_corr_scaled,
    }
    #profiles_path = save_profiles(output_dir, prefix, profile_columns)
    profiles_txt_path = save_profiles_txt(output_dir, prefix, profile_columns)
    parameters_txt_path = save_parameters_txt(output_dir, prefix, summary)

    plot_path: Path | None = None
    if not args.no_plots:
        plot_path = save_concentration_plot(
            output_dir=output_dir,
            prefix=prefix,
            z=z,
            cm_full=cm_full,
            cm_corr=cm_corr,
            cm_corr_scaled=cm_corr_scaled,
            cm_const=cm_const,
            constant_sh=args.constant_sh,
            constant_chi=args.constant_chi,
            y_min=args.plot_ymin,
            y_max=args.plot_ymax,
            dpi=args.plot_dpi,
        )

    print("\nUpdated tube/Poiseuille correlation")
    print(f"  Sh(Pe,Da)                    = {corr.sh:.10g}")
    print(f"  chi(Pe,Da)                   = {corr.chi:.10g}")
    print(f"  Pe_c(Da)                     = {corr.pe_c_sh:.10g}")
    print("\nDominant decay-rate comparison")
    print(f"  beta_exact                   = {beta_exact:.10g}")
    print(f"  beta_correlated              = {beta_corr:.10g}")
    print(f"  relative difference          = {beta_rel_error:.6e}")
    print(f"  beta fitted from 2D profile  = {gamma_fit.beta_full_profile_fit:.10g}")
    print("\nDownstream amplitude fit")
    print(f"  Gamma                        = {gamma_fit.Gamma:.10g}")
    print(
        f"  fit window                   = "
        f"[{gamma_fit.z_min:.6g}, {gamma_fit.z_max:.6g}]"
    )
    print(
        f"  scaled mean abs rel error    = "
        f"{gamma_fit.mean_abs_rel_error_scaled:.6e}"
    )
    print("\nSaved files")
    #print(f"  {profiles_path}")
    print(f"  {profiles_txt_path}")
    #print(f"  {parameters_path}")
    print(f"  {parameters_txt_path}")
    if plot_path is not None:
        print(f"  {plot_path}")

    if args.show and plot_path is not None:
        print("\nPlot was saved; open the PNG file to inspect it.")

    return summary


# =========================================================
# Command-line interface
# =========================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Solve the circular-tube extended-Graetz problem, evaluate the "
            "updated Sh/chi correlations, fit Gamma, and save decay diagnostics."
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
        help="velocity used in the Danckwerts inlet condition",
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
    parser.add_argument("--beta-start", type=float, default=1.0e-12)
    parser.add_argument("--beta-max", type=float, default=50.0)
    parser.add_argument("--beta-growth", type=float, default=1.12)
    parser.add_argument("--verbose-eigen", action="store_true")
    parser.add_argument("--output-dir", default="tube_solver_output")
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

    if not (0.0 <= args.gamma_zmin_frac < args.gamma_zmax_frac <= 1.0):
        raise ValueError(
            "Require 0 <= gamma-zmin-frac < gamma-zmax-frac <= 1."
        )
    if args.dz <= 0.0:
        raise ValueError("dz must be positive.")
    if args.Nr < 2:
        raise ValueError("Nr must be at least 2.")
    if not (0.0 < args.plot_ymin < args.plot_ymax):
        raise ValueError("Require 0 < plot-ymin < plot-ymax.")
    if args.plot_dpi <= 0:
        raise ValueError("plot-dpi must be positive.")

    run_case(args)


if __name__ == "__main__":
    main()
