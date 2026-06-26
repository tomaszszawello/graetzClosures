"""
Fast/parallel circular-tube Poiseuille extended-Graetz solver.

Features
--------
1. Augmented ODE states for the cross-sectional integrals, avoiding the
   silent small-Da quadrature/cancellation bias in Sh.
2. Adaptive beta_start for high-Pe/low-Da cases.
3. Normalized Robin residual, with no special Da -> infinity branch.
4. Parallel row-wise grid computation with ProcessPoolExecutor.
5. NaN-on-failure behavior for large sweeps, instead of crashing the run.

Definitions
-----------
Pe = ubar R / D
Da = k R / D
Sh_Dh is based on hydraulic diameter Dh = 2R.
chi = C_a / C_m.
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm


# =========================================================
# Global numerical settings
# =========================================================
RHO0 = 1e-7
RTOL_RESIDUAL = 1e-9
ATOL_RESIDUAL = 1e-11
RTOL_FINAL = 1e-10
ATOL_FINAL = 1e-12

# Conservative global bounds for the radius-based Sherwood number and chi.
# These are only used to choose a safe beta_start decade.
SH_R_MIN_BOUND = 1.75   # below tube Dirichlet limit, 3.6568/2 = 1.8284
SH_R_MAX_BOUND = 3.10   # above tube low-Pe weak-transfer limit, 6/2 = 3
CHI_MIN_BOUND = 0.69
CHI_MAX_BOUND = 1.02


# =========================================================
# Problem definition
# =========================================================
def velocity_profile(rho):
    """Dimensionless Poiseuille profile in a circular tube, mean normalized."""
    rho = np.asarray(rho, dtype=float)
    return 2.0 * (1.0 - rho**2)


def graetz_ode_tube(rho, y, beta, Pe):
    """
    Radial ODE system for the circular-tube eigenproblem.

    y[0] = phi
    y[1] = dphi/drho

    phi'' + (1/rho) phi' + [beta^2 + Pe beta U(rho)] phi = 0.
    """
    phi, dphi = y
    coeff = beta**2 + Pe * beta * velocity_profile(rho)
    ddphi = -(dphi / rho) - coeff * phi
    return [dphi, ddphi]


def graetz_ode_tube_aug(rho, y, beta, Pe):
    """
    Augmented radial ODE system.

    y[0] = phi
    y[1] = dphi/drho
    y[2] = Im = 2 int_0^rho s U(s) phi(s) ds
    y[3] = Ia = 2 int_0^rho s phi(s) ds

    The auxiliary integrals avoid trapezoidal quadrature errors in Cm and Ca.
    """
    phi, dphi, Im, Ia = y
    U = velocity_profile(rho)
    coeff = beta**2 + Pe * beta * U
    ddphi = -(dphi / rho) - coeff * phi
    dIm = 2.0 * rho * U * phi
    dIa = 2.0 * rho * phi
    return [dphi, ddphi, dIm, dIa]


def initial_conditions_series(beta, Pe, rho0=RHO0):
    """
    Regular series expansion near rho = 0.

    phi(rho) = 1 + c2 rho^2 + O(rho^4), with c2 = -a(0)/4.
    Here a(0) = beta^2 + 2 Pe beta.
    """
    a0 = beta**2 + 2.0 * Pe * beta
    phi0 = 1.0 - 0.25 * a0 * rho0**2
    dphi0 = -0.5 * a0 * rho0
    return phi0, dphi0


# =========================================================
# Residual and eigenfunction solves
# =========================================================
def wall_residual(beta, Pe, Da, rho0=RHO0, rtol=RTOL_RESIDUAL, atol=ATOL_RESIDUAL):
    """
    Normalized Robin wall residual:

        R(beta) = [-phi'(1) - Da phi(1)] / (1 + Da).

    The normalization avoids a large residual scale at high Da and removes the
    need for a separate Dirichlet branch.
    """
    if beta <= 0:
        return np.nan

    phi0, dphi0 = initial_conditions_series(beta, Pe, rho0)

    sol = solve_ivp(
        graetz_ode_tube,
        t_span=(rho0, 1.0),
        y0=[phi0, dphi0],
        args=(beta, Pe),
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solve failed for beta={beta}: {sol.message}")

    phi_w = sol.y[0, -1]
    dphi_w = sol.y[1, -1]

    return (-dphi_w - Da * phi_w) / (1.0 + Da)


def eigenfunction_with_integrals(
    beta,
    Pe,
    n_points=None,
    rho0=RHO0,
    rtol=RTOL_FINAL,
    atol=ATOL_FINAL,
):
    """
    Compute eigenfunction and accurate cross-sectional integrals.

    If n_points is None, only the endpoint and final integral values are
    returned. If n_points is an integer, rho, phi, dphi arrays are also sampled.

    Returns:
        rho, phi, dphi, Im, Ia
    where
        Im = 2 int_0^1 rho U phi d rho,
        Ia = 2 int_0^1 rho phi d rho.
    """
    phi0, dphi0 = initial_conditions_series(beta, Pe, rho0)

    # Small-rho integral contributions:
    # Im0 = 2 int_0^rho0 s*U(0)*phi(0) ds = 2 rho0^2
    # Ia0 = 2 int_0^rho0 s*phi(0) ds = rho0^2
    Im0 = 2.0 * rho0**2
    Ia0 = rho0**2

    if n_points is None:
        t_eval = None
    else:
        t_eval = np.linspace(rho0, 1.0, n_points)

    sol = solve_ivp(
        graetz_ode_tube_aug,
        t_span=(rho0, 1.0),
        y0=[phi0, dphi0, Im0, Ia0],
        t_eval=t_eval,
        args=(beta, Pe),
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solve failed for beta={beta}: {sol.message}")

    Im = sol.y[2, -1]
    Ia = sol.y[3, -1]

    if n_points is None:
        rho = np.array([0.0, 1.0])
        phi = np.array([1.0, sol.y[0, -1]])
        dphi = np.array([0.0, sol.y[1, -1]])
    else:
        rho = np.concatenate(([0.0], sol.t))
        phi = np.concatenate(([1.0], sol.y[0]))
        dphi = np.concatenate(([0.0], sol.y[1]))

    return rho, phi, dphi, Im, Ia


def fully_developed_ratios_from_integrals(phi_w, Im, Ia):
    """
    Return Cm/Cw, Ca/Cw, Ca/Cm from auxiliary integrals.
    """
    if abs(phi_w) < 1e-300:
        raise ZeroDivisionError("Wall eigenfunction value too small.")
    if abs(Im) < 1e-300:
        raise ZeroDivisionError("Mixing-cup integral too small.")

    Cm_over_Cw = Im / phi_w
    Cavg_over_Cw = Ia / phi_w
    Cavg_over_Cm = Ia / Im

    return Cm_over_Cw, Cavg_over_Cw, Cavg_over_Cm


def sherwood_radius_from_integrals(Da, phi_w, Im):
    """
    Stable radius-based Sherwood number:

        Sh_R = Da C_w / (C_m - C_w)
             = Da phi_w / (Im - phi_w).
    """
    denom = Im - phi_w
    if abs(denom) < 1e-300:
        raise ZeroDivisionError("Cm - Cw too close to zero.")
    return Da * phi_w / denom


# =========================================================
# Adaptive bracketing
# =========================================================
def beta_from_K_chi(Pe, K, chi):
    """
    Stable positive root of chi beta^2 + Pe beta = K.
    """
    K = max(float(K), 0.0)
    chi = max(float(chi), 1e-300)
    Pe = max(float(Pe), 0.0)

    if K == 0.0:
        return 0.0

    disc = np.sqrt(Pe * Pe + 4.0 * chi * K)
    # Stable version of (-Pe + disc)/(2 chi)
    return 2.0 * K / (Pe + disc)


def beta_start_auto(Pe, Da):
    """
    Conservative start value for root scanning, based on global bounds for
    Sh_R and chi in the closure identity.
    """
    if Da <= 0:
        return 1e-14

    K_min = 2.0 * Da * SH_R_MIN_BOUND / (Da + SH_R_MIN_BOUND)
    beta_min_est = beta_from_K_chi(Pe, K_min, CHI_MAX_BOUND)

    if not np.isfinite(beta_min_est) or beta_min_est <= 0:
        return 1e-14

    return max(1e-300, 0.05 * beta_min_est)


def mode_is_physical(beta, Pe, Da, rho0=RHO0):
    """
    Physical test for the dominant mode.

    The old absolute condition Cm/Cw > 1 + 1e-8 rejects the true mode at very
    small Da, because Cm/Cw - 1 = O(Da). Here we only reject clearly negative
    excess values, while retaining positivity/nodelessness of phi.
    """
    rho, phi, dphi, Im, Ia = eigenfunction_with_integrals(
        beta,
        Pe,
        n_points=450,
        rho0=rho0,
        rtol=1e-9,
        atol=1e-11,
    )

    if phi[-1] <= 0.0:
        return False
    if np.min(phi) <= 0.0:
        return False

    Cm_over_Cw, _, _ = fully_developed_ratios_from_integrals(phi[-1], Im, Ia)
    excess = Cm_over_Cw - 1.0

    # Allow tiny roundoff-scale violations only.
    if excess < -max(1e-12, 1e-6 * max(Da, 1e-300)):
        return False

    return True


class ResidualEvaluator:
    def __init__(
        self,
        Pe,
        Da,
        rho0=RHO0,
        rtol=RTOL_RESIDUAL,
        atol=ATOL_RESIDUAL,
        verbose=False,
        print_every=1,
    ):
        self.Pe = Pe
        self.Da = Da
        self.rho0 = rho0
        self.rtol = rtol
        self.atol = atol
        self.verbose = verbose
        self.print_every = print_every
        self.cache = {}
        self.n_eval = 0

    def __call__(self, beta):
        beta = float(beta)
        if beta in self.cache:
            return self.cache[beta]

        r = wall_residual(
            beta,
            self.Pe,
            self.Da,
            rho0=self.rho0,
            rtol=self.rtol,
            atol=self.atol,
        )
        self.cache[beta] = r
        self.n_eval += 1

        if self.verbose and (self.n_eval % self.print_every == 0):
            sign = "+" if r > 0 else "-" if r < 0 else "0"
            print(f"[eval {self.n_eval:03d}] beta = {beta:12.6g} | residual = {r: .6e} | sign {sign}")

        return r


def scan_candidate_roots(
    Pe,
    Da,
    beta_start=None,
    beta_max=50.0,
    growth=1.12,
    rho0=RHO0,
    rtol_ode=RTOL_RESIDUAL,
    atol_ode=ATOL_RESIDUAL,
    max_roots=2,
    verbose=False,
):
    """
    Scan the normalized residual and collect candidate roots in ascending beta.
    """
    if beta_start is None:
        beta_start = beta_start_auto(Pe, Da)

    R = ResidualEvaluator(
        Pe=Pe,
        Da=Da,
        rho0=rho0,
        rtol=rtol_ode,
        atol=atol_ode,
        verbose=verbose,
        print_every=1,
    )

    roots = []

    a = beta_start
    fa = R(a)

    # If for some reason we started above the first sign region, shrink.
    n_shrink = 0
    while np.isfinite(fa) and fa > 0.0 and a > 1e-300 and n_shrink < 40:
        a *= 0.1
        fa = R(a)
        n_shrink += 1

    while a < beta_max and len(roots) < max_roots:
        b = min(a * growth + 1e-300, beta_max)
        fb = R(b)

        candidate = None

        if fa == 0.0:
            candidate = a

        elif np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0.0:
            root = root_scalar(
                R,
                bracket=(a, b),
                method="brentq",
                xtol=1e-12,
                rtol=1e-12,
            )

            if root.converged:
                candidate = root.root

        if candidate is not None:
            if (
                len(roots) == 0
                or abs(candidate - roots[-1]) > 1e-10 * max(1.0, abs(candidate))
            ):
                roots.append(candidate)
                if verbose:
                    print(f"candidate root #{len(roots)}: beta = {candidate:.12g}")

        a, fa = b, fb

    return roots


def find_beta1_beta2(
    Pe,
    Da,
    beta_start=None,
    beta_max=50.0,
    growth=1.12,
    rho0=RHO0,
    rtol_ode=RTOL_RESIDUAL,
    atol_ode=ATOL_RESIDUAL,
    verbose=False,
):
    """
    Return beta1 = first physical root and beta2 = next candidate root.
    """
    roots = scan_candidate_roots(
        Pe=Pe,
        Da=Da,
        beta_start=beta_start,
        beta_max=beta_max,
        growth=growth,
        rho0=rho0,
        rtol_ode=rtol_ode,
        atol_ode=atol_ode,
        max_roots=8,
        verbose=verbose,
    )

    if len(roots) == 0:
        raise RuntimeError("No candidate roots found.")

    beta1 = None
    beta2 = np.nan

    for i, beta in enumerate(roots):
        ok = mode_is_physical(beta, Pe, Da, rho0=rho0)

        if verbose:
            print(f"root beta = {beta:.12g}, physical = {ok}")

        if ok:
            beta1 = beta
            if i + 1 < len(roots):
                beta2 = roots[i + 1]
            break

    if beta1 is None:
        raise RuntimeError("No physical beta1 found among candidate roots.")

    return beta1, beta2


# =========================================================
# Main point solver
# =========================================================
def solve_beta1_and_sh_tube(
    Pe,
    Da,
    beta_start=None,
    beta_max=50.0,
    growth=1.12,
    rho0=RHO0,
    verbose=False,
):
    beta1, beta2 = find_beta1_beta2(
        Pe=Pe,
        Da=Da,
        beta_start=beta_start,
        beta_max=beta_max,
        growth=growth,
        rho0=rho0,
        verbose=verbose,
    )

    rho, phi, dphi, Im, Ia = eigenfunction_with_integrals(
        beta1,
        Pe=Pe,
        n_points=None,
        rho0=rho0,
        rtol=RTOL_FINAL,
        atol=ATOL_FINAL,
    )

    phi_w = phi[-1]

    Cm_over_Cw, Cavg_over_Cw, Cavg_over_Cm = fully_developed_ratios_from_integrals(
        phi_w,
        Im,
        Ia,
    )

    Sh_R = sherwood_radius_from_integrals(Da, phi_w, Im)

    if np.isfinite(beta2) and beta2 > beta1:
        entrance_length = 1.0 / (beta2 - beta1)
    else:
        entrance_length = np.nan

    return {
        "Pe": Pe,
        "Da": Da,
        "beta1": beta1,
        "beta2": beta2,
        "Cm_over_Cw": Cm_over_Cw,
        "Cavg_over_Cw": Cavg_over_Cw,
        "Cavg_over_Cm": Cavg_over_Cm,
        "chi": Cavg_over_Cm,
        "entrance_length": entrance_length,
        "Sh_radius": Sh_R,
        "Sh_Dh": 2.0 * Sh_R,
        "wall_bc_residual": -dphi[-1] - Da * phi[-1],
        "Im": Im,
        "Ia": Ia,
    }


def solve_point_safe(Pe, Da, beta_max=50.0, growth=1.12):
    """
    Safe point wrapper for parallel sweeps. Returns NaNs on failure.
    """
    try:
        result = solve_beta1_and_sh_tube(
            Pe=Pe,
            Da=Da,
            beta_start=None,
            beta_max=beta_max,
            growth=growth,
            verbose=False,
        )
        return (
            result["Sh_Dh"],
            result["chi"],
            result["entrance_length"],
            result["beta1"],
            result["beta2"],
            0,
        )

    except Exception:
        # Retry once with a slower, denser scan.
        try:
            result = solve_beta1_and_sh_tube(
                Pe=Pe,
                Da=Da,
                beta_start=None,
                beta_max=max(beta_max, 100.0),
                growth=1.06,
                verbose=False,
            )
            return (
                result["Sh_Dh"],
                result["chi"],
                result["entrance_length"],
                result["beta1"],
                result["beta2"],
                0,
            )
        except Exception:
            return (np.nan, np.nan, np.nan, np.nan, np.nan, 1)


def compute_row(args):
    """
    Compute one Da row. This is the unit of parallelization.
    """
    i_da, Da, Pe_tab, beta_max, growth = args

    n_pe = len(Pe_tab)

    Sh_row = np.full(n_pe, np.nan)
    chi_row = np.full(n_pe, np.nan)
    Le_row = np.full(n_pe, np.nan)
    beta1_row = np.full(n_pe, np.nan)
    beta2_row = np.full(n_pe, np.nan)
    fail_row = np.zeros(n_pe, dtype=int)

    for j, Pe in enumerate(Pe_tab):
        Sh, chi, Le, beta1, beta2, fail = solve_point_safe(
            Pe=Pe,
            Da=Da,
            beta_max=beta_max,
            growth=growth,
        )
        Sh_row[j] = Sh
        chi_row[j] = chi
        Le_row[j] = Le
        beta1_row[j] = beta1
        beta2_row[j] = beta2
        fail_row[j] = fail

    return i_da, Sh_row, chi_row, Le_row, beta1_row, beta2_row, fail_row


# =========================================================
# Plotting helpers
# =========================================================
def plot_map(
    Pe_tab,
    Da_tab,
    Z,
    title,
    cbar_label,
    filename,
    cmap="viridis",
    log_color=False,
    symmetric=False,
):
    Pe_grid, Da_grid = np.meshgrid(Pe_tab, Da_tab)

    fig, ax = plt.subplots(figsize=(10, 8))

    if symmetric:
        vmax = np.nanmax(np.abs(Z))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        pcm = ax.pcolormesh(Pe_grid, Da_grid, Z, shading="auto", cmap=cmap, norm=norm)

    elif log_color:
        Zplot = np.array(Z, dtype=float)
        Zplot[~np.isfinite(Zplot)] = np.nan
        Zplot = np.where(Zplot > 0, Zplot, np.nan)
        pcm = ax.pcolormesh(
            Pe_grid,
            Da_grid,
            Zplot,
            shading="auto",
            cmap=cmap,
            norm=LogNorm(vmin=np.nanmin(Zplot), vmax=np.nanmax(Zplot)),
        )

    else:
        pcm = ax.pcolormesh(Pe_grid, Da_grid, Z, shading="auto", cmap=cmap)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Pe")
    ax.set_ylabel("Da")
    ax.set_title(title)

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(cbar_label)

    fig.savefig(filename, bbox_inches="tight", dpi=300)
    plt.show()


def plot_all_maps(Pe_tab, Da_tab, Sh_tab, chi_tab, Le_tab, prefix="tube_cp_pois_fast"):
    plot_map(
        Pe_tab,
        Da_tab,
        Sh_tab,
        title=r"$Sh_{D_h}(Pe,Da)$",
        cbar_label=r"$Sh_{D_h}$",
        filename=f"{prefix}_Sh.png",
        cmap="Reds",
        log_color=False,
    )

    plot_map(
        Pe_tab,
        Da_tab,
        chi_tab,
        title=r"$\chi(Pe,Da)=C_a/C_m$",
        cbar_label=r"$\chi$",
        filename=f"{prefix}_chi.png",
        cmap="Blues",
        log_color=False,
    )

    plot_map(
        Pe_tab,
        Da_tab,
        Le_tab,
        title=r"Entrance length $L_e/R \sim 1/(\beta_2-\beta_1)$",
        cbar_label=r"$L_e/R$",
        filename=f"{prefix}_Le.png",
        cmap="viridis",
        log_color=True,
    )


# =========================================================
# Diagnostics
# =========================================================
def check_low_pe_behavior():
    Pe_values = [0.0, 1e-5, 1e-4, 1e-3]
    Da_values = np.logspace(-4, 1, 120)

    plt.figure(figsize=(7.0, 5.0))

    for Pe in Pe_values:
        Sh_values = []
        for Da in Da_values:
            result = solve_beta1_and_sh_tube(
                Pe=Pe,
                Da=Da,
                beta_start=None,
                beta_max=10.0,
                growth=1.08,
                verbose=False,
            )
            Sh_values.append(result["Sh_Dh"])

        Sh_values = np.asarray(Sh_values)

        print(f"\nPe = {Pe:g}")
        print(f"Sh min = {np.min(Sh_values):.10g}")
        print(f"Sh max = {np.max(Sh_values):.10g}")
        print(f"monotone decreasing? {np.all(np.diff(Sh_values) <= 0.0)}")
        imax = np.argmax(Sh_values)
        print(f"maximum at Da = {Da_values[imax]:.10g}, Sh = {Sh_values[imax]:.10g}")

        plt.semilogx(Da_values, Sh_values, label=rf"$Pe={Pe:g}$")

    plt.axhline(6.0, color="black", lw=1.0, ls="--", label=r"$Pe=0, Da\to0: 6$")
    plt.axhline(48.0 / 11.0, color="gray", lw=1.0, ls=":", label=r"$Da\to0, Pe>0: 48/11$")
    plt.xlabel("Da")
    plt.ylabel(r"$Sh_{D_h}$")
    plt.legend(fontsize=9)
    plt.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    plt.savefig("lowPe_Sh_Da_diagnostic_fast.png", dpi=300, bbox_inches="tight")
    plt.show()


# =========================================================
# Main parallel sweep
# =========================================================
def run_parallel_grid(
    Pe_tab,
    Da_tab,
    prefix="tube_cp_pois_fast",
    n_workers=None,
    beta_max=10.0,
    growth=1.12,
    make_plots=True,
):
    Pe_tab = np.asarray(Pe_tab, dtype=float)
    Da_tab = np.asarray(Da_tab, dtype=float)

    n_da = len(Da_tab)
    n_pe = len(Pe_tab)

    Sh_tab = np.full((n_da, n_pe), np.nan)
    chi_tab = np.full((n_da, n_pe), np.nan)
    Le_tab = np.full((n_da, n_pe), np.nan)
    beta1_tab = np.full((n_da, n_pe), np.nan)
    beta2_tab = np.full((n_da, n_pe), np.nan)
    fail_tab = np.zeros((n_da, n_pe), dtype=int)

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    tasks = [
        (i_da, float(Da), Pe_tab, beta_max, growth)
        for i_da, Da in enumerate(Da_tab)
    ]

    start_time = time.time()

    if n_workers == 1:
        iterator = map(compute_row, tasks)
        for row_result in iterator:
            i_da, Sh_row, chi_row, Le_row, beta1_row, beta2_row, fail_row = row_result
            Sh_tab[i_da, :] = Sh_row
            chi_tab[i_da, :] = chi_row
            Le_tab[i_da, :] = Le_row
            beta1_tab[i_da, :] = beta1_row
            beta2_tab[i_da, :] = beta2_row
            fail_tab[i_da, :] = fail_row
            elapsed = time.time() - start_time
            print(f"completed row {i_da + 1}/{n_da}  Da={Da_tab[i_da]:.6g}  elapsed={elapsed:.1f}s  failures={np.sum(fail_row)}")

    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(compute_row, task) for task in tasks]

            n_done = 0
            for fut in as_completed(futures):
                i_da, Sh_row, chi_row, Le_row, beta1_row, beta2_row, fail_row = fut.result()

                Sh_tab[i_da, :] = Sh_row
                chi_tab[i_da, :] = chi_row
                Le_tab[i_da, :] = Le_row
                beta1_tab[i_da, :] = beta1_row
                beta2_tab[i_da, :] = beta2_row
                fail_tab[i_da, :] = fail_row

                n_done += 1
                elapsed = time.time() - start_time
                print(
                    f"completed row {n_done}/{n_da}  "
                    f"Da={Da_tab[i_da]:.6g}  "
                    f"elapsed={elapsed:.1f}s  failures={np.sum(fail_row)}",
                    flush=True,
                )

    rows_out = []
    for i, Da in enumerate(Da_tab):
        for j, Pe in enumerate(Pe_tab):
            rows_out.append(
                [
                    Pe,
                    Da,
                    Sh_tab[i, j],
                    chi_tab[i, j],
                    Le_tab[i, j],
                    beta1_tab[i, j],
                    beta2_tab[i, j],
                    fail_tab[i, j],
                ]
            )
    rows_out = np.asarray(rows_out)

    np.savetxt(f"Sh_{prefix}.txt", Sh_tab)
    np.savetxt(f"chi_{prefix}.txt", chi_tab)
    np.savetxt(f"Le_{prefix}.txt", Le_tab)
    np.savetxt(f"beta1_{prefix}.txt", beta1_tab)
    np.savetxt(f"beta2_{prefix}.txt", beta2_tab)
    np.savetxt(f"fail_{prefix}.txt", fail_tab, fmt="%d")
    np.savetxt(f"Pe_{prefix}.txt", Pe_tab)
    np.savetxt(f"Da_{prefix}.txt", Da_tab)

    np.savetxt(
        f"{prefix}_flat.txt",
        rows_out,
        header="Pe Da Sh_Dh chi entrance_length beta1 beta2 fail",
    )

    print("\nSaved files:")
    print(f"  Sh_{prefix}.txt")
    print(f"  chi_{prefix}.txt")
    print(f"  Le_{prefix}.txt")
    print(f"  Pe_{prefix}.txt")
    print(f"  Da_{prefix}.txt")
    print(f"  {prefix}_flat.txt")
    print(f"Total failures: {np.sum(fail_tab)} / {n_da * n_pe}")

    if make_plots:
        plot_all_maps(Pe_tab, Da_tab, Sh_tab, chi_tab, Le_tab, prefix=prefix)

    return Sh_tab, chi_tab, Le_tab, beta1_tab, beta2_tab, fail_tab


if __name__ == "__main__":
    # -----------------------------------------------------
    # Choose what to run
    # -----------------------------------------------------
    RUN_LOW_PE_DIAGNOSTIC = False
    RUN_FULL_TABLE = True

    if RUN_LOW_PE_DIAGNOSTIC:
        check_low_pe_behavior()

    if RUN_FULL_TABLE:
        # Start with a smaller grid to test speed and robustness.
        # Increase to 1000 only when you are ready for a long production run.
        N_PE = 1000
        N_DA = 1000

        Pe_tab = np.logspace(-3, 3, N_PE)
        Da_tab = np.logspace(-3, 3, N_DA)

        run_parallel_grid(
            Pe_tab=Pe_tab,
            Da_tab=Da_tab,
            prefix="tube_cp_pois_fast",
            n_workers=None,     # None -> use all but one CPU core
            beta_max=10.0,
            growth=1.10,
            make_plots=True,
        )
