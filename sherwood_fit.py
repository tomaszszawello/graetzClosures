import argparse
import numpy as np
from scipy.optimize import least_squares


# =========================================================
# Geometry-specific settings
# Allowed command-line arguments: tube, plates, oneplate
# =========================================================
J01 = 2.404825557695773  # first zero of J0

GEOMETRY_CONFIGS = {
    "tube": {
        "label": "Circular tube, Poiseuille flow",
        "short": "tube",
        "sh_file": "Sh_tube_pois.txt",
        "pe_file": "Pe.txt",
        "da_file": "Da.txt",
        "output_prefix": "tube_pois",
        # hydraulic-diameter Sherwood, Dh = 2R
        "SH_INF_DA0_EXACT": 48.0 / 11.0,
        "SH_INF_DAINF_EXACT": 3.65679,
        "SH_0_DA0_EXACT": 6.0,
        "SH_0_DAINF_EXACT": J01**4 / 8.0,
    },
    "plates": {
        "label": "Parallel plates, Poiseuille flow, two reactive surfaces",
        "short": "plates",
        "sh_file": "Sh_plates_pois.txt",
        "pe_file": "Pe.txt",
        "da_file": "Da.txt",
        "output_prefix": "plates_pois",
        # hydraulic-diameter Sherwood, Dh = 4a
        "SH_INF_DA0_EXACT": 140.0 / 17.0,
        "SH_INF_DAINF_EXACT": 7.5410,
        "SH_0_DA0_EXACT": 10.0,
        "SH_0_DAINF_EXACT": np.pi**4 / 12.0,
    },
    "oneplate": {
        "label": "Parallel plates, Poiseuille flow, one reactive surface",
        "short": "oneplate",
        "sh_file": "Sh_oneplate_pois.txt",
        "pe_file": "Pe.txt",
        "da_file": "Da.txt",
        "output_prefix": "oneplate_pois",
        # hydraulic-diameter Sherwood, Dh = 2a
        "SH_INF_DA0_EXACT": 70.0 / 13.0,
        "SH_INF_DAINF_EXACT": 4.86073677894,
        "SH_0_DA0_EXACT": 40.0 / 7.0,
        "SH_0_DAINF_EXACT": np.pi**4 / (24.0 * (4.0 - np.pi)),
    },
}

# Fixed exponents
M_PC_FIXED = 2.0 / 3.0
NP_FIXED = 4.0 / 3.0

# These are set from the selected geometry in set_geometry().
GEOMETRY = None
CONFIG = None
SH_INF_DA0_EXACT = None
SH_INF_DAINF_EXACT = None
SH_0_DA0_EXACT = None
SH_0_DAINF_EXACT = None


# =========================================================
# Argument parsing and geometry selection
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fit Sherwood-number correlation with fixed exponents for one geometry. "
            "Allowed geometries: tube, plates, oneplate. Saves one text summary only."
        )
    )
    parser.add_argument(
        "geometry",
        choices=["tube", "plates", "oneplate"],
        help="Geometry to fit: tube, plates, or oneplate.",
    )
    parser.add_argument(
        "--n-top-pe",
        type=int,
        default=5,
        help="Number of largest-Pe columns averaged for the high-Pe asymptote.",
    )
    parser.add_argument(
        "--n-low-pe",
        type=int,
        default=5,
        help="Number of smallest-Pe columns averaged for the low-Pe asymptote.",
    )
    parser.add_argument(
        "--log-residuals",
        action="store_true",
        help="Fit the Pe-blend using logarithmic residuals instead of relative residuals.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing Sh, Pe, and Da text files. Default: data",
    )
    parser.add_argument(
        "--output-dir",
        default="fits",
        help="Directory for the fit summary file. Default: fits",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional full output path. Overrides --output-dir.",
    )
    return parser.parse_args()


def set_geometry(geometry):
    global GEOMETRY, CONFIG
    global SH_INF_DA0_EXACT, SH_INF_DAINF_EXACT, SH_0_DA0_EXACT, SH_0_DAINF_EXACT

    GEOMETRY = geometry
    CONFIG = GEOMETRY_CONFIGS[geometry]

    SH_INF_DA0_EXACT = CONFIG["SH_INF_DA0_EXACT"]
    SH_INF_DAINF_EXACT = CONFIG["SH_INF_DAINF_EXACT"]
    SH_0_DA0_EXACT = CONFIG["SH_0_DA0_EXACT"]
    SH_0_DAINF_EXACT = CONFIG["SH_0_DAINF_EXACT"]

    return CONFIG


# =========================================================
# Rounding helpers
# =========================================================
def round4(x):
    return float(np.round(x, 4))


def rounded_constants():
    return {
        "SH_INF_DA0": round4(SH_INF_DA0_EXACT),
        "SH_INF_DAINF": round4(SH_INF_DAINF_EXACT),
        "SH_0_DA0": round4(SH_0_DA0_EXACT),
        "SH_0_DAINF": round4(SH_0_DAINF_EXACT),
    }


# =========================================================
# Data loading
# =========================================================
def load_data(config, data_dir="data"):
    Sh_tab = np.loadtxt(f"{data_dir}/{config['sh_file']}")
    Pe_tab = np.loadtxt(f"{data_dir}/{config['pe_file']}")
    Da_tab = np.loadtxt(f"{data_dir}/{config['da_file']}")

    Pe_tab = np.asarray(Pe_tab, dtype=float).ravel()
    Da_tab = np.asarray(Da_tab, dtype=float).ravel()
    Sh_tab = np.asarray(Sh_tab, dtype=float)

    if np.any(Pe_tab <= 0):
        raise ValueError("All Pe values must be > 0.")
    if np.any(Da_tab <= 0):
        raise ValueError("All Da values must be > 0.")

    if Sh_tab.ndim != 2:
        raise ValueError("Sh table must be 2D.")

    if Sh_tab.shape == (len(Pe_tab), len(Da_tab)) and Sh_tab.shape != (len(Da_tab), len(Pe_tab)):
        print("Transposing Sh_tab to match (len(Da_tab), len(Pe_tab)).")
        Sh_tab = Sh_tab.T
    elif Sh_tab.shape != (len(Da_tab), len(Pe_tab)):
        raise ValueError(
            f"Unexpected Sh_tab shape {Sh_tab.shape}, "
            f"expected {(len(Da_tab), len(Pe_tab))}."
        )

    i_pe = np.argsort(Pe_tab)
    i_da = np.argsort(Da_tab)

    Pe_tab = Pe_tab[i_pe]
    Da_tab = Da_tab[i_da]
    Sh_tab = Sh_tab[np.ix_(i_da, i_pe)]

    return Pe_tab, Da_tab, Sh_tab


# =========================================================
# First-order profile in Da
# =========================================================
def first_order_profile_physical(Da, y_0, y_inf, Da_c):
    Da = np.asarray(Da, dtype=float)
    return y_inf + (y_0 - y_inf) / (1.0 + Da / Da_c)


def first_order_profile_log(Da, y_0, y_inf, logDa_c):
    Da_c = 10.0 ** logDa_c
    return first_order_profile_physical(Da, y_0, y_inf, Da_c)


# =========================================================
# High-Pe asymptote
# =========================================================
def sh_inf_model(params_inf, Da):
    (logDa_c_inf,) = params_inf
    return first_order_profile_log(
        Da,
        SH_INF_DA0_EXACT,
        SH_INF_DAINF_EXACT,
        logDa_c_inf,
    )


def sh_inf_model_physical(Da, Da_c_inf, constants):
    return first_order_profile_physical(
        Da,
        constants["SH_INF_DA0"],
        constants["SH_INF_DAINF"],
        Da_c_inf,
    )


def residuals_sh_inf(params_inf, Da, Sh_inf_data):
    Sh_fit = sh_inf_model(params_inf, Da)
    if np.any(~np.isfinite(Sh_fit)) or np.any(Sh_fit <= 0):
        return 1e6 * np.ones_like(Sh_inf_data)
    return (Sh_fit - Sh_inf_data) / Sh_inf_data


def fit_sh_inf(Da_tab, Sh_tab, n_top_pe=5):
    n_top_pe = min(n_top_pe, Sh_tab.shape[1])
    Sh_inf_data_all = np.nanmean(Sh_tab[:, -n_top_pe:], axis=1)

    mask = np.isfinite(Da_tab) & np.isfinite(Sh_inf_data_all) & (Da_tab > 0) & (Sh_inf_data_all > 0)
    if np.count_nonzero(mask) < 2:
        raise RuntimeError("Not enough finite high-Pe asymptote data to fit.")

    Da_fit = Da_tab[mask]
    Sh_inf_data = Sh_inf_data_all[mask]

    lda_min = np.log10(np.min(Da_fit))
    lda_max = np.log10(np.max(Da_fit))
    lda_mid = 0.5 * (lda_min + lda_max)

    guesses = [[lda_mid], [lda_min], [lda_max]]
    lower = np.array([lda_min - 6.0])
    upper = np.array([lda_max + 6.0])

    best = None
    best_cost = np.inf

    for p0 in guesses:
        res = least_squares(
            residuals_sh_inf,
            p0,
            args=(Da_fit, Sh_inf_data),
            bounds=(lower, upper),
            method="trf",
            loss="soft_l1",
            f_scale=0.05,
            max_nfev=50000,
        )

        if res.cost < best_cost:
            best = res
            best_cost = res.cost

    return best, Sh_inf_data_all


# =========================================================
# Low-Pe asymptote
# =========================================================
def sh0_model(params0, Da):
    (logDa_c0,) = params0
    return first_order_profile_log(
        Da,
        SH_0_DA0_EXACT,
        SH_0_DAINF_EXACT,
        logDa_c0,
    )


def sh0_model_physical(Da, Da_c0, constants):
    return first_order_profile_physical(
        Da,
        constants["SH_0_DA0"],
        constants["SH_0_DAINF"],
        Da_c0,
    )


def residuals_sh0(params0, Da, Sh0_data):
    Sh_fit = sh0_model(params0, Da)
    if np.any(~np.isfinite(Sh_fit)) or np.any(Sh_fit <= 0):
        return 1e6 * np.ones_like(Sh0_data)
    return (Sh_fit - Sh0_data) / Sh0_data


def fit_sh0(Da_tab, Sh_tab, n_low_pe=5):
    n_low_pe = min(n_low_pe, Sh_tab.shape[1])
    Sh0_data_all = np.nanmean(Sh_tab[:, :n_low_pe], axis=1)

    mask = np.isfinite(Da_tab) & np.isfinite(Sh0_data_all) & (Da_tab > 0) & (Sh0_data_all > 0)
    if np.count_nonzero(mask) < 2:
        raise RuntimeError("Not enough finite low-Pe asymptote data to fit.")

    Da_fit = Da_tab[mask]
    Sh0_data = Sh0_data_all[mask]

    lda_min = np.log10(np.min(Da_fit))
    lda_max = np.log10(np.max(Da_fit))
    lda_mid = 0.5 * (lda_min + lda_max)

    guesses = [[lda_mid], [lda_min], [lda_max]]
    lower = np.array([lda_min - 6.0])
    upper = np.array([lda_max + 6.0])

    best = None
    best_cost = np.inf

    for p0 in guesses:
        res = least_squares(
            residuals_sh0,
            p0,
            args=(Da_fit, Sh0_data),
            bounds=(lower, upper),
            method="trf",
            loss="soft_l1",
            f_scale=0.05,
            max_nfev=50000,
        )

        if res.cost < best_cost:
            best = res
            best_cost = res.cost

    return best, Sh0_data_all


# =========================================================
# Pc(Da), with fixed m_pc = 2/3
# =========================================================
def pc_model_physical(Da, Pc0, Pc_inf, Da_c_pc, m_pc=M_PC_FIXED):
    Da = np.asarray(Da, dtype=float)

    t = m_pc * (np.log(Da) - np.log(Da_c_pc))
    t = np.clip(t, -60.0, 60.0)
    x = np.exp(t)

    return Pc_inf - (Pc_inf - Pc0) / (1.0 + x)


def pc_model_log(Da, logPc0, logPc_inf, logDa_c_pc):
    Pc0 = 10.0 ** logPc0
    Pc_inf = 10.0 ** logPc_inf
    Da_c_pc = 10.0 ** logDa_c_pc

    return pc_model_physical(Da, Pc0, Pc_inf, Da_c_pc, M_PC_FIXED)


# =========================================================
# Full blend with fixed nP = 4/3 and m_pc = 2/3
# params_blend = [logPc0, logPc_inf, logDa_c_pc]
# =========================================================
def sh_blend_model(params_blend, params_inf, params0, Pe, Da):
    logPc0, logPc_inf, logDa_c_pc = params_blend

    Sh_inf = sh_inf_model(params_inf, Da)
    Sh_0 = sh0_model(params0, Da)
    Pc = pc_model_log(Da, logPc0, logPc_inf, logDa_c_pc)

    return Sh_inf + (Sh_0 - Sh_inf) / (1.0 + (Pe / Pc) ** NP_FIXED)


def sh_blend_model_physical(Pe, Da, pars, constants):
    Sh_inf = sh_inf_model_physical(Da, pars["Da_c_inf"], constants)
    Sh_0 = sh0_model_physical(Da, pars["Da_c0"], constants)

    Pc = pc_model_physical(
        Da,
        pars["Pc0"],
        pars["Pc_inf"],
        pars["Da_c_pc"],
        M_PC_FIXED,
    )

    return Sh_inf + (Sh_0 - Sh_inf) / (1.0 + (Pe / Pc) ** NP_FIXED)


def residuals_blend_relative(params_blend, params_inf, params0, Pe, Da, Sh):
    Sh_fit = sh_blend_model(params_blend, params_inf, params0, Pe, Da)

    if np.any(~np.isfinite(Sh_fit)) or np.any(Sh_fit <= 0):
        return 1e6 * np.ones_like(Sh)

    return (Sh_fit - Sh) / Sh


def residuals_blend_log(params_blend, params_inf, params0, Pe, Da, Sh):
    Sh_fit = sh_blend_model(params_blend, params_inf, params0, Pe, Da)

    if np.any(~np.isfinite(Sh_fit)) or np.any(Sh_fit <= 0):
        return 1e6 * np.ones_like(Sh)

    return np.log(Sh_fit) - np.log(Sh)


def initial_guesses_for_blend():
    # Geometry-specific initial guesses are not critical, but these are close to
    # the values observed in the fitted tables and speed up convergence.
    if GEOMETRY == "tube":
        return [
            [np.log10(0.012), np.log10(1.0), np.log10(0.6)],
            [np.log10(0.020), np.log10(1.0), np.log10(0.5)],
            [np.log10(0.010), np.log10(1.1), np.log10(1.0)],
            [np.log10(0.030), np.log10(0.9), np.log10(0.3)],
        ]
    if GEOMETRY == "plates":
        return [
            [np.log10(0.009), np.log10(0.8), np.log10(0.6)],
            [np.log10(0.015), np.log10(0.8), np.log10(0.5)],
            [np.log10(0.010), np.log10(1.0), np.log10(1.0)],
            [np.log10(0.030), np.log10(0.7), np.log10(0.3)],
        ]
    # oneplate
    return [
        [np.log10(0.010), np.log10(0.8), np.log10(0.6)],
        [np.log10(0.015), np.log10(0.8), np.log10(0.5)],
        [np.log10(0.010), np.log10(1.0), np.log10(1.0)],
        [np.log10(0.030), np.log10(0.7), np.log10(0.3)],
    ]


def fit_blend_model(Pe_data, Da_data, Sh_data, params_inf, params0, use_log_residuals=False):
    residual_fun = residuals_blend_log if use_log_residuals else residuals_blend_relative

    lda_min = np.log10(np.min(Da_data))
    lda_max = np.log10(np.max(Da_data))
    lda_mid = 0.5 * (lda_min + lda_max)

    guesses = initial_guesses_for_blend()
    guesses.append([np.log10(0.015), np.log10(1.0), lda_mid])

    lower = np.array([
        -12.0,          # logPc0
        -12.0,          # logPc_inf
        lda_min - 6.0,  # logDa_c_pc
    ])

    upper = np.array([
        12.0,
        12.0,
        lda_max + 6.0,
    ])

    best = None
    best_cost = np.inf

    for p0 in guesses:
        res = least_squares(
            residual_fun,
            p0,
            args=(params_inf, params0, Pe_data, Da_data, Sh_data),
            bounds=(lower, upper),
            method="trf",
            loss="soft_l1",
            f_scale=0.05,
            max_nfev=50000,
        )

        if res.cost < best_cost:
            best = res
            best_cost = res.cost

    return best


# =========================================================
# Diagnostics and summary formatting
# =========================================================
def unpack_full_precision_params(params_inf, params0, params_blend):
    logDa_c_inf = params_inf[0]
    logDa_c0 = params0[0]
    logPc0, logPc_inf, logDa_c_pc = params_blend

    return {
        "Da_c_inf": 10.0 ** logDa_c_inf,
        "Da_c0": 10.0 ** logDa_c0,
        "Pc0": 10.0 ** logPc0,
        "Pc_inf": 10.0 ** logPc_inf,
        "Da_c_pc": 10.0 ** logDa_c_pc,
    }


def round_params_4(pars):
    return {k: round4(v) for k, v in pars.items()}


def fit_quality(Sh_true, Sh_fit):
    rel_err = (Sh_fit - Sh_true) / Sh_true
    abs_rel_err = np.abs(rel_err)

    return {
        "mean_abs_rel": np.mean(abs_rel_err),
        "median_abs_rel": np.median(abs_rel_err),
        "max_abs_rel": np.max(abs_rel_err),
        "rms_rel": np.sqrt(np.mean(rel_err**2)),
    }


def worst_point(Pe, Da, Sh_true, Sh_fit):
    rel_err = (Sh_fit - Sh_true) / Sh_true
    abs_rel_err = np.abs(rel_err)
    imax = np.argmax(abs_rel_err)

    return {
        "Pe": Pe[imax],
        "Da": Da[imax],
        "Sh_true": Sh_true[imax],
        "Sh_fit": Sh_fit[imax],
        "rel_err": rel_err[imax],
        "abs_rel_err": abs_rel_err[imax],
    }


def corner_limits(pars_round, constants_round):
    Pe_small = 1e-30
    Pe_big = 1e30
    Da_small = 1e-30
    Da_big = 1e30

    return {
        "Pe0_Da0": sh_blend_model_physical(Pe_small, Da_small, pars_round, constants_round),
        "Pe0_Dainf": sh_blend_model_physical(Pe_small, Da_big, pars_round, constants_round),
        "Peinf_Da0": sh_blend_model_physical(Pe_big, Da_small, pars_round, constants_round),
        "Peinf_Dainf": sh_blend_model_physical(Pe_big, Da_big, pars_round, constants_round),
    }


def add_key_values(lines, title, values, float_fmt=".12g"):
    lines.append("")
    lines.append(title)
    lines.append("-" * len(title))
    for key, val in values.items():
        if isinstance(val, (float, np.floating)):
            lines.append(f"{key:20s} = {val:{float_fmt}}")
        else:
            lines.append(f"{key:20s} = {val}")


def build_summary_text(
    config,
    args,
    pars_full,
    pars_round,
    constants_round,
    quality_full,
    quality_rounded,
    worst_full,
    worst_rounded,
    corner_round,
    n_valid_points,
):
    lines = []

    lines.append("Sherwood correlation fit summary")
    lines.append("================================")
    lines.append(f"geometry              = {args.geometry}")
    lines.append(f"label                 = {config['label']}")
    lines.append(f"Sh file               = data/{config['sh_file']}")
    lines.append(f"Pe file               = data/{config['pe_file']}")
    lines.append(f"Da file               = data/{config['da_file']}")
    lines.append(f"valid fitted points   = {n_valid_points}")
    lines.append(f"n_top_pe              = {args.n_top_pe}")
    lines.append(f"n_low_pe              = {args.n_low_pe}")
    lines.append(f"blend residuals       = {'logarithmic' if args.log_residuals else 'relative'}")

    lines.append("")
    lines.append("Fitted structure")
    lines.append("----------------")
    lines.append("Sh_inf(Da) = SH_INF_DAINF + (SH_INF_DA0 - SH_INF_DAINF)/(1 + Da/Da_c_inf)")
    lines.append("Sh_0(Da)   = SH_0_DAINF  + (SH_0_DA0  - SH_0_DAINF )/(1 + Da/Da_c0)")
    lines.append("Pc(Da)     = Pc_inf - (Pc_inf - Pc0)/(1 + (Da/Da_c_pc)^(2/3))")
    lines.append("Sh(Pe,Da)  = Sh_inf(Da) + (Sh_0(Da) - Sh_inf(Da))/(1 + (Pe/Pc(Da))^(4/3))")

    exact_constants = {
        "SH_0_DA0": SH_0_DA0_EXACT,
        "SH_0_DAINF": SH_0_DAINF_EXACT,
        "SH_INF_DA0": SH_INF_DA0_EXACT,
        "SH_INF_DAINF": SH_INF_DAINF_EXACT,
        "m_pc": M_PC_FIXED,
        "nP": NP_FIXED,
    }
    add_key_values(lines, "Exact constants and fixed exponents", exact_constants)
    add_key_values(lines, "Full-precision fitted parameters", pars_full)

    rounded_all = dict(constants_round)
    rounded_all.update(pars_round)
    rounded_all["m_pc"] = "2/3"
    rounded_all["nP"] = "4/3"
    add_key_values(lines, "Rounded constants and fitted parameters", rounded_all, float_fmt=".4f")

    add_key_values(lines, "Corner limits from rounded formula", corner_round)
    add_key_values(lines, "Fit quality: full-precision formula", quality_full)
    add_key_values(lines, "Worst point: full-precision formula", worst_full)
    add_key_values(lines, "Fit quality: rounded formula", quality_rounded)
    add_key_values(lines, "Worst point: rounded formula", worst_rounded)

    return "\n".join(lines) + "\n"


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()
    config = set_geometry(args.geometry)
    prefix = config["output_prefix"]

    from pathlib import Path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output or str(output_dir / f"{prefix}_fit_summary.txt")

    Pe_tab, Da_tab, Sh_tab = load_data(config, data_dir=args.data_dir)

    Pe_grid, Da_grid = np.meshgrid(Pe_tab, Da_tab)

    Pe_data = Pe_grid.ravel()
    Da_data = Da_grid.ravel()
    Sh_data = Sh_tab.ravel()

    mask = (
        np.isfinite(Pe_data)
        & np.isfinite(Da_data)
        & np.isfinite(Sh_data)
        & (Pe_data > 0)
        & (Da_data > 0)
        & (Sh_data > 0)
    )

    Pe_data = Pe_data[mask]
    Da_data = Da_data[mask]
    Sh_data = Sh_data[mask]

    # Step 1: fit high-Pe asymptote
    res_inf, _Sh_inf_data = fit_sh_inf(Da_tab, Sh_tab, n_top_pe=args.n_top_pe)
    params_inf = res_inf.x

    # Step 2: fit low-Pe asymptote
    res0, _Sh0_data = fit_sh0(Da_tab, Sh_tab, n_low_pe=args.n_low_pe)
    params0 = res0.x

    # Step 3: fit Pe blend with fixed exponents
    res_blend = fit_blend_model(
        Pe_data,
        Da_data,
        Sh_data,
        params_inf=params_inf,
        params0=params0,
        use_log_residuals=args.log_residuals,
    )
    params_blend = res_blend.x

    # Full-precision and rounded parameters
    pars_full = unpack_full_precision_params(params_inf, params0, params_blend)
    pars_round = round_params_4(pars_full)
    constants_round = rounded_constants()

    # Errors from full-precision formula
    Sh_fit_full = sh_blend_model(params_blend, params_inf, params0, Pe_data, Da_data)
    quality_full = fit_quality(Sh_data, Sh_fit_full)
    worst_full = worst_point(Pe_data, Da_data, Sh_data, Sh_fit_full)

    # Errors from rounded formula
    Sh_fit_rounded = sh_blend_model_physical(Pe_data, Da_data, pars_round, constants_round)
    quality_rounded = fit_quality(Sh_data, Sh_fit_rounded)
    worst_rounded = worst_point(Pe_data, Da_data, Sh_data, Sh_fit_rounded)

    # Corner limits from rounded formula
    corner_round = corner_limits(pars_round, constants_round)

    summary = build_summary_text(
        config=config,
        args=args,
        pars_full=pars_full,
        pars_round=pars_round,
        constants_round=constants_round,
        quality_full=quality_full,
        quality_rounded=quality_rounded,
        worst_full=worst_full,
        worst_rounded=worst_rounded,
        corner_round=corner_round,
        n_valid_points=len(Sh_data),
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"Saved fitted parameters and errors to {output_file}")


if __name__ == "__main__":
    main()
