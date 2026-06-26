import argparse
import os
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
        "chi_file": "chi_tube_pois.txt",
        "pe_file": "Pe.txt",
        "da_file": "Da.txt",
        "output_prefix": "tube_pois_chi",
        "CHI_0_DA0": 1.0,
        "CHI_0_DAINF": J01**2 / 8.0,
        "CHI_INF_DA0": 1.0,
        "CHI_INF_DAINF": 0.7095056047585214,
    },
    "plates": {
        "label": "Parallel plates, Poiseuille flow, two reactive surfaces",
        "short": "plates",
        "chi_file": "chi_plates_pois.txt",
        "pe_file": "Pe.txt",
        "da_file": "Da.txt",
        "output_prefix": "plates_pois_chi",
        "CHI_0_DA0": 1.0,
        "CHI_0_DAINF": np.pi**2 / 12.0,
        "CHI_INF_DA0": 1.0,
        "CHI_INF_DAINF": 0.8170888406237363,
    },
    "oneplate": {
        "label": "Parallel plates, Poiseuille flow, one reactive surface",
        "short": "oneplate",
        "chi_file": "chi_oneplate_pois.txt",
        "pe_file": "Pe.txt",
        "da_file": "Da.txt",
        "output_prefix": "oneplate_pois_chi",
        "CHI_0_DA0": 1.0,
        "CHI_0_DAINF": np.pi**2 / (12.0 * (4.0 - np.pi)),
        "CHI_INF_DA0": 1.0,
        "CHI_INF_DAINF": 0.94593699617,
    },
}

# These are set from the selected geometry in set_geometry().
GEOMETRY = None
CONFIG = None
CHI_0_DA0 = None
CHI_0_DAINF = None
CHI_INF_DA0 = None
CHI_INF_DAINF = None


# =========================================================
# Argument parsing and geometry selection
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fit chi=C_a/C_m correlation for one geometry. "
            "Allowed geometries: tube, plates, oneplate. Saves one text summary only."
        )
    )
    parser.add_argument(
        "geometry",
        choices=["tube", "plates", "oneplate"],
        help="Geometry to fit: tube, plates, or oneplate.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing chi, Pe, and Da text files. Default: data",
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
        "--relative-residuals",
        action="store_true",
        help="Fit the Pe-blend using relative residuals instead of absolute residuals.",
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
    global CHI_0_DA0, CHI_0_DAINF, CHI_INF_DA0, CHI_INF_DAINF

    GEOMETRY = geometry
    CONFIG = GEOMETRY_CONFIGS[geometry]

    CHI_0_DA0 = CONFIG["CHI_0_DA0"]
    CHI_0_DAINF = CONFIG["CHI_0_DAINF"]
    CHI_INF_DA0 = CONFIG["CHI_INF_DA0"]
    CHI_INF_DAINF = CONFIG["CHI_INF_DAINF"]

    return CONFIG


# =========================================================
# Rounding helpers
# =========================================================
def round4(x):
    return float(np.round(x, 4))


def rounded_constants():
    return {
        "CHI_0_DA0": round4(CHI_0_DA0),
        "CHI_0_DAINF": round4(CHI_0_DAINF),
        "CHI_INF_DA0": round4(CHI_INF_DA0),
        "CHI_INF_DAINF": round4(CHI_INF_DAINF),
    }


# =========================================================
# Data loading
# =========================================================
def path_in_data_dir(data_dir, filename):
    return os.path.join(data_dir, filename)


def load_data(config, data_dir="data"):
    chi_path = path_in_data_dir(data_dir, config["chi_file"])
    pe_path = path_in_data_dir(data_dir, config["pe_file"])
    da_path = path_in_data_dir(data_dir, config["da_file"])

    chi_tab = np.loadtxt(chi_path)
    Pe_tab = np.loadtxt(pe_path)
    Da_tab = np.loadtxt(da_path)

    Pe_tab = np.asarray(Pe_tab, dtype=float).ravel()
    Da_tab = np.asarray(Da_tab, dtype=float).ravel()
    chi_tab = np.asarray(chi_tab, dtype=float)

    if np.any(Pe_tab <= 0):
        raise ValueError("All Pe values must be > 0.")
    if np.any(Da_tab <= 0):
        raise ValueError("All Da values must be > 0.")

    if chi_tab.ndim != 2:
        raise ValueError("chi table must be 2D.")

    if chi_tab.shape == (len(Pe_tab), len(Da_tab)) and chi_tab.shape != (len(Da_tab), len(Pe_tab)):
        print("Transposing chi_tab to match (len(Da_tab), len(Pe_tab)).")
        chi_tab = chi_tab.T
    elif chi_tab.shape != (len(Da_tab), len(Pe_tab)):
        raise ValueError(
            f"Unexpected chi_tab shape {chi_tab.shape}, "
            f"expected {(len(Da_tab), len(Pe_tab))}."
        )

    i_pe = np.argsort(Pe_tab)
    i_da = np.argsort(Da_tab)

    Pe_tab = Pe_tab[i_pe]
    Da_tab = Da_tab[i_da]
    chi_tab = chi_tab[np.ix_(i_da, i_pe)]

    return Pe_tab, Da_tab, chi_tab


# =========================================================
# First-order profile in Da
#
# chi_branch(Da) = chi_Dainf + (chi_Da0 - chi_Dainf)/(1 + Da/Da_c)
# Here chi_Da0 = 1 for all geometries used here.
# =========================================================
def chi_branch_physical(Da, chi_da0, chi_dainf, Da_c):
    Da = np.asarray(Da, dtype=float)
    return chi_dainf + (chi_da0 - chi_dainf) / (1.0 + Da / Da_c)


def chi_branch_log(Da, chi_da0, chi_dainf, logDa_c):
    Da_c = 10.0 ** logDa_c
    return chi_branch_physical(Da, chi_da0, chi_dainf, Da_c)


# =========================================================
# Low-Pe and high-Pe chi branches
# =========================================================
def chi0_model(params0, Da):
    (logDa_c0,) = params0
    return chi_branch_log(Da, CHI_0_DA0, CHI_0_DAINF, logDa_c0)


def chi0_model_physical(Da, Da_c0, constants):
    return chi_branch_physical(
        Da,
        constants["CHI_0_DA0"],
        constants["CHI_0_DAINF"],
        Da_c0,
    )


def chi_inf_model(params_inf, Da):
    (logDa_c_inf,) = params_inf
    return chi_branch_log(Da, CHI_INF_DA0, CHI_INF_DAINF, logDa_c_inf)


def chi_inf_model_physical(Da, Da_c_inf, constants):
    return chi_branch_physical(
        Da,
        constants["CHI_INF_DA0"],
        constants["CHI_INF_DAINF"],
        Da_c_inf,
    )


def residuals_chi0(params0, Da, chi0_data):
    chi_fit = chi0_model(params0, Da)
    if np.any(~np.isfinite(chi_fit)) or np.any(chi_fit <= 0.0) or np.any(chi_fit > 1.05):
        return 1e6 * np.ones_like(chi0_data)
    return chi_fit - chi0_data


def residuals_chi_inf(params_inf, Da, chi_inf_data):
    chi_fit = chi_inf_model(params_inf, Da)
    if np.any(~np.isfinite(chi_fit)) or np.any(chi_fit <= 0.0) or np.any(chi_fit > 1.05):
        return 1e6 * np.ones_like(chi_inf_data)
    return chi_fit - chi_inf_data


def fit_chi0(Da_tab, chi_tab, n_low_pe=5):
    n_low_pe = min(n_low_pe, chi_tab.shape[1])
    chi0_data_all = np.nanmean(chi_tab[:, :n_low_pe], axis=1)

    mask = np.isfinite(Da_tab) & np.isfinite(chi0_data_all) & (Da_tab > 0) & (chi0_data_all > 0)
    if np.count_nonzero(mask) < 2:
        raise RuntimeError("Not enough finite low-Pe asymptote data to fit.")

    Da_fit = Da_tab[mask]
    chi0_data = chi0_data_all[mask]

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
            residuals_chi0,
            p0,
            args=(Da_fit, chi0_data),
            bounds=(lower, upper),
            method="trf",
            loss="soft_l1",
            f_scale=0.001,
            max_nfev=50000,
        )
        if res.cost < best_cost:
            best = res
            best_cost = res.cost

    return best, chi0_data_all


def fit_chi_inf(Da_tab, chi_tab, n_top_pe=5):
    n_top_pe = min(n_top_pe, chi_tab.shape[1])
    chi_inf_data_all = np.nanmean(chi_tab[:, -n_top_pe:], axis=1)

    mask = np.isfinite(Da_tab) & np.isfinite(chi_inf_data_all) & (Da_tab > 0) & (chi_inf_data_all > 0)
    if np.count_nonzero(mask) < 2:
        raise RuntimeError("Not enough finite high-Pe asymptote data to fit.")

    Da_fit = Da_tab[mask]
    chi_inf_data = chi_inf_data_all[mask]

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
            residuals_chi_inf,
            p0,
            args=(Da_fit, chi_inf_data),
            bounds=(lower, upper),
            method="trf",
            loss="soft_l1",
            f_scale=0.001,
            max_nfev=50000,
        )
        if res.cost < best_cost:
            best = res
            best_cost = res.cost

    return best, chi_inf_data_all


# =========================================================
# Pe blending
#
# chi(Pe,Da) = chi_inf(Da) + (chi_0(Da) - chi_inf(Da))/(1 + Pe/Pchi)
# params_blend = [logPchi]
# =========================================================
def low_pe_weight(Pe, logPchi):
    Pe = np.asarray(Pe, dtype=float)
    Pchi = 10.0 ** logPchi

    t = np.log(Pe) - np.log(Pchi)
    t = np.clip(t, -60.0, 60.0)

    return 1.0 / (1.0 + np.exp(t))


def low_pe_weight_physical(Pe, Pchi):
    Pe = np.asarray(Pe, dtype=float)
    t = np.log(Pe) - np.log(Pchi)
    t = np.clip(t, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(t))


def chi_blend_model(params_blend, params0, params_inf, Pe, Da):
    (logPchi,) = params_blend

    chi0 = chi0_model(params0, Da)
    chi_inf = chi_inf_model(params_inf, Da)
    w0 = low_pe_weight(Pe, logPchi)

    return chi_inf + (chi0 - chi_inf) * w0


def chi_blend_model_physical(Pe, Da, pars, constants):
    chi0 = chi0_model_physical(Da, pars["Da_c0"], constants)
    chi_inf = chi_inf_model_physical(Da, pars["Da_c_inf"], constants)
    w0 = low_pe_weight_physical(Pe, pars["Pchi"])

    return chi_inf + (chi0 - chi_inf) * w0


def residuals_chi_blend_absolute(params_blend, params0, params_inf, Pe, Da, chi):
    chi_fit = chi_blend_model(params_blend, params0, params_inf, Pe, Da)

    if np.any(~np.isfinite(chi_fit)) or np.any(chi_fit <= 0.0) or np.any(chi_fit > 1.05):
        return 1e6 * np.ones_like(chi)

    return chi_fit - chi


def residuals_chi_blend_relative(params_blend, params0, params_inf, Pe, Da, chi):
    chi_fit = chi_blend_model(params_blend, params0, params_inf, Pe, Da)

    if np.any(~np.isfinite(chi_fit)) or np.any(chi_fit <= 0.0) or np.any(chi_fit > 1.05):
        return 1e6 * np.ones_like(chi)

    return (chi_fit - chi) / chi


def initial_guesses_for_blend():
    if GEOMETRY == "tube":
        return [[np.log10(x)] for x in [0.25, 0.5, 0.7, 1.0, 1.5, 2.0, 4.0]]
    if GEOMETRY == "plates":
        return [[np.log10(x)] for x in [0.2, 0.4, 0.55, 0.8, 1.0, 1.5, 3.0]]
    # oneplate
    return [[np.log10(x)] for x in [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 3.0]]


def fit_chi_blend(Pe_data, Da_data, chi_data, params0, params_inf, use_relative_residuals=False):
    residual_fun = residuals_chi_blend_relative if use_relative_residuals else residuals_chi_blend_absolute

    guesses = initial_guesses_for_blend()
    lower = np.array([-12.0])
    upper = np.array([12.0])

    best = None
    best_cost = np.inf

    for p0 in guesses:
        res = least_squares(
            residual_fun,
            p0,
            args=(params0, params_inf, Pe_data, Da_data, chi_data),
            bounds=(lower, upper),
            method="trf",
            loss="soft_l1",
            f_scale=0.001,
            max_nfev=50000,
        )
        if res.cost < best_cost:
            best = res
            best_cost = res.cost

    return best


# =========================================================
# Diagnostics and summary formatting
# =========================================================
def unpack_full_precision_params(params0, params_inf, params_blend):
    logDa_c0 = params0[0]
    logDa_c_inf = params_inf[0]
    (logPchi,) = params_blend

    return {
        "Da_c0": 10.0 ** logDa_c0,
        "Da_c_inf": 10.0 ** logDa_c_inf,
        "Pchi": 10.0 ** logPchi,
    }


def round_params_4(pars):
    return {k: round4(v) for k, v in pars.items()}


def fit_quality(chi_true, chi_fit):
    err = chi_fit - chi_true
    abs_err = np.abs(err)
    rel_err = err / chi_true
    abs_rel_err = np.abs(rel_err)

    return {
        "mean_abs_error": np.mean(abs_err),
        "median_abs_error": np.median(abs_err),
        "max_abs_error": np.max(abs_err),
        "rms_error": np.sqrt(np.mean(err**2)),
        "mean_abs_rel_error": np.mean(abs_rel_err),
        "median_abs_rel_error": np.median(abs_rel_err),
        "max_abs_rel_error": np.max(abs_rel_err),
        "rms_rel_error": np.sqrt(np.mean(rel_err**2)),
    }


def worst_point(Pe, Da, chi_true, chi_fit):
    err = chi_fit - chi_true
    abs_err = np.abs(err)
    rel_err = err / chi_true
    abs_rel_err = np.abs(rel_err)
    imax = np.argmax(abs_err)

    return {
        "Pe": Pe[imax],
        "Da": Da[imax],
        "chi_true": chi_true[imax],
        "chi_fit": chi_fit[imax],
        "error": err[imax],
        "abs_error": abs_err[imax],
        "rel_error": rel_err[imax],
        "abs_rel_error": abs_rel_err[imax],
    }


def corner_limits(pars_round, constants_round):
    Pe_small = 1e-30
    Pe_big = 1e30
    Da_small = 1e-30
    Da_big = 1e30

    return {
        "Pe0_Da0": chi_blend_model_physical(Pe_small, Da_small, pars_round, constants_round),
        "Pe0_Dainf": chi_blend_model_physical(Pe_small, Da_big, pars_round, constants_round),
        "Peinf_Da0": chi_blend_model_physical(Pe_big, Da_small, pars_round, constants_round),
        "Peinf_Dainf": chi_blend_model_physical(Pe_big, Da_big, pars_round, constants_round),
    }


def add_key_values(lines, title, values, float_fmt=".12g"):
    lines.append("")
    lines.append(title)
    lines.append("-" * len(title))
    for key, val in values.items():
        if isinstance(val, (float, np.floating)):
            lines.append(f"{key:24s} = {val:{float_fmt}}")
        else:
            lines.append(f"{key:24s} = {val}")


def build_summary_text(
    config,
    args,
    data_paths,
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

    lines.append("Chi correlation fit summary")
    lines.append("===========================")
    lines.append(f"geometry                = {args.geometry}")
    lines.append(f"label                   = {config['label']}")
    lines.append(f"chi file                = {data_paths['chi']}")
    lines.append(f"Pe file                 = {data_paths['pe']}")
    lines.append(f"Da file                 = {data_paths['da']}")
    lines.append(f"valid fitted points     = {n_valid_points}")
    lines.append(f"n_top_pe                = {args.n_top_pe}")
    lines.append(f"n_low_pe                = {args.n_low_pe}")
    lines.append(f"blend residuals         = {'relative' if args.relative_residuals else 'absolute'}")

    lines.append("")
    lines.append("Fitted structure")
    lines.append("----------------")
    lines.append("chi_0(Da)   = CHI_0_DAINF   + (CHI_0_DA0   - CHI_0_DAINF  )/(1 + Da/Da_c0)")
    lines.append("chi_inf(Da) = CHI_INF_DAINF + (CHI_INF_DA0 - CHI_INF_DAINF)/(1 + Da/Da_c_inf)")
    lines.append("chi(Pe,Da)  = chi_inf(Da) + (chi_0(Da) - chi_inf(Da))/(1 + Pe/Pchi)")

    exact_constants = {
        "CHI_0_DA0": CHI_0_DA0,
        "CHI_0_DAINF": CHI_0_DAINF,
        "CHI_INF_DA0": CHI_INF_DA0,
        "CHI_INF_DAINF": CHI_INF_DAINF,
    }
    add_key_values(lines, "Exact constants", exact_constants)
    add_key_values(lines, "Full-precision fitted parameters", pars_full)

    rounded_all = dict(constants_round)
    rounded_all.update(pars_round)
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

    import pathlib
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output or str(output_dir / f"{prefix}_fit_summary.txt")

    Pe_tab, Da_tab, chi_tab = load_data(config, data_dir=args.data_dir)

    Pe_grid, Da_grid = np.meshgrid(Pe_tab, Da_tab)

    Pe_data = Pe_grid.ravel()
    Da_data = Da_grid.ravel()
    chi_data = chi_tab.ravel()

    mask = (
        np.isfinite(Pe_data)
        & np.isfinite(Da_data)
        & np.isfinite(chi_data)
        & (Pe_data > 0)
        & (Da_data > 0)
        & (chi_data > 0)
    )

    Pe_data = Pe_data[mask]
    Da_data = Da_data[mask]
    chi_data = chi_data[mask]

    # Step 1: fit low-Pe branch
    res0, _chi0_data = fit_chi0(Da_tab, chi_tab, n_low_pe=args.n_low_pe)
    params0 = res0.x

    # Step 2: fit high-Pe branch
    res_inf, _chi_inf_data = fit_chi_inf(Da_tab, chi_tab, n_top_pe=args.n_top_pe)
    params_inf = res_inf.x

    # Step 3: fit Pe blend
    res_blend = fit_chi_blend(
        Pe_data,
        Da_data,
        chi_data,
        params0=params0,
        params_inf=params_inf,
        use_relative_residuals=args.relative_residuals,
    )
    params_blend = res_blend.x

    # Full-precision and rounded parameters
    pars_full = unpack_full_precision_params(params0, params_inf, params_blend)
    pars_round = round_params_4(pars_full)
    constants_round = rounded_constants()

    # Errors from full-precision formula
    chi_fit_full = chi_blend_model(params_blend, params0, params_inf, Pe_data, Da_data)
    quality_full = fit_quality(chi_data, chi_fit_full)
    worst_full = worst_point(Pe_data, Da_data, chi_data, chi_fit_full)

    # Errors from rounded formula
    chi_fit_rounded = chi_blend_model_physical(Pe_data, Da_data, pars_round, constants_round)
    quality_rounded = fit_quality(chi_data, chi_fit_rounded)
    worst_rounded = worst_point(Pe_data, Da_data, chi_data, chi_fit_rounded)

    # Corner limits from rounded formula
    corner_round = corner_limits(pars_round, constants_round)

    data_paths = {
        "chi": path_in_data_dir(args.data_dir, config["chi_file"]),
        "pe": path_in_data_dir(args.data_dir, config["pe_file"]),
        "da": path_in_data_dir(args.data_dir, config["da_file"]),
    }

    summary = build_summary_text(
        config=config,
        args=args,
        data_paths=data_paths,
        pars_full=pars_full,
        pars_round=pars_round,
        constants_round=constants_round,
        quality_full=quality_full,
        quality_rounded=quality_rounded,
        worst_full=worst_full,
        worst_rounded=worst_rounded,
        corner_round=corner_round,
        n_valid_points=len(chi_data),
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"Saved fitted parameters and errors to {output_file}")


if __name__ == "__main__":
    main()
