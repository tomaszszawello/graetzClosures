"""Compact Sh and chi correlations from Szawełło & Szymczak (2026).

All Sherwood numbers use the hydraulic diameter. The same expressions apply
to heat transfer by replacing Sh → Nu and Da → Bi.

Dimensionless groups
--------------------
Circular tube (radius R):
    Pe = u_bar R / D,   Da = k R / D,   Sh = 2R j_w / [D (C_m - C_w)]

Parallel plates, two reactive walls (half-gap a):
    Pe = u_bar a / D,   Da = k a / D,   Sh = 4a j_w / [D (C_m - C_w)]

Parallel plates, one reactive wall (gap b):
    Pe = u_bar b / D,   Da = k b / D,   Sh = 2b j_w / [D (C_m - C_w)]
"""

from __future__ import annotations

import warnings

import numpy as np

# ---------------------------------------------------------------------------
# Coefficient tables — taken directly from Appendix B of the manuscript
# ---------------------------------------------------------------------------

# Plug-flow Sh (Appendix B, Table B.1):
#   Sh(Da) = sh_inf + (sh0 - sh_inf) / (1 + Da/da_c)
_PLUG_SH: dict[str, dict[str, float]] = {
    "tube":     {"sh0": 8.0,      "sh_inf": 5.7832, "da_c": 2.7783},
    "plates":   {"sh0": 12.0,     "sh_inf": 9.8696, "da_c": 2.4008},
    "oneplate": {"sh0": 6.0,      "sh_inf": 4.9348, "da_c": 2.4008},
}

# Poiseuille-flow Sh (Appendix B, Eqs. (B.5)–(B.9) and Table B.2):
#
#   Sh(Pe, Da) = sh_inf(Da) + (sh0(Da) - sh_inf(Da)) / (1 + (Pe/pe_c(Da))^(4/3))
#
#   sh0(Da)    = sh00_inf  + (sh00   - sh00_inf)   / (1 + Da/da_c0)
#   sh_inf(Da) = shinf_inf + (shinf0 - shinf_inf)  / (1 + Da/da_c_inf)
#   pe_c(Da)   = pec_inf   - (pec_inf - pec0)      / (1 + (Da/da_c_pe)^(2/3))
_POIS_SH: dict[str, dict[str, float]] = {
    "tube": {
        "sh00": 6.0,      "sh00_inf": 4.1807, "da_c0":    2.6886,
        "shinf0": 4.3636, "shinf_inf": 3.6568, "da_c_inf": 1.8848,
        "pec0": 0.0108,   "pec_inf": 0.9728,   "da_c_pe":  0.5687,
    },
    "plates": {
        "sh00": 10.0,     "sh00_inf": 8.1174, "da_c0":    2.3607,
        "shinf0": 8.2353, "shinf_inf": 7.5410, "da_c_inf": 1.9189,
        "pec0": 0.0091,   "pec_inf": 0.7853,   "da_c_pe":  0.6338,
    },
    "oneplate": {
        "sh00": 5.7143,   "sh00_inf": 4.7282, "da_c0":    2.4053,
        "shinf0": 5.3846, "shinf_inf": 4.8607, "da_c_inf": 2.4858,
        "pec0": 0.0100,   "pec_inf": 1.0710,   "da_c_pe":  0.8747,
    },
}

# Poiseuille-flow chi (Appendix B, Eqs. (B.10)–(B.11) and Table B.3):
#
#   chi(Pe, Da) = chi_inf(Da) + (chi0(Da) - chi_inf(Da)) / (1 + Pe/pec_chi)
#
#   chi0(Da)    = chi0_inf  + (1 - chi0_inf)  / (1 + Da/da_c_chi0)
#   chi_inf(Da) = chinf_inf + (1 - chinf_inf) / (1 + Da/da_c_chinf)
_POIS_CHI: dict[str, dict[str, float]] = {
    "tube":     {"chi0_inf": 0.7229, "da_c_chi0": 2.8167,
                 "chinf_inf": 0.7095, "da_c_chinf": 2.1190, "pec_chi": 0.6874},
    "plates":   {"chi0_inf": 0.8225, "da_c_chi0": 2.4109,
                 "chinf_inf": 0.8171, "da_c_chinf": 2.0398, "pec_chi": 0.5367},
    "oneplate": {"chi0_inf": 0.9581, "da_c_chi0": 2.2750,
                 "chinf_inf": 0.9459, "da_c_chinf": 2.3928, "pec_chi": 0.8772},
}

# Ratio Sh_Dh / Sh_ell for each geometry (hydraulic diameter / length scale).
_HYDRAULIC_MULTIPLIER: dict[str, float] = {
    "tube": 2.0,      # Dh = 2R
    "plates": 4.0,    # Dh = 4a
    "oneplate": 2.0,  # Dh = 2b
}

_VALID_GEOMETRIES = frozenset({"tube", "plates", "oneplate"})
_VALID_FLOWS = frozenset({"poiseuille", "plug"})


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _validate(geometry: str, flow: str) -> None:
    if geometry not in _VALID_GEOMETRIES:
        raise ValueError(
            f"geometry must be one of {sorted(_VALID_GEOMETRIES)!r}, got {geometry!r}"
        )
    if flow not in _VALID_FLOWS:
        raise ValueError(
            f"flow must be one of {sorted(_VALID_FLOWS)!r}, got {flow!r}"
        )


def _check_range(Pe, Da, flow: str) -> None:
    """Warn when inputs fall outside the fitted correlation window (1e-3 ≤ Pe, Da ≤ 1e3)."""
    if Da is not None:
        Da_arr = np.asarray(Da, dtype=float)
        if np.any((Da_arr < 1e-3) | (Da_arr > 1e3)):
            warnings.warn(
                "The correlations were fitted for 1e-3 ≤ Da ≤ 1e3; "
                "this call extrapolates outside that range.",
                RuntimeWarning,
                stacklevel=3,
            )

    if flow == "poiseuille" and Pe is not None:
        Pe_arr = np.asarray(Pe, dtype=float)
        if np.any((Pe_arr < 1e-3) | (Pe_arr > 1e3)):
            warnings.warn(
                "The correlations were fitted for 1e-3 ≤ Pe ≤ 1e3; "
                "this call extrapolates outside that range.",
                RuntimeWarning,
                stacklevel=3,
            )
        if Da is not None:
            Da_arr = np.asarray(Da, dtype=float)
            # singular weak-exchange corner: Da << Pe^2 << 1
            if np.any((Pe_arr < 0.1) & (Da_arr < Pe_arr ** 2)):
                warnings.warn(
                    "This call lies in the singular weak-exchange corner "
                    "Da ≪ Pe² ≪ 1, which is not well resolved by the compact correlations.",
                    RuntimeWarning,
                    stacklevel=3,
                )


def _to_scalar_or_array(x: np.ndarray):
    return x.item() if x.ndim == 0 else x


# ---------------------------------------------------------------------------
# Plug-flow Sh
# ---------------------------------------------------------------------------
def _sh_plug(Da: np.ndarray, geometry: str) -> np.ndarray:
    p = _PLUG_SH[geometry]
    return p["sh_inf"] + (p["sh0"] - p["sh_inf"]) / (1.0 + Da / p["da_c"])


# ---------------------------------------------------------------------------
# Poiseuille-flow Sh
# ---------------------------------------------------------------------------
def _sh_pois(Pe: np.ndarray, Da: np.ndarray, geometry: str) -> np.ndarray:
    p = _POIS_SH[geometry]
    sh0 = p["sh00_inf"] + (p["sh00"] - p["sh00_inf"]) / (1.0 + Da / p["da_c0"])
    sh_inf = p["shinf_inf"] + (p["shinf0"] - p["shinf_inf"]) / (1.0 + Da / p["da_c_inf"])
    pe_c = p["pec_inf"] - (p["pec_inf"] - p["pec0"]) / (
        1.0 + (Da / p["da_c_pe"]) ** (2.0 / 3.0)
    )
    return sh_inf + (sh0 - sh_inf) / (1.0 + (Pe / pe_c) ** (4.0 / 3.0))


# ---------------------------------------------------------------------------
# Poiseuille-flow chi
# ---------------------------------------------------------------------------
def _chi_pois(Pe: np.ndarray, Da: np.ndarray, geometry: str) -> np.ndarray:
    p = _POIS_CHI[geometry]
    chi0 = p["chi0_inf"] + (1.0 - p["chi0_inf"]) / (1.0 + Da / p["da_c_chi0"])
    chi_inf = p["chinf_inf"] + (1.0 - p["chinf_inf"]) / (1.0 + Da / p["da_c_chinf"])
    return chi_inf + (chi0 - chi_inf) / (1.0 + Pe / p["pec_chi"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def sh(Pe=None, Da=None, geometry: str = "tube", flow: str = "poiseuille",
       check_range: bool = True):
    """Hydraulic-diameter Sherwood number from the compact correlation.

    Parameters
    ----------
    Pe : float or array-like, optional
        Péclet number. Not required for plug flow.
    Da : float or array-like
        Damköhler number.
    geometry : {"tube", "plates", "oneplate"}
        Duct geometry (default: "tube").
    flow : {"poiseuille", "plug"}
        Velocity profile (default: "poiseuille").
    check_range : bool
        If True (default), emit a RuntimeWarning when Pe or Da fall outside
        the fitted window 1e-3 ≤ Pe, Da ≤ 1e3 or into the singular corner
        Da ≪ Pe² ≪ 1.

    Returns
    -------
    float or ndarray
    """
    _validate(geometry, flow)
    if Da is None:
        raise ValueError("Da is required.")
    if check_range:
        _check_range(Pe, Da, flow)
    Da = np.asarray(Da, dtype=float)
    if flow == "plug":
        return _to_scalar_or_array(_sh_plug(Da, geometry))
    if Pe is None:
        raise ValueError("Pe is required for Poiseuille flow.")
    Pe = np.asarray(Pe, dtype=float)
    return _to_scalar_or_array(_sh_pois(Pe, Da, geometry))


def sh_l(Pe=None, Da=None, geometry: str = "tube", flow: str = "poiseuille",
         check_range: bool = True):
    """Characteristic-length Sherwood number (Sh_ell = Sh_Dh / multiplier).

    This is the form used in the 1D averaged transport closure (Eq. (17) of
    the manuscript). The relationship to the hydraulic-diameter Sh returned
    by sh() is:

        Sh_ell = Sh_Dh / 2   (tube, oneplate)
        Sh_ell = Sh_Dh / 4   (plates)

    All parameters are identical to sh().
    """
    sh_dh = sh(Pe=Pe, Da=Da, geometry=geometry, flow=flow, check_range=check_range)
    return sh_dh / _HYDRAULIC_MULTIPLIER[geometry]


def chi(Pe=None, Da=None, geometry: str = "tube", flow: str = "poiseuille",
        check_range: bool = True):
    """Averaging factor chi = C_a / C_m from the compact correlation.

    For plug flow, chi = 1 exactly (independent of Pe and Da). If Pe or Da
    are supplied as arrays for plug flow, an array of ones with the
    corresponding broadcast shape is returned.

    Parameters
    ----------
    Pe : float or array-like, optional
        Péclet number. Ignored for plug flow.
    Da : float or array-like, optional
        Damköhler number. Ignored for plug flow.
    geometry : {"tube", "plates", "oneplate"}
        Duct geometry (default: "tube").
    flow : {"poiseuille", "plug"}
        Velocity profile (default: "poiseuille").
    check_range : bool
        If True (default), emit a RuntimeWarning when inputs fall outside
        the fitted window or into the singular corner.

    Returns
    -------
    float or ndarray
        For plug flow with scalar inputs, returns 1.0.
        For plug flow with array inputs, returns an array of ones.
    """
    _validate(geometry, flow)
    if flow == "plug":
        arrs = [np.asarray(x, dtype=float) for x in (Pe, Da) if x is not None]
        if arrs:
            out = np.ones(np.broadcast(*arrs).shape, dtype=float)
            return _to_scalar_or_array(out)
        return 1.0
    if check_range:
        _check_range(Pe, Da, flow)
    if Pe is None:
        raise ValueError("Pe is required for Poiseuille flow.")
    if Da is None:
        raise ValueError("Da is required for Poiseuille flow.")
    Pe = np.asarray(Pe, dtype=float)
    Da = np.asarray(Da, dtype=float)
    return _to_scalar_or_array(_chi_pois(Pe, Da, geometry))
