"""graetz_closures — compact Sh and chi correlations for the extended Graetz problem.

Basic usage
-----------
    import graetz_closures as gc

    gc.sh(Pe=10, Da=100)                              # tube, Poiseuille (defaults)
    gc.sh(Da=10, flow="plug")                         # tube, plug flow
    gc.sh(Pe=10, Da=100, geometry="plates")           # parallel plates, Poiseuille
    gc.sh_l(Pe=10, Da=100)                            # characteristic-length Sh (for 1D closure)
    gc.chi(Pe=10, Da=100, geometry="oneplate")        # one reactive plate, Poiseuille

    import numpy as np
    Pe = np.logspace(-2, 3, 200)
    sh_array = gc.sh(Pe, Da=50, geometry="tube")      # vectorised

Reference
---------
T. Szawełło and P. Szymczak, "Extended Graetz problem in laminar duct flows
with Robin boundary conditions: Sherwood/Nusselt correlations and averaged
transport closures", International Journal of Heat and Mass Transfer (submitted).
"""

from graetz_closures._correlations import sh, sh_l, chi

__all__ = ["sh", "sh_l", "chi"]
__version__ = "0.1.0"
