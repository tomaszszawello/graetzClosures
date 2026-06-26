# graetzClosures
<<<<<<< HEAD

Companion code for the manuscript

> **Extended Graetz problem in laminar duct flows with Robin boundary conditions:
> Sherwood/Nusselt correlations and averaged transport closures**
> T. Szawełło and P. Szymczak
> *submitted to International Journal of Heat and Mass Transfer*

The repository contains eigenvalue solvers, correlation-fitting scripts, and a
standalone tube solver that reproduce all numerical results and figures in the
manuscript. It also ships a small installable Python package,
**`graetz-closures`**, that exposes the final correlations directly.

---

## Quick start — using the correlations

```bash
pip install .
```

```python
import graetz_closures as gc

# Sherwood number — Poiseuille flow (default)
gc.sh(Pe=10, Da=100)                              # circular tube
gc.sh(Pe=10, Da=100, geometry="plates")           # parallel plates, two reactive walls
gc.sh(Pe=10, Da=100, geometry="oneplate")         # parallel plates, one reactive wall

# Sherwood number — plug flow (Pe not needed; chi = 1 exactly)
gc.sh(Da=10, flow="plug")
gc.sh(Da=10, flow="plug", geometry="plates")

# Averaging factor chi = C_a / C_m — Poiseuille flow only
gc.chi(Pe=10, Da=100)
gc.chi(Pe=10, Da=100, geometry="plates")
gc.chi(Pe=10, Da=100, geometry="oneplate")

# Fully vectorised — pass numpy arrays for Pe and/or Da
import numpy as np
Pe = np.logspace(-2, 3, 300)
sh_curve  = gc.sh(Pe, Da=50, geometry="tube")     # shape (300,)
chi_curve = gc.chi(Pe, Da=50, geometry="tube")    # shape (300,)

Pe_grid, Da_grid = np.meshgrid(np.logspace(-1, 2, 50), np.logspace(-1, 2, 50))
sh_map = gc.sh(Pe_grid, Da_grid, geometry="plates")   # shape (50, 50)
```

**Geometry options** (`geometry=`, default `"tube"`):

| Value | Geometry | Length scale | Hydraulic diameter |
|---|---|---|---|
| `"tube"` | Circular tube | radius R | Dh = 2R |
| `"plates"` | Parallel plates, two reactive walls | half-gap a | Dh = 4a |
| `"oneplate"` | Parallel plates, one reactive wall | gap b | Dh = 2b |

**Flow options** (`flow=`, default `"poiseuille"`):

| Value | Profile | Notes |
|---|---|---|
| `"poiseuille"` | Poiseuille | Sh and chi both depend on Pe and Da |
| `"plug"` | Plug flow | Sh depends on Da only; chi = 1 exactly |

For heat transfer, substitute Nu for Sh and Bi for Da — the correlation expressions are identical.

---

## Problem overview

The code addresses the steady extended Graetz problem in two canonical duct
geometries under Poiseuille flow:

- **Circular tube** — one reactive cylindrical wall
- **Parallel plates** — two reactive walls (symmetric) or one reactive / one inert wall

The wall condition is a Robin (third-kind) boundary condition parameterised by
the Damköhler number Da = k R / D (or k a / D for plates), which interpolates
between the impermeable limit Da → 0 and the Dirichlet limit Da → ∞.

For each geometry the code computes:

- The dominant eigenvalue β₁(Pe, Da) and the resulting hydraulic-diameter
  Sherwood number **Sh(Pe, Da)**
- The cross-sectional averaging factor **χ(Pe, Da)** = C_a / C_m
- The entrance length **Le(Pe, Da)**
- Compact two-parameter correlations for Sh and χ and their fitted coefficients

The standalone `tube_solver.py` also solves the full 2D axisymmetric
advection–diffusion equation and compares it against the correlated averaged
model, producing concentration profiles and decay-rate diagnostics.

---

## Repository layout

```
.
├── data/                              # Input grids and small reference data
│   ├── Pe.txt                         # Pe grid (shared by all geometries)
│   ├── Da.txt                         # Da grid (shared by all geometries)
│   ├── Sh_tube_plug.txt               # Plug-flow Sh(Da), tube
│   ├── Sh_plates_plug.txt             # Plug-flow Sh(Da), two-wall plates
│   └── ...                            # (large Poiseuille tables excluded — see below)
│
├── fits/                              # Fit summaries (Sh and χ coefficients)
│   ├── tube_pois_fit_summary.txt
│   ├── tube_pois_chi_fit_summary.txt
│   └── ...
│
├── figures/                           # Publication figures
│   ├── sh_pois_tube.png
│   ├── sh_pois_plates.png
│   ├── sh_vs_da.png
│   └── tube_Pe10_Da100.png
│
├── tube_poiseuille_parallel_fast.py   # Eigenvalue sweep — circular tube
├── plates_poiseuille_parallel_fast.py # Eigenvalue sweep — parallel plates (2 walls)
├── plates_onewall_poiseuille_parallel_fast.py  # Eigenvalue sweep — one-wall plates
├── sherwood_plug.py                   # Plug-flow Sh and χ (Pe-independent)
├── sherwood_fit.py                    # Fit Sh(Pe, Da) correlation
├── chi_fit.py                         # Fit χ(Pe, Da) correlation
├── plot_full_diagrams.py              # Full (Pe, Da) maps of Sh, χ, Le
├── plot_sh_vs_da.py                   # Sh(Da) curves for tube and plates
├── tube_solver.py                     # 2D tube solver + closure comparison
├── requirements.txt
└── README.md
```

---

## Dependencies

Python 3.9+ and the packages listed in `requirements.txt`:

```
pip install -r requirements.txt
```

| Package | Purpose |
|---|---|
| NumPy | array operations |
| SciPy | ODE solvers, root finding, least-squares fitting |
| Matplotlib | all figures |

---

## Reproducing the results

The large Poiseuille eigenvalue tables (`Sh_*_pois.txt`, `chi_*_pois.txt`,
`Le_*_pois.txt`, ~25 MB each) are not tracked in Git because they are
regenerated by the scripts below. All other data and fit files are included.

### 1 — Plug-flow reference curves (fast, seconds)

```bash
python sherwood_plug.py tube --output-dir data
python sherwood_plug.py plates --output-dir data
python sherwood_plug.py oneplate --output-dir data
```

Writes `data/Sh_tube_plug.txt`, `data/Sh_plates_plug.txt`, and
`data/Sh_oneplate_plug.txt`. Add `--with-entrance` to also save
entrance-length estimates versus Da.

### 2 — Poiseuille eigenvalue sweeps (parallel, minutes to hours)

```bash
python tube_poiseuille_parallel_fast.py
python plates_poiseuille_parallel_fast.py
python plates_onewall_poiseuille_parallel_fast.py
```

Each script sweeps the full (Pe, Da) grid and writes the corresponding
`Sh_*_pois.txt`, `chi_*_pois.txt`, and `Le_*_pois.txt` files into `data/`.
The number of parallel workers is controlled by the `--workers` flag (defaults
to the number of CPU cores).

### 3 — Fit the Sh and χ correlations

```bash
python sherwood_fit.py tube
python sherwood_fit.py plates
python sherwood_fit.py oneplate

python chi_fit.py tube
python chi_fit.py plates
python chi_fit.py oneplate
```

Results are written to `fits/` by default. Both scripts read from `data/` by
default; use `--data-dir` to override. Use `--output-dir` to write elsewhere,
or `--output` for a specific file path.

### 4 — Figures

Full (Pe, Da) maps:

```bash
python plot_full_diagrams.py tube
python plot_full_diagrams.py plates
python plot_full_diagrams.py oneplate
```

Sh(Da) comparison (tube and plates, plug vs. Poiseuille):

```bash
python plot_sh_vs_da.py
```

Figures are saved to `figures/`.

### 5 — Standalone tube solver

Solves the 2D axisymmetric problem for a single (Pe, Da) pair, evaluates the
correlated closure, and saves profiles, diagnostics, and a concentration plot:

```bash
python tube_solver.py --Pe 10 --Da 100 --Lambda 5 --output-dir data
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--Pe` | 10 | Péclet number Pe = ū R / D |
| `--Da` | 100 | Damköhler number Da = k R / D |
| `--Lambda` | 5 | Domain length L / R |
| `--Nr` | 50 | Radial cells |
| `--dz` | 0.025 | Target axial spacing (if `--nz` omitted) |
| `--output-dir` | `tube_solver_output` | Output directory |
| `--no-plots` | — | Skip the PNG figure |

Each run writes four files:
- `{prefix}_profiles.csv` — axial concentration profiles
- `{prefix}_profiles.txt` — same, space-delimited (numpy format)
- `{prefix}_parameters_errors.json` — all scalar diagnostics
- `{prefix}_parameters.txt` — same, key = value text format

---

## Dimensionless groups and notation

| Symbol | Definition | Notes |
|---|---|---|
| Pe | ū R / D (tube) or ū a / D (plates) | Péclet number |
| Da | k R / D (tube) or k a / D (plates) | Damköhler number |
| Sh | 2R j_w / [D (C_m − C_w)] | Hydraulic-diameter Sherwood number |
| χ | C_a / C_m | Cross-sectional averaging factor |
| Z | z / R (tube) or z / a (plates) | Dimensionless axial coordinate |
| β | downstream decay rate, C_m ~ exp(−β Z) | Dominant eigenvalue |

---

## Citation

If you use this code, please cite the manuscript once it is published. Until
then, please contact the authors.

---

## License

MIT License — see [LICENSE](LICENSE).
=======
>>>>>>> 10000252822783cfc8aa8c52431839d15f2e42e5
