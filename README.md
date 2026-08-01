<!--
SUBMISSION METADATA
  Title (27 chars, limit 30): Sliding-bar simulation code
  Description (28 words, limit 30):
    Python modules and Jupyter notebooks reproducing every figure in the
    paper and re-running every numerical and symbolic correctness check
    the paper reports, with runnable, documented source for each.
-->

# Sliding-bar simulation code

Second supplementary item for the paper. This is a standalone code
package: the article, this directory and a Python install are everything
needed to run it.

## What's here

Four importable modules, holding the physics with no notebook-only
state:

| module | contents |
|---|---|
| `inductance.py` | Closed-form self-inductance $L(x,l)$, its gradient $dL/dx$, and the enclosed-current-fraction correction $\Delta L$, for a general wire radius $d$ |
| `field.py` | The nine-region piecewise induced field $f_B(p,h;x)$ the self-inductance is the flux integral of, and its enclosed-current-fraction-weighted analogue — appendix-level building blocks, used only by Notebook 2 |
| `dynamics.py` | The velocity-Verlet (kick-drift-kick) integrator for the equations of motion, in the paper's reduced units ($d=1$, $M=1$) |
| `models.py` | The two limiting-case analytic models the paper compares the simulation against (the no-self-inductance solution and the damped-oscillator/RLC model), and the `curve_fit` wrapper used to fit the second one |
| `figures.py` | Plotting helpers, one per reproduced figure |
| `_jit.py` | The optional-`numba` shim shared by `inductance.py`/`dynamics.py` (see **Performance** below) |

and two notebooks:

- **`notebook1_numerical_experiments.ipynb`** — runs the simulation and
  reproduces the paper's seven computed figures, with the paper's own
  parameter values. Lets you vary $B_0$, $\rho$ and the initial
  conditions and re-run.
- **`notebook2_correctness_tests.ipynb`** — re-runs every numerical or
  symbolic correctness check the paper states it performed (e.g. "checked
  against direct numerical integration", "verified against a
  finite-difference derivative"), quoting the paper's own sentence next
  to each measured value and its threshold, ending in an explicit
  PASS/FAIL.

The paper's two schematic diagrams — the circuit itself, and the field of
a current element — are hand drawn rather than computed, so they have no
corresponding figure or check here.

## How to run

```bash
pip install -r requirements.txt   # numpy, scipy, mpmath, matplotlib;
                                   # numba and jupyter are optional/recommended
jupyter notebook notebook1_numerical_experiments.ipynb
jupyter notebook notebook2_correctness_tests.ipynb
```

Both notebooks run top to bottom from a clean kernel with no manual
steps and no dependency on anything outside this directory.

## Performance: `numba` is optional but strongly recommended

Every function here is plain, correct numpy/Python and needs nothing
beyond `requirements.txt`'s four required packages. But `dynamics.py`'s
inner loop calls the closed-form self-inductance several times per
integration step, and a full-resolution run (the paper's own step
counts, or the $17\times17$ parameter-grid scan behind the
goodness-of-fit figure) is impractically slow without JIT
compilation: **a single $10^6$-step trajectory takes about 140 s in
pure Python and well under 1 s with `numba` installed** (`pip install
numba`, no compiler needed) — roughly a 150x difference, confirmed by
direct benchmark on this codebase. `_jit.py` supplies `numba.njit` when
available and a transparent no-op otherwise, so results are identical
either way; only the wall-clock time changes.

Because of this, **Notebook 1 defaults to `FULL_RESOLUTION = False`**: a
fast demo (well under two minutes end to end, with or without `numba`)
that still exercises every figure and every physical regime the paper
does, though a couple of panels are visibly coarser than the paper's own
(documented at each affected cell — the weak-field decay figure only
shows the start of the transient, and the reduced $3\times3$ grid scan is
noisier than the paper's $17\times17$ one). Setting
`FULL_RESOLUTION = True` reproduces the paper's figures at its exact
parameters and step counts; with `numba` installed this finishes in
about 5 minutes (the $17\times17$, $10^6$-step grid scan dominates);
without it, that same scan is on the order of a day. Notebook 2 has no
such switch — every check in it completes in well under a minute
regardless of `numba`.

## Provenance

Every formula in `inductance.py`, `field.py`, `dynamics.py` and
`models.py` is a direct transcription of the paper's own equations, or —
for `field.py` — of the derivations in the companion derivations
supplement. Each was cross-checked against an independent implementation
of the same formulas before being relied on here: every closed-form
self-inductance quantity agrees exactly, and full trajectories agree to
better than $10^{-16}$ relative at every step.

Running without `numba` changes nothing that matters, but it does change
the order in which the compiler evaluates a few sums. The only visible
consequence is in the intermediate corner-flux term at the symmetric
configuration $x=l$, where the uncompiled and compiled evaluations differ
by $5\times10^{-11}$ relative; $L(x,l)$ itself is unaffected there, and
nothing this small reaches a figure or a correctness threshold.

Notebook 2 does not depend on any of this: it re-derives the closed forms
from the raw field by independent numerical quadrature and
extended-precision differentiation, entirely within this directory.

## Data availability

This code is sufficient to regenerate every dataset and figure in the
paper from the stated parameters; no separately-hosted data files are
required.
