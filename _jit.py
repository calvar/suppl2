"""Optional numba acceleration, shared by inductance.py and dynamics.py.

numba is NOT required: every function in this supplement is plain
numpy/Python and produces identical results without it. But the hot loop
in dynamics.verlet_integrate calls the closed-form self-inductance formulas
several times per step, and a full production run (paper: 1e6-4e6 steps per
trajectory, or a few dozen ring-downs for the eps sweep) is impractically
slow in pure, uncompiled Python -- see the runtime note in README.md. If numba
is installed, `njit` below is the real @numba.njit(cache=True) decorator;
if not, it is a no-op that leaves every function as plain Python, just
slower.
"""

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def wrap(f):
            return f
        return wrap
