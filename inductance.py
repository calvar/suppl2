"""Closed-form self-inductance of a rectangular circuit of round wire, with
one side (the bar) free to slide, and its gradient with respect to the
bar's position.

Every function below is a direct transcription of the corresponding
closed form in the paper and its companion derivations supplement, for a
general wire radius d. Each was checked against an independent
implementation of the same formulas, and the flux integrals they
represent are re-derived from the raw field by numerical quadrature in
Notebook 2 (correctness tests), which needs nothing outside this
directory.

Conventions, matching the paper: mu is the vacuum permeability, l the
(fixed) circuit width, x the bar's position, d the wire radius. The
"reduced units" used throughout the paper's numerical sections fix
d=1, mu=1, M=1; every function here keeps d and mu explicit so the same
code serves both the general appendix-level checks (Notebook 2) and the
reduced-units dynamics (Notebook 1, via dynamics.py).

Every function below is decorated with @njit(cache=True) via the
soft-import shim in _jit.py: with numba installed this compiles the hot
scalar math to machine code (needed for a full-resolution reproduction of
the paper's figures in reasonable time, see README.md); without it,
`njit` is a no-op and every function is still correct, plain, uncompiled
Python/numpy -- just slower.
"""

import numpy as np

from _jit import njit


@njit(cache=True)
def fnorm(x, y):
    return np.sqrt(x**2 + y**2)


@njit(cache=True)
def chi(x, y):
    return x / fnorm(x, y)


@njit(cache=True)
def zeta(x, y):
    sqr = fnorm(x, y)
    frac = (sqr - x) / (sqr + x)
    return np.log(frac) / 2


@njit(cache=True)
def eta(x, y):
    return x * zeta(x, y) + y * zeta(y, x)


@njit(cache=True)
def resistance(rho, d, l, x):
    A = np.pi * d**2
    R = rho * 2 * (l + x) / A
    return R


@njit(cache=True)
def self_inductance_outwire(mu, d, l, x):
    c1 = (x - d) * (zeta(x - d, l - d) - zeta(x - d, d))
    c2 = (l - d) * (zeta(l - d, x - d) - zeta(l - d, d))
    c3 = -d * (zeta(d, l - d) + zeta(d, x - d) - 2 * zeta(d, d))
    c4 = 2 * ((x - d) / chi(x - d, l - d) + d / chi(d, d)
              - (x - d) / chi(x - d, d) - d / chi(d, l - d))
    return mu * (c1 + c2 + c3 + c4) / np.pi


@njit(cache=True)
def corner_flux(x, l, d):
    """D(x,l): the flux through the four corners of the circuit (wire
    radius d), as derived in the supplementary derivations."""
    bracket1 = (fnorm(l, d)**3 - l**3 - fnorm(l - d, d)**3 + (l - d)**3
                + 4 * d**3 * (np.sqrt(2) - 1)
                + fnorm(x, d)**3 - x**3 - fnorm(x - d, d)**3 + (x - d)**3) / (3 * d**2)
    term_l = d * (zeta(d, l) - zeta(d, l - d)) + fnorm(d, l) - fnorm(d, l - d) - d
    term_x = d * (zeta(d, x) - zeta(d, x - d)) + fnorm(x, d) - fnorm(x - d, d) - d
    mixed_norm = 2 * (fnorm(x, l) - fnorm(x, l - d) - fnorm(x - d, l) + fnorm(x - d, l - d))
    mixed_eta = eta(x, l) - eta(x, l - d) - eta(x - d, l) + eta(x - d, l - d)
    return bracket1 + term_l + term_x + mixed_norm + mixed_eta


@njit(cache=True)
def phi_selfL(v, d):
    """Helper for the enclosed-current-fraction (r/d)^2 flux-linkage
    correction below. Algebraically equal to
    (2 v^5 + 5 d^2 v^3 - 2(d^2+v^2)^2.5)/d^4, rewritten to avoid the
    catastrophic cancellation of that form at large v (both terms grow
    like v^5 while their difference grows only like v)."""
    R = np.sqrt(d**2 + v**2)
    return (v * R * (v**2 - 2 * d**2) - v**4 - 6 * d**2 * v**2 - 2 * d**4) / ((R + v) * d**2)


@njit(cache=True)
def self_inductance_correction(mu, d, l, x):
    """Delta L(x,l): the additive enclosed-current-fraction weighting term
    (eq. DeltaL in the supplementary derivations)."""
    return mu * (phi_selfL(x, d) + phi_selfL(l, d) + 4 * d) / (15 * np.pi)


@njit(cache=True)
def self_inductance_correction_dx(mu, d, x):
    """d(Delta L)/dx, the gradient of the enclosed-current-fraction term.
    Algebraically equal to
    mu*x*(3 d^2 x + 2x^3 - 2(d^2+x^2)^1.5)/(3 pi d^4), rewritten to avoid
    catastrophic cancellation at large x (see phi_selfL above)."""
    R = np.sqrt(d**2 + x**2)
    return -mu * x * (x + 2 * R) / (3 * np.pi * (R + x)**2)


@njit(cache=True)
def self_inductance_inwire(mu, d, l, x):
    D = corner_flux(x, l, d)

    A1 = (eta(x - d, d) - eta(x - d, l - d) + eta(x, l - d) - eta(x, d)
          + 2 * (fnorm(x, l - d) + fnorm(x - d, d) - fnorm(x - d, l - d) - fnorm(x, d)) + D / 2)
    B1 = d * (zeta(d, l - d) - zeta(d, d)) + fnorm(d, l - d) - fnorm(d, d) + d - (l - d)
    C1 = (fnorm(l - d, d)**3 - (l - d)**3) / (3 * d**2) + (d - fnorm(d, d)**3 / d**2) / 3

    A2 = (eta(x - d, l) - eta(x - d, l - d) + eta(l - d, d) - eta(l, d)
          + 2 * (fnorm(l - d, d) + fnorm(x - d, l) - fnorm(x - d, l - d) - fnorm(l, d)) + D / 2)
    B2 = d * (zeta(d, x - d) - zeta(d, d)) + fnorm(x - d, d) - fnorm(d, d) + d - (x - d)
    C2 = (fnorm(x - d, d)**3 - (x - d)**3) / (3 * d**2) + (d - fnorm(d, d)**3 / d**2) / 3

    naive = mu * (A1 + B1 + C1 + A2 + B2 + C2) / np.pi
    return naive + self_inductance_correction(mu, d, l, x)


@njit(cache=True)
def self_inductance(mu, d, l, x):
    """L(x,l): the full self-inductance, outside- plus
    inside-the-wire contributions (the latter already including the
    enclosed-current-fraction weighting)."""
    return self_inductance_outwire(mu, d, l, x) + self_inductance_inwire(mu, d, l, x)


@njit(cache=True)
def self_inductance_dx(mu, d, l, x):
    """dL/dx: total gradient (outside- and inside-the-wire
    contributions combined), closed form."""
    term = zeta(x, l) - zeta(x, d) + 1 / chi(x, l) + (x / d)**2 * (1 / chi(x, d) - 1) - 1
    naive = mu * term / np.pi
    return naive + self_inductance_correction_dx(mu, d, x)


def self_inductance_asymp(mu, d, l, x):
    """L(x,l) as x -> infinity: linear in x with slope
    (mu/pi)(ln|l/d| + 1/4)."""
    r = np.sqrt(d**2 + l**2)
    slope = (mu / np.pi) * (np.log(np.abs(l / d)) + 0.25)
    const = (mu / (3 * np.pi * d**2)) * (
        4 * d**3 - 3 * d**2 * l - l**3 - 3 * d**2 * l * zeta(l, d) + (l**2 - 2 * d**2) * r
    )
    const += mu * (phi_selfL(l, d) + 4 * d) / (15 * np.pi)
    return slope * x + const


def self_inductance_dx_asymp(mu, d, l):
    """dL/dx as x -> infinity: the constant slope alone."""
    return (mu / np.pi) * (np.log(np.abs(l / d)) + 0.25)


@njit(cache=True)
def force(l, B, I, mu, d, x):
    """Magnetic force on the bar, eqs. (xforce)/(acc1): the braking term
    plus the inductance-gradient thrust. Returns (f0, f1) separately."""
    f0 = -l * B * I
    f1 = self_inductance_dx(mu, d, l, x) * I**2 / 2
    return f0, f1
