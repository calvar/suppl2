"""The two limiting-case analytic models the paper compares the full
simulation against -- the standard treatment that ignores self-inductance,
and the damped-oscillator/RLC-equivalent model -- with the helpers used to
compare them with the simulation. Reduced units throughout (mu=1, d=1,
M=1), matching dynamics.py.

Every expression here is a direct transcription of the corresponding
displayed equation in the paper, and each was checked against an
independent implementation of the same formulas.
"""

import numpy as np

from inductance import (self_inductance, self_inductance_dx,
                        resistance as _resistance)

MU = 1.0
D = 1.0


# ------------------------------------------- "the standard treatment" model
# The standard treatment, ignoring self-inductance: v(t)=v0 exp(-alpha0 t),
# eq. (overdamped), with alpha0 = B^2 l^2 / (R(x0) M), R evaluated at the
# (approximately constant) initial position x0.

def alpha0(B, rho, l, x0):
    R = _resistance(rho, D, l, x0)
    return B**2 * l**2 / R


def p_pos0(t, p0, B, rho, l, x0):
    """Momentum (eq. vel; M=1 in reduced units, so p=v)."""
    a = alpha0(B, rho, l, x0)
    return p0 * np.exp(-a * t)


def x_pos0(t, x0, p0, B, rho, l):
    """Position of the bar in the no-self-inductance solution."""
    a = alpha0(B, rho, l, x0)
    return x0 + (p0 / a) * (1 - np.exp(-a * t))


def I_pos0(t, p0, B, rho, l, x0):
    """Current: I(t) = I0 exp(-alpha0 t), I0 = B l p0 / R(x0)."""
    R = _resistance(rho, D, l, x0)
    a = alpha0(B, rho, l, x0)
    I0 = B * l * p0 / R
    return I0 * np.exp(-a * t)


# --------------------------------------- "approximate model and RLC analogy"
# The damped-oscillator / RLC-equivalent model, eqs. (oscillator)-(udsol).

def omega0(l, x, B):
    """Undamped natural frequency of the RLC-equivalent circuit,
    omega_0 = l B / sqrt(L(x)) (M=1 in reduced units)."""
    L = self_inductance(MU, D, l, x)
    return l * B / np.sqrt(L)


def alpha_damp(l, rho):
    """Damping rate alpha = R/(2L), asymptotic (x -> infinity) form;
    equals 1/(2 tau_ch(l))."""
    lfact = MU * (1 + np.log(l**2))
    return 2 * rho / lfact


def omega_predicted(l, x, B, rho):
    """Damped oscillation frequency omega_d = sqrt(omega_0^2 - gamma^2),
    with omega_0 and gamma = R/(2L) evaluated at x, as eq. (bcrit) does;
    0 if the system is over-damped or critically damped."""
    w0 = omega0(l, x, B)
    g = gamma_exact(l, x, rho)
    disc = w0**2 - g**2
    return np.sqrt(np.where(disc > 0, disc, 0.0))


def B0_critical(l, x0, rho):
    """B_{0,c}(rho) = sqrt(M L(x0)) gamma(x0) / l, eq. (bcrit): the field at
    which omega_0(x0) = gamma(x0), i.e. the over-/under-damped crossover,
    with L and R evaluated at x0 as in the paper."""
    L_x0 = self_inductance(MU, D, l, x0)
    return gamma_exact(l, x0, rho) * np.sqrt(L_x0) / l


def damped_oscillator(t, x0, p0, B, rho, l):
    """(x, p, I) of the damped-oscillator approximation for the launch
    conditions p(0)=p0, I(0)=0 (M=1 in reduced units, so p=v): the
    over-damped eq. (odsol), the under-damped eq. (udsol), and the
    critical case between them. gamma and omega_0 are evaluated at x0, as
    the paper does.

    The two regimes are one analytic continuation, Omega -> i omega_d, so
    a single branch on the sign of gamma^2 - omega_0^2 covers both:

        p(t) = p0 [EC(t) + gamma ES(t)],    I(t) = (l B / L) p0 ES(t),

    with (EC, ES) = e^{-gamma t} times (cosh(Om t), sinh(Om t)/Om) when
    gamma > omega_0, (cos(wd t), sin(wd t)/wd) when gamma < omega_0, and
    (1, t) at gamma = omega_0. Integrating the oscillator equation once,
    with p'(0) = -l B I(0) = 0, gives the position in closed form,
    x - x0 = [2 gamma p0 - p'(t) - 2 gamma p(t)] / omega_0^2 and
    p' = -l B I; its t -> infinity limit is the drift 2 gamma p0/omega_0^2,
    i.e. chi_inf = -1/Q, the eps-independent term of eq. (restlaw)."""
    t = np.asarray(t, dtype=float)
    L = self_inductance(MU, D, l, x0)
    w0 = omega0(l, x0, B)
    g = gamma_exact(l, x0, rho)
    disc = g**2 - w0**2
    if disc > 0:                                  # over-damped, eq. (odsol)
        Om = np.sqrt(disc)
        # e^{-gamma t} cosh(Om t) and e^{-gamma t} sinh(Om t)/Om, written as
        # two decaying exponentials: the same expressions rearranged so that
        # neither factor overflows when Om*t is large.
        slow, fast = np.exp(-(g - Om) * t), np.exp(-(g + Om) * t)
        EC, ES = 0.5 * (slow + fast), 0.5 * (slow - fast) / Om
    elif disc < 0:                                # under-damped, eq. (udsol)
        wd = np.sqrt(-disc)
        decay = np.exp(-g * t)
        EC, ES = decay * np.cos(wd * t), decay * np.sin(wd * t) / wd
    else:                                         # critical damping
        decay = np.exp(-g * t)
        EC, ES = decay, decay * t
    p = p0 * (EC + g * ES)
    I = (l * B / L) * p0 * ES
    x = x0 + (2 * g * p0 + l * B * I - 2 * g * p) / w0**2
    return x, p, I


# ------------------------------- "beyond the RLC limit": eps and its signatures
# The single dimensionless parameter eps and the two finite-amplitude
# signatures it governs, eqs. (thresh), (thresh), (restlaw) and (epsstar).
# eps = L' I_m / (l B0) = (L'/L(x0)) X_m is the fractional change of the
# self-inductance over the free amplitude X_m = |v0|/omega_0, and the
# natural length for the motion is X_m itself: chi = (x-x0)/X_m.
# gamma and Q here use the exact L(x) and R(x), not the asymptotic
# alpha_damp() above, since Sec. "Beyond the RLC limit" works at a single
# x0 where the constant term of eq. (Lsimple) is not negligible.

def gamma_exact(l, x, rho):
    """Damping rate gamma = R(x)/(2L(x)) of the equivalent RLC circuit."""
    L = self_inductance(MU, D, l, x)
    R = _resistance(rho, D, l, x)
    return R / (2 * L)


def quality_factor(l, x, B, rho):
    """Q = omega_0/(2 gamma), which sets both the eps-independent term
    -1/Q of eq. (restlaw) and the reversal point eps* of eq. (epsstar)."""
    return omega0(l, x, B) / (2 * gamma_exact(l, x, rho))


def p0_from_epsilon(eps, l, x0, B):
    """|p0| launching the bar at eps, eq. (thresh):
    |p0| = eps l B sqrt(L(x0)) / L'(x0), with M=1 in reduced units. This is
    eq. (thresh) read backwards, |v0| = omega_0 X_m with X_m = eps L(x0)/L',
    using the exact L(x0) and L'(x0) (replacing L(x0) by L'x0 would change
    p0 by the 12% the constant term of eq. (Lsimple) still carries at
    x0=500)."""
    return eps * l * B * np.sqrt(self_inductance(MU, D, l, x0)) / \
        self_inductance_dx(MU, D, l, x0)


def free_amplitude(eps, l, x0):
    """X_m = |v0|/omega_0 = eps L(x0)/L'(x0), eq. (thresh): the amplitude the
    bar would keep without dissipation, and the length that rescales the
    displacement into chi = (x-x0)/X_m."""
    return eps * self_inductance(MU, D, l, x0) / self_inductance_dx(MU, D, l, x0)


def x_damped_oscillator_norm(t, l, x0, B, rho):
    """The linear damped-oscillator excursion rescaled by X_m, i.e.
    chi_0(t) = (x-x0)/X_m. Integrating v(t) = v0 e^{-gamma t}[cos(w_d t) +
    (gamma/w_d) sin(w_d t)] once and dividing by X_m = |v0|/omega_0
    cancels every trace of eps, so a single curve serves every amplitude
    and any departure from it is the dL/dx terms alone. Its t -> infinity
    limit is -1/Q, the eps-independent term of eq. (restlaw)."""
    w0 = omega0(l, x0, B)
    g = gamma_exact(l, x0, rho)
    wd = np.sqrt(w0**2 - g**2)
    osc = -2 * g * np.cos(wd * t) + ((w0**2 - 2 * g**2) / wd) * np.sin(wd * t)
    return -(2 * g + np.exp(-g * t) * osc) / w0


def instantaneous_frequency(t, I):
    """Instantaneous angular frequency from consecutive zero crossings of
    I(t). Each crossing time is linearly interpolated between the two
    bracketing samples; successive crossings are half a period apart, so
    omega = pi/(t_{k+1}-t_k). Returns (t_mid, omega)."""
    neg = np.signbit(I)
    k = np.flatnonzero(neg[1:] != neg[:-1])
    tc = t[k] + (t[k + 1] - t[k]) * np.abs(I[k]) / (np.abs(I[k]) + np.abs(I[k + 1]))
    return 0.5 * (tc[1:] + tc[:-1]), np.pi / np.diff(tc)


def fit_drift_linear(eps, chi_inf, Q):
    """Least-squares fit of eq. (restlaw), chi_inf = -1/Q + c eps, with the
    eps-independent term held at its predicted value -1/Q so that c is the
    only measured number. Returns (c, eps_star) with eps_star = 1/(c Q),
    eq. (epsstar)."""
    eps, chi_inf = np.asarray(eps, float), np.asarray(chi_inf, float)
    c = np.sum((chi_inf + 1.0 / Q) * eps) / np.sum(eps**2)
    return c, 1.0 / (c * Q)
