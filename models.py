"""The two limiting-case analytic models the paper compares the full
simulation against -- the standard treatment that ignores self-inductance,
and the damped-oscillator/RLC-equivalent model -- together with the
curve-fitting helpers used to fit the second one to simulated data.
Reduced units throughout (mu=1, d=1, M=1), matching dynamics.py.

Every expression here is a direct transcription of the corresponding
displayed equation in the paper, and each was checked against an
independent implementation of the same formulas.
"""

import numpy as np
from scipy.optimize import curve_fit

from inductance import self_inductance, resistance as _resistance

MU = 1.0
D = 1.0


# ------------------------------------------- "the standard treatment" model
# The standard treatment, ignoring self-inductance: v(t)=v0 exp(-alpha0 t),
# eqs. (vel)-(pos0), with alpha0 = B^2 l^2 / (R(x0) M), R evaluated at the
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
# The damped-oscillator / RLC-equivalent model, eqs. (currsol)-(fapp).

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
    """Damped oscillation frequency omega_d = sqrt(omega_0^2 - alpha^2);
    0 if the system is over-damped or critically damped."""
    w0 = omega0(l, x, B)
    alpha = alpha_damp(l, rho)
    disc = w0**2 - alpha**2
    return np.sqrt(np.where(disc > 0, disc, 0.0))


def B0_critical(l, x0, rho):
    """B_{0,c}(rho): the field at which omega_0(x0) = alpha,
    i.e. the over-/under-damped crossover."""
    L_x0 = self_inductance(MU, D, l, x0)
    return alpha_damp(l, rho) * np.sqrt(L_x0) / l


def oscillating_exp_fix_decay_freq(t, A, alpha, omega, phi):
    """The fitted damped-oscillator force model:
    F(t) = A exp(-alpha t) cos(omega t + phi), with alpha and omega fixed
    to the analytic predictions (alpha_damp, omega_predicted) and only A,
    phi left free. A = l B0 I_mx, so I_mx = A/(l B0)."""
    return A * np.exp(-alpha * t) * np.cos(omega * t + phi)


def fit_to_function_fix_decay_freq(ini_guesses, decay, omega, t, Ftot):
    """Fit (A, phi) of oscillating_exp_fix_decay_freq to (t, Ftot) via
    scipy.optimize.curve_fit, with decay and omega held fixed. Returns
    (R^2, RMSE, [(label, value, stderr), ...])."""
    fwrapper = lambda t, A, phi: oscillating_exp_fix_decay_freq(t, A, decay, omega, phi)

    popt, pcov = curve_fit(fwrapper, t, Ftot, p0=ini_guesses, maxfev=10000)

    residuals = Ftot - fwrapper(t, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((Ftot - np.mean(Ftot))**2)
    r_squared = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(ss_res / len(Ftot))
    perr = np.sqrt(np.diag(pcov))

    labels = ['A', 'phi']
    return r_squared, rmse, list(zip(labels, popt, perr))


def normalize_amplitude_phase(A, phi):
    """curve_fit is free to converge to A<0 paired with a phi shifted by
    pi (same curve, since A cos(x) = (-A) cos(x+pi)). Normalize to A>=0 so
    that the reported I_mx = A/(l B0) and phi are the physically
    meaningful pair satisfying I(0) = I_mx cos(phi); also wraps phi to
    (-pi, pi]."""
    if A < 0:
        A = -A
        phi = phi + np.pi
    phi = (phi + np.pi) % (2 * np.pi) - np.pi
    return A, phi
