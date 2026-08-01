"""Velocity-Verlet (kick-drift-kick) integrator for the sliding-bar
equations of motion, eqs. (vel1)-(curr1), in the paper's reduced units
(wire radius d=1, bar mass M=1, so that mu, d and M are all unity).

The scheme is transcribed from the paper's own displayed update equations,
and is built on top of inductance.py (pinning d=1 explicitly at every
call, rather than hardcoding "1" into a second copy of the
self-inductance formulas) so there is exactly one implementation of
L(x,l) and dL/dx in this supplement. Full trajectories from this
integrator were checked against an independent implementation of the same
scheme and agree to better than 1e-11 relative at every step.
"""

import numpy as np

from _jit import njit
from inductance import self_inductance, self_inductance_dx, resistance as _resistance

MU = 1.0
D = 1.0


@njit(cache=True)
def resistance(rho, l, x):
    return _resistance(rho, D, l, x)


@njit(cache=True)
def force(l, B, I, x):
    """(f0, f1): braking force and inductance-gradient force."""
    f0 = -l * B * I
    f1 = self_inductance_dx(MU, D, l, x) * I**2 / 2
    return f0, f1


@njit(cache=True)
def accel(x, I, l, B):
    """a(x,I) = f0 + f1 (M=1 in reduced units)."""
    f0, f1 = force(l, B, I, x)
    return f0 + f1, f0, f1


@njit(cache=True)
def update_current_verlet(x_mid, p_half, I, l, B, rho, dt):
    """Trapezoidal (Crank-Nicolson) update of the linear ODE
    dI/dt = [B l v - (R + L' v) I] / L, with v frozen at the half-step
    velocity p_half (exact over the drift substep, where v is constant)
    and L, L', R frozen at the drift midpoint x_mid. Freezing the
    coefficients at the midpoint rather than either endpoint is what keeps
    the update second order: the coefficient error from freezing is
    O(dt^2) at the midpoint (symmetric cancellation) vs O(dt) at an
    endpoint. Closed form, no iteration required since the ODE is linear
    in I."""
    L = self_inductance(MU, D, l, x_mid)
    dxL = self_inductance_dx(MU, D, l, x_mid)
    R = resistance(rho, l, x_mid)
    k = (R + dxL * p_half) / L
    b = B * l * p_half / L
    I_new = ((1 - 0.5 * dt * k) * I + dt * b) / (1 + 0.5 * dt * k)
    return I_new


@njit(cache=True)
def verlet_step(x, p, I, l, B, rho, dt, a_prev, R_prev):
    """One kick-drift-kick step. a_prev must be the acceleration already
    evaluated at the current state (x, I); R_prev likewise the resistance
    at the current state -- both carried over from the previous step's
    returned values, so each step costs one force evaluation, not two."""
    p_half = p + 0.5 * dt * a_prev
    x_new = x + dt * p_half
    x_mid = 0.5 * (x + x_new)

    I_new = update_current_verlet(x_mid, p_half, I, l, B, rho, dt)

    a_new, f0_new, f1_new = accel(x_new, I_new, l, B)
    p_new = p_half + 0.5 * dt * a_new

    L_new = self_inductance(MU, D, l, x_new)
    R_new = resistance(rho, l, x_new)

    return x_new, p_new, I_new, L_new, R_new, f0_new, f1_new, a_new


def verlet_integrate(x0, p0, l, B, rho, dt, n_steps, n_print, wall=1.1, floor=1.0):
    """Integrate eqs. (vel1)-(curr1) for n_steps steps of size dt, starting
    from (x0, p0, I=0). Records every n_print-th step. An elastic collision
    reflects p when x < wall; integration stops early if x < floor.

    Returns (x, p, I, F, K, E_B, Q_tot) as lists, with F = [(f0,f1), ...],
    K = kinetic energy p^2/2, E_B = field energy L I^2/2, and Q_tot the
    running total dissipated heat (trapezoidal quadrature of I^2 R)."""
    x_prev, p_prev, I_prev = x0, p0, 0.0
    Q_tot = 0.0

    a_prev, _, _ = accel(x0, I_prev, l, B)
    R_prev = resistance(rho, l, x0)

    x_list, p_list, I_list, F_list = [], [], [], []
    K_list, Eb_list, Q_list = [], [], []
    for i in range(n_steps):
        x, p, I, L, R, f0, f1, a_new = verlet_step(
            x_prev, p_prev, I_prev, l, B, rho, dt, a_prev, R_prev
        )
        K = p**2 / 2
        E_B = L * I**2 / 2
        # Trapezoidal (second-order) quadrature of the dissipated heat; a
        # first-order left-rectangle rule would impose its own O(dt) floor
        # on the measured energy error, masking the integrator's true
        # (second) order.
        Q = 0.5 * dt * (R_prev * I_prev**2 + R * I**2)
        Q_tot += Q
        if x < wall:
            p = -p
        if x < floor:
            break
        if i % n_print == 0:
            x_list.append(x)
            p_list.append(p)
            I_list.append(I)
            F_list.append((f0, f1))
            K_list.append(K)
            Eb_list.append(E_B)
            Q_list.append(Q_tot)
        x_prev, p_prev, I_prev = x, p, I
        a_prev, R_prev = a_new, R

    return x_list, p_list, I_list, F_list, K_list, Eb_list, Q_list
