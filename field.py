"""The induced field, region by region: the nine-piece closed form
f_B(p,h;x) (appendix "Field at the corners", eq. bfun) that
inductance.py's self_inductance/self_inductance_dx are the flux integral
of, plus the enclosed-current-fraction WEIGHTED analogues of the four
inside-wire pieces used to derive Delta L (appendix "Enclosed-current-
fraction weighting", eq. DeltaL).

This is appendix-level, correctness-check-only material -- Notebook 1's
dynamics never call it, only Notebook 2. Each piece is transcribed from
the corresponding displayed expression in the derivations supplement and
written to match inductance.py's calling convention, with mu carried
explicitly.
"""

import numpy as np

from inductance import chi


# ---------------------------------------------------------- outside the wire

def f_out(p, h, x, l, d, mu=1.0):
    return (mu / (4 * np.pi)) * (
        1 / ((x - p) * chi(h, x - p)) + 1 / (p * chi(h, p))
        + 1 / ((x - p) * chi(l - h, x - p)) + 1 / (p * chi(l - h, p))
    )


# ----------------------------------------------------- inside the wire (non-corner)

def f_in_rt(p, h, x, l, d, mu=1.0):
    return (mu / (4 * np.pi)) * (
        1 / (p * chi(h, p)) + 1 / (p * chi(l - h, p))
        + chi(x - p, h) / h + chi(x - p, l - h) / (l - h)
        + (l - h) * chi(x - p, l - h) / d**2 + h * chi(x - p, h) / d**2
    )


def f_in_lt(p, h, x, l, d, mu=1.0):
    return (mu / (4 * np.pi)) * (
        1 / ((x - p) * chi(h, x - p)) + 1 / ((x - p) * chi(l - h, x - p))
        + chi(p, h) / h + chi(p, l - h) / (l - h)
        + (l - h) * chi(p, l - h) / d**2 + h * chi(p, h) / d**2
    )


def f_in_up(p, h, x, l, d, mu=1.0):
    return (mu / (4 * np.pi)) * (
        1 / (p * chi(h, p)) + 1 / ((x - p) * chi(h, x - p))
        + chi(l - h, x - p) / (x - p) + chi(l - h, p) / p
        + p * chi(l - h, p) / d**2 + (x - p) * chi(l - h, x - p) / d**2
    )


def f_in_lo(p, h, x, l, d, mu=1.0):
    return (mu / (4 * np.pi)) * (
        1 / (p * chi(l - h, p)) + 1 / ((x - p) * chi(l - h, x - p))
        + chi(h, x - p) / (x - p) + chi(h, p) / p
        + p * chi(h, p) / d**2 + (x - p) * chi(h, x - p) / d**2
    )


# -------------------------------------------------------------------- corners

def f_lo_rt(p, h, x, l, d, mu=1.0):
    return (
        (mu / (4 * np.pi * d**2)) * ((l - h) * chi(x - p, l - h) + 2 * h * chi(x - p, h) + p * chi(h, p))
        + (mu / (4 * np.pi * (l - h))) * (chi(p, l - h) + chi(x - p, l - h))
        + (mu / (4 * np.pi * p)) * (chi(h, p) + chi(l - h, p))
    )


def f_up_rt(p, h, x, l, d, mu=1.0):
    return (
        (mu / (4 * np.pi * d**2)) * (2 * (l - h) * chi(x - p, l - h) + h * chi(x - p, h) + p * chi(l - h, p))
        + (mu / (4 * np.pi * h)) * (chi(x - p, h) + chi(p, h))
        + (mu / (4 * np.pi * p)) * (chi(h, p) + chi(l - h, p))
    )


def f_lo_lt(p, h, x, l, d, mu=1.0):
    return (
        (mu / (4 * np.pi * d**2)) * ((l - h) * chi(p, l - h) + 2 * p * chi(h, p) + (x - p) * chi(h, x - p))
        + (mu / (4 * np.pi * (l - h))) * (chi(p, l - h) + chi(x - p, l - h))
        + (mu / (4 * np.pi * (x - p))) * (chi(l - h, x - p) + chi(h, x - p))
    )


def f_up_lt(p, h, x, l, d, mu=1.0):
    return (
        (mu / (4 * np.pi * d**2)) * (2 * p * chi(l - h, p) + h * chi(p, h) + (x - p) * chi(l - h, x - p))
        + (mu / (4 * np.pi * h)) * (chi(x - p, h) + chi(p, h))
        + (mu / (4 * np.pi * (x - p))) * (chi(l - h, x - p) + chi(h, x - p))
    )


def f_B(p, h, x, l, d, mu=1.0):
    """The nine-piece closed form of the induced field: dispatches to the
    region containing (p, h)."""
    if d < h < l - d and d < p < x - d:
        return f_out(p, h, x, l, d, mu)
    if 0 <= h <= d and d < p < x - d:
        return f_in_lo(p, h, x, l, d, mu)
    if l - d <= h <= l and d < p < x - d:
        return f_in_up(p, h, x, l, d, mu)
    if d < h < l - d and 0 <= p <= d:
        return f_in_lt(p, h, x, l, d, mu)
    if d < h < l - d and x - d <= p <= x:
        return f_in_rt(p, h, x, l, d, mu)
    if 0 <= h <= d and 0 <= p <= d:
        return f_lo_lt(p, h, x, l, d, mu)
    if l - d <= h <= l and 0 <= p <= d:
        return f_up_lt(p, h, x, l, d, mu)
    if 0 <= h <= d and x - d <= p <= x:
        return f_lo_rt(p, h, x, l, d, mu)
    if l - d <= h <= l and x - d <= p <= x:
        return f_up_rt(p, h, x, l, d, mu)
    raise ValueError((p, h))


# ---------------------------------------------------- enclosed-current-fraction
# WEIGHTED analogues of the four inside-wire pieces (only the "own-field"
# term of each gets the (r/d)^2 weight; see appendix "Enclosed-current-
# fraction weighting"). Used only to re-derive Delta L independently, by
# direct quadrature, in Notebook 2.

def f_in_rt_w(p, h, x, l, d, mu=1.0):
    cross = 1 / (p * chi(h, p)) + 1 / (p * chi(l - h, p)) + chi(x - p, h) / h + chi(x - p, l - h) / (l - h)
    self_ = ((l - h) * chi(x - p, l - h) + h * chi(x - p, h)) / d**2
    return (mu / (4 * np.pi)) * (cross + ((x - p) / d)**2 * self_)


def f_in_lt_w(p, h, x, l, d, mu=1.0):
    cross = 1 / ((x - p) * chi(h, x - p)) + 1 / ((x - p) * chi(l - h, x - p)) + chi(p, h) / h + chi(p, l - h) / (l - h)
    self_ = ((l - h) * chi(p, l - h) + h * chi(p, h)) / d**2
    return (mu / (4 * np.pi)) * (cross + (p / d)**2 * self_)


def f_in_up_w(p, h, x, l, d, mu=1.0):
    cross = 1 / (p * chi(h, p)) + 1 / ((x - p) * chi(h, x - p)) + chi(l - h, x - p) / (x - p) + chi(l - h, p) / p
    self_ = (p * chi(l - h, p) + (x - p) * chi(l - h, x - p)) / d**2
    return (mu / (4 * np.pi)) * (cross + ((l - h) / d)**2 * self_)


def f_in_lo_w(p, h, x, l, d, mu=1.0):
    cross = 1 / (p * chi(l - h, p)) + 1 / ((x - p) * chi(l - h, x - p)) + chi(h, x - p) / (x - p) + chi(h, p) / p
    self_ = (p * chi(h, p) + (x - p) * chi(h, x - p)) / d**2
    return (mu / (4 * np.pi)) * (cross + (h / d)**2 * self_)


def f_lo_rt_w(p, h, x, l, d, mu=1.0):
    a, b = x - p, h
    bar_self = b * chi(a, b) + (l - b) * chi(a, l - b)
    rail_self = b * chi(a, b) + (x - a) * chi(b, x - a)
    cross = (chi(p, l - h) + chi(x - p, l - h)) / (l - h) + (chi(h, p) + chi(l - h, p)) / p
    return (mu / (4 * np.pi)) * (((a / d)**2 * bar_self + (b / d)**2 * rail_self) / d**2 + cross)


def f_up_rt_w(p, h, x, l, d, mu=1.0):
    a, b = x - p, l - h
    bar_self = b * chi(a, b) + h * chi(a, h)
    rail_self = b * chi(a, b) + (x - a) * chi(b, x - a)
    cross = (chi(x - p, h) + chi(p, h)) / h + (chi(h, p) + chi(l - h, p)) / p
    return (mu / (4 * np.pi)) * (((a / d)**2 * bar_self + (b / d)**2 * rail_self) / d**2 + cross)


def f_lo_lt_w(p, h, x, l, d, mu=1.0):
    a, b = p, h
    left_self = b * chi(a, b) + (l - b) * chi(a, l - b)
    rail_self = b * chi(a, b) + (x - a) * chi(b, x - a)
    cross = (chi(p, l - h) + chi(x - p, l - h)) / (l - h) + (chi(l - h, x - p) + chi(h, x - p)) / (x - p)
    return (mu / (4 * np.pi)) * (((a / d)**2 * left_self + (b / d)**2 * rail_self) / d**2 + cross)


def f_up_lt_w(p, h, x, l, d, mu=1.0):
    a, b = p, l - h
    left_self = b * chi(a, b) + h * chi(a, h)
    rail_self = b * chi(a, b) + (x - a) * chi(b, x - a)
    cross = (chi(x - p, h) + chi(p, h)) / h + (chi(l - h, x - p) + chi(h, x - p)) / (x - p)
    return (mu / (4 * np.pi)) * (((a / d)**2 * left_self + (b / d)**2 * rail_self) / d**2 + cross)


UNWEIGHTED = dict(rt=f_in_rt, lt=f_in_lt, lo=f_in_lo, up=f_in_up,
                   lort=f_lo_rt, uprt=f_up_rt, lolt=f_lo_lt, uplt=f_up_lt)
WEIGHTED = dict(rt=f_in_rt_w, lt=f_in_lt_w, lo=f_in_lo_w, up=f_in_up_w,
                lort=f_lo_rt_w, uprt=f_up_rt_w, lolt=f_lo_lt_w, uplt=f_up_lt_w)
