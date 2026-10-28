"""Plotting helpers for Notebook 1, one per reproduced figure of the paper
and of the first supplementary item. Each
function takes already-computed data (from dynamics.py/models.py/
inductance.py) and an optional save path, and returns the matplotlib
Figure. Layouts reproduce those of the paper's own figures. The paper's
two schematic diagrams -- the circuit itself, and the field of a current
element -- are hand drawn rather than computed, so they have no
corresponding function here.

Figures are drawn at their final printed size (single figures 8.5 cm
wide, wide figures 15 cm wide) with 8-9 pt text, so that the labels come
out in the 8-12 pt range the journal asks for without any rescaling at
inclusion time. Series are separated by line style and marker as well as
by colour, so the figures survive conversion to greyscale; multi-panel
figures carry (a)/(b)/(c) labels.
"""

import numpy as np
import matplotlib.pyplot as plt

CM = 1 / 2.54          # centimetres -> inches
WIDTH_SINGLE = 8.5 * CM   # single-figure width
WIDTH_WIDE = 15.0 * CM    # wide (two-panel) figure width
DPI = 300

# Text sizes are in points at the final printed size above.
STYLE = {
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'lines.linewidth': 1.2,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
}


def _panel_label(ax, text):
    """Lower-case part label, placed just inside the top-left corner."""
    ax.text(0.02, 0.95, text, transform=ax.transAxes,
            ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle='square,pad=0.15', facecolor='white',
                      edgecolor='none', alpha=0.75))


def plot_xpI(t, x, p, I, x_model, p_model, I_model, xlim=None, save_path=None,
             model_label='damped oscillator'):
    """x, p, I vs t, with an analytic model overlaid (dashed).
    Reproduces the paper's two three-panel trajectory figures, which
    overlay the damped-oscillator solution eqs. (odsol)/(udsol) on either
    side of Q=1/2; pass model_label to overlay a different model (the
    no-self-inductance one, say). Panels are labelled (a), (b), (c) from
    top to bottom."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 1, sharex=True,
                                 figsize=(WIDTH_SINGLE, 1.3 * WIDTH_SINGLE))
        axes[0].plot(t, x, linestyle='-', color='C0', label='full solution')
        axes[0].plot(t, x_model, linestyle='--', color='k',
                     label=model_label)
        axes[0].set_ylabel('$x$')
        axes[0].legend(loc='best')
        axes[1].plot(t, p, linestyle='-', color='C0')
        axes[1].plot(t, p_model, linestyle='--', color='k')
        axes[1].set_ylabel('$p_x$')
        axes[2].plot(t, I, linestyle='-', color='C0')
        axes[2].plot(t, I_model, linestyle='--', color='k')
        axes[2].set_ylabel('$I$')
        axes[2].set_xlabel('$t$')
        for ax, tag in zip(axes, ('(a)', '(b)', '(c)')):
            _panel_label(ax, tag)
        if xlim is not None:
            for ax in axes:
                ax.set_xlim(*xlim)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=DPI)
    return fig


def plot_rosa_grover(x, L_exact, L_rosa, save_path=None):
    """Self-inductance L(x,l), this work vs. Rosa/Grover. Reproduces
    the paper's self-inductance comparison figure."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(WIDTH_SINGLE, 0.72 * WIDTH_SINGLE))
        ax.plot(x, L_exact, linestyle='-', color='C0',
                label='exact closed form (this work)')
        ax.plot(x, L_rosa, linestyle='--', color='k',
                label='Rosa and Grover')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$L$')
        ax.legend()
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=DPI)
    return fig


def plot_L_vs_asymptote(x, L_exact, L_asymp, save_path=None):
    """L(x,l) vs. its x -> infinity asymptotic form. Reproduces
    the paper's self-inductance-vs-asymptote figure."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(WIDTH_SINGLE, 0.72 * WIDTH_SINGLE))
        ax.plot(x, L_exact, linestyle='-', color='C0', label='$L(x)$')
        ax.plot(x, L_asymp, linestyle='--', color='k',
                label=r'$L(x\rightarrow\infty)$')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$L(x)$')
        ax.legend()
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=DPI)
    return fig


def plot_position_vs_eps(panels, x0, save_path=None):
    """Bar position vs time at two values of eps on either side of the
    critical point (paper Sec. "Beyond the RLC limit"). Same content as the
    top panel of the trajectory figures above -- x(t) full (solid) against
    the eps -> 0 damped-oscillator reference (dashed) -- on a two-by-two
    grid whose rows are the two regimes (over-damped above, under-damped
    below) and whose columns are a small and a large eps. The dotted line
    marks the launch position x0, so the gap between the two curves at late
    times is the shift of the resting position that eps produces, in both
    regimes.

    panels: list of four (label, t_over_T0, x_full, x_oscillator), in
            reading order; the label is placed in the panel corner and
            should name the regime's Q and the panel's eps.
    x0: launch position, drawn as a horizontal reference in every panel.
    """
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2,
                                 figsize=(WIDTH_WIDE, 0.62 * WIDTH_WIDE))
        for ax, tag, (label, tt, xx, xm) in zip(axes.ravel(),
                                                ('(a)', '(b)', '(c)', '(d)'),
                                                panels):
            ax.axhline(x0, color='0.6', linewidth=0.6, linestyle=':', zorder=0)
            ax.plot(tt, xx, linestyle='-', color='C0', label='full solution')
            ax.plot(tt, xm, linestyle='--', color='k',
                    label='damped oscillator')
            ax.set_ylabel('$x$')
            ax.set_xlim(tt[0], tt[-1])
            # headroom at the top so the panel label never sits on the curves
            lo = min(xx.min(), xm.min())
            hi = max(xx.max(), xm.max(), x0)
            ax.set_ylim(lo - 0.04 * (hi - lo), hi + 0.26 * (hi - lo))
            _panel_label(ax, f'{tag} {label}')
        for ax in axes[-1]:
            ax.set_xlabel('$t/T_0$')
        axes[0, 1].legend(loc='center right', framealpha=0.9, fontsize=7,
                          handlelength=2.2, borderpad=0.4, labelspacing=0.3)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=DPI)
    return fig
