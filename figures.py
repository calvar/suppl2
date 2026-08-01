"""Plotting helpers for Notebook 1, one per reproduced paper figure. Each
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
import matplotlib.patheffects as pe

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


def plot_xpI(t, x, p, I, x_model, p_model, I_model, xlim=None, save_path=None):
    """x, p, I vs t, with the no-self-inductance model overlaid (dashed).
    Reproduces the paper's two three-panel trajectory figures (the
    over-damped and under-damped cases). Panels are labelled (a), (b),
    (c) from top to bottom."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 1, sharex=True,
                                 figsize=(WIDTH_SINGLE, 1.3 * WIDTH_SINGLE))
        axes[0].plot(t, x, linestyle='-', color='C0', label='full solution')
        axes[0].plot(t, x_model, linestyle='--', color='k',
                     label='no self-inductance')
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


def plot_energy(t, K, Eb, Q, xlim=None, save_path=None):
    """K, E_B, Q_tot and their sum vs t. Reproduces the paper's energy
    figure. Each contribution gets its own line style so the curves stay
    distinguishable in greyscale."""
    E_total = K + Eb + Q
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(WIDTH_SINGLE, 0.72 * WIDTH_SINGLE))
        ax.plot(t, K, linestyle='-', color='C0', label='$K$')
        ax.plot(t, Eb, linestyle='--', color='C1', label='$E_B$')
        ax.plot(t, Q, linestyle='-.', color='C2', label='$Q_{tot}$')
        ax.plot(t, E_total, linestyle=(0, (1, 1.5)), color='k',
                label='$K+E_B+Q_{tot}$')
        ax.set_xlabel('$t$')
        ax.set_ylabel('Energy')
        ax.legend()
        if xlim is not None:
            ax.set_xlim(*xlim)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=DPI)
    return fig


def plot_force_fit(t, Ftot, Ffit, xlim=None, save_path=None):
    """Simulated total force with the fitted damped-oscillator model
    overlaid. Reproduces the paper's force-fit figure. The simulated
    force is drawn as markers and the fit as a dashed line, so the two
    are told apart by shape as well as by colour."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(WIDTH_SINGLE, 0.72 * WIDTH_SINGLE))
        ax.plot(t, Ftot, marker='o', markersize=1.5, linestyle='None',
                color='k', label='Force')
        ax.plot(t, Ffit, linestyle=(0, (5, 2)), linewidth=1.4, color='red',
                label='Fitted model')
        ax.set_xlabel('Time')
        ax.set_ylabel('Force')
        ax.legend()
        if xlim is not None:
            ax.set_xlim(*xlim)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=DPI)
    return fig


def plot_Rsq_heatmaps(rho, B, Rsq_current, rho_ns, B_ns, Rsq_noselfL,
                       rho_line, B0c_line, save_path=None):
    """R^2(rho, B0) heatmaps for the damped-oscillator model and the
    no-self-inductance model, with B_{0,c}(rho) overlaid on both.
    Reproduces the paper's goodness-of-fit heatmap figure. The panels are
    labelled (a) and (b); the critical-field line is drawn white with a
    dark outline so that it stays visible over the whole colour map, in
    colour and in greyscale."""
    vmin, vmax = 0, 1.0
    outline = [pe.Stroke(linewidth=2.2, foreground='k'), pe.Normal()]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(WIDTH_WIDE, 0.42 * WIDTH_WIDE),
                                 sharex=True, sharey=True)

        im0 = axes[0].pcolormesh(rho, B, np.clip(Rsq_current, vmin, vmax),
                                  shading='auto', cmap='viridis',
                                  vmin=vmin, vmax=vmax)
        im1 = axes[1].pcolormesh(rho_ns, B_ns, np.clip(Rsq_noselfL, vmin, vmax),
                                  shading='auto', cmap='viridis',
                                  vmin=vmin, vmax=vmax)
        titles = ('oscillator-model fit', 'no-inductance model')
        for ax, title, tag in zip(axes, titles, ('(a)', '(b)')):
            ax.plot(rho_line, B0c_line, color='w', linestyle='--',
                    linewidth=1.4, path_effects=outline,
                    label='$B_{0,c}(\\rho)$')
            ax.set_xlabel(r'$\rho$')
            ax.set_ylabel(r'$B_0$')
            ax.set_title(title)
            ax.legend(loc='upper right')
            _panel_label(ax, tag)

        fig.colorbar(im1, ax=axes, label='$R^2$', shrink=0.9)
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
