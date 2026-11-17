"""Matplotlib styling for the paper figures.

Figures are authored at the width they occupy in the paper (SIAM siamart
\\textwidth = 370.38374 pt) so that LaTeX includes them without rescaling.
"""

import matplotlib as mpl

PT_PER_IN = 72.27
TEXTWIDTH_PT = 370.38374
TEXTWIDTH_IN = TEXTWIDTH_PT / PT_PER_IN

BODY_PT = 9
TICK_PT = 8
LEGEND_PT = 8.5

# Okabe-Ito vermillion (ground truth) and blue (BTJD).
C_GT = '#D55E00'
C_BT = '#0072B2'


def use_paper_style():
    mpl.rcParams.update({
        'font.size':        BODY_PT,
        'axes.titlesize':   BODY_PT,
        'axes.labelsize':   BODY_PT,
        'xtick.labelsize':  TICK_PT,
        'ytick.labelsize':  TICK_PT,
        'legend.fontsize':  LEGEND_PT,
        'figure.titlesize': BODY_PT,
        'font.family':      'serif',
        'mathtext.fontset': 'cm',
        'savefig.bbox':     'standard',
    })


def figsize(fraction=1.0, aspect=0.5):
    """(width, height) in inches for a figure spanning `fraction` of \\textwidth."""
    w = fraction * TEXTWIDTH_IN
    return (w, w * aspect)


def scale_panel_fonts(panel_fraction):
    """Shrink label/tick fonts for panels narrower than half the text width."""
    label_pt = max(6.5, BODY_PT * panel_fraction / 0.48)
    tick_pt = max(6.0, TICK_PT * panel_fraction / 0.48)
    mpl.rcParams.update({
        'axes.labelsize':  label_pt,
        'xtick.labelsize': tick_pt,
        'ytick.labelsize': tick_pt,
    })
