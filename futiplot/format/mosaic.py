# src/futiplot/format/mosaic.py

from matplotlib import pyplot as plt

def create_story_axes(mosaic, *, fig_kwargs=None, gridspec_kw=None, constrained_layout=True):
    """
    Build a figure and named axes from an ASCII-style mosaic spec.

    Parameters
    ----------
    mosaic : list of lists of str
        Layout grid where each string is the name of an axis in that cell.
    fig_kwargs : dict, optional
        Passed through to plt.figure (e.g. figsize, dpi, facecolor).
    gridspec_kw : dict, optional
        Passed through subplot_mosaic's gridspec_kw (width_ratios, height_ratios, wspace, hspace).
    constrained_layout : bool
        Whether to enable constrained layout for automatic padding.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : dict[str, Axes]
        Mapping from the names you used in the mosaic spec to actual Axes objects.
    """
    fig = plt.figure(**(fig_kwargs or {}), constrained_layout=constrained_layout)
    axes = fig.subplot_mosaic(mosaic, gridspec_kw=gridspec_kw or {})
    return fig, axes
