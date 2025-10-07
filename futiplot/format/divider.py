# src/futiplot/format/divider.py

from mpl_toolkits.axes_grid1 import make_axes_locatable

def append_axes(ax, side="right", size="20%", pad=0.05):
    """
    Append a new axes to an existing axes, on a given side.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The parent axes to extend.
    side : {'left','right','top','bottom'}
    size : str
        Fractional width or height of the new axes (e.g. '20%').
    pad : float
        Fractional padding between the parent and new axes.

    Returns
    -------
    new_ax : matplotlib.axes.Axes
    """
    divider = make_axes_locatable(ax)
    return divider.append_axes(side, size=size, pad=pad)
