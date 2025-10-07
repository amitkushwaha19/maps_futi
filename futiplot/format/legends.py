import numpy as np
from futiplot.utils.plotting import plot_dotted_line, plot_comet, plot_point
from futiplot.utils.colors import futicolor
from futiplot.utils.fontutils import get_font

def draw_event_legend(
    ax,
    orientation: str = "horizontal",
    symbol_color: str = futicolor.light,
    label_color: str = futicolor.light,
    label_fontsize: int = 14,
    padding: float = 0.1,
    label_gap: float = 0.3,
    dot_size: float = 3,
    dot_spacing: float = 0.012,
    glyph_linewidth: float = 5,
    glyph_line_length: float = 0.15,
    glyph_segments_per_unit: int = 1000,
    point_size: float = 250,
    comet_head_offset: float = 0.001
):
    """
    Render a legend for Carry, Pass, and Touch events.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the legend into (already positioned).
    orientation : {'horizontal','vertical'}
        Layout direction of the legend items.
    symbol_color : str
        Color for the glyphs (defaults to futicolor.light).
    label_color : str
        Color for the text labels (defaults to futicolor.light).
    label_fontsize : int
        Font size of the labels.
    padding : float
        Fractional padding inside the axis (in axes-fraction units).
    dot_size : float
        Size of the dots for "Carry".
    dot_spacing : float
        Spacing between dots in "Carry".
    glyph_linewidth : float
        Line width for the comet and outline of glyphs.
    glyph_line_length : float
        Total length (in axes-fraction units) of line-based glyphs.
    glyph_segments_per_unit : int
        Segments per unit for the comet.
    point_size : float
        Marker size for "Touch" glyph.
    comet_head_offset : float
        Vertical offset (axes-fraction) for the comet head above baseline.
    """
    ax.axis("off")
    ax.set_facecolor(ax.figure.get_facecolor())
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="datalim")

    symbol_color = symbol_color or futicolor.light
    label_color  = label_color  or futicolor.light
    label_font   = get_font("regular")

    # Define mapping from drawing function to glyph offsets
    half_span = glyph_line_length / 2
    glyph_offsets = {
        plot_dotted_line: ([-half_span, 0], [ half_span, 0]),
        plot_comet:       ([-half_span, 0], [ half_span, comet_head_offset]),
        plot_point:       ([0, 0],          [0, 0])
    }

    labels = ["Carry", "Pass", "Touch"]
    funcs  = [plot_dotted_line, plot_comet, plot_point]
    n      = len(labels)

    # Compute uniform glyph width based on the widest item and fixed gap
    raw_widths = [abs(glyph_offsets[fn][1][0] - glyph_offsets[fn][0][0]) for fn in funcs]
    max_width  = max(raw_widths)
    widths     = [max_width] * n
    gap        = padding
    total_width = sum(widths) + gap * (n - 1)
    left       = 0.5 - total_width / 2
    centers_x  = []
    cursor     = left
    for w in widths:
        centers_x.append(cursor + w / 2)
        cursor += w + gap
    centers = np.column_stack([centers_x, np.full(n, 0.5)])


    # Fixed text offsets
    text_dirs = {
        "horizontal": np.array([[0, -label_gap]] * n),
        "vertical":   np.array([[label_gap,  0]] * n)
    }[orientation]

    # Draw each glyph and label
    for (label, fn, text_dir), (cx, cy) in zip(zip(labels, funcs, text_dirs), centers):
        d1, d2 = glyph_offsets[fn]
        start   = np.array([[cx + d1[0], cy + d1[1]]])
        end     = np.array([[cx + d2[0], cy + d2[1]]])

        if fn is plot_dotted_line:
            plot_dotted_line(
                ax, start, end,
                color=symbol_color,
                dot_size=dot_size,
                dot_spacing=dot_spacing
            )
        elif fn is plot_comet:
            plot_comet(
                ax, start, end,
                pitch_color=ax.figure.get_facecolor(),
                event_color=symbol_color,
                linewidth=glyph_linewidth,
                segments_per_unit=glyph_segments_per_unit
            )
        else:
            plot_point(
                ax, start,
                size=point_size,
                color=symbol_color,
                edgecolor=futicolor.dark,
                linewidth=glyph_linewidth,
                zorder=5
            )

        tx, ty = cx + text_dir[0], cy + text_dir[1]
        ax.text(
            tx, ty, label,
            ha="center", va="top",
            fontsize=label_fontsize,
            color=label_color,
            fontproperties=label_font,
            transform=ax.transAxes
        )
