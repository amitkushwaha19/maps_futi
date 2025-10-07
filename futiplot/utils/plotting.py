import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.patches import Circle
from matplotlib.image import imread
from matplotlib.transforms import Affine2D
from svgpath2mpl import parse_path
from importlib import resources

from .colors import futicolor
from .utils import transform_xy


# shield stuff
shield_d = """
M517.000000,271.000000
C517.000000,295.495422 516.908813,319.491333 517.028625,343.486145
C517.125610,362.905762 511.247528,380.681702 502.669220,397.771210
C495.659058,411.736664 487.009552,424.617706 476.362091,436.045258
C466.500153,446.629822 456.968781,457.679688 446.051727,467.085327
C431.318817,479.778564 415.716583,491.535492 399.918884,502.904175
C372.029388,522.974609 342.001984,539.485962 311.118683,554.484131
C304.327576,557.782166 297.839203,557.878723 291.094604,554.617371
C261.647797,540.378540 233.101181,524.523926 206.199860,505.910858
C192.617569,496.513214 179.471420,486.421356 166.669144,475.979950
C155.973648,467.256805 145.368027,458.221588 135.953064,448.171875
C115.821671,426.683105 98.183563,403.292694 89.988861,374.381683
C87.316971,364.955200 85.263985,354.979523 85.213455,345.243195
C84.825203,270.426117 84.999931,195.606094 85.000084,120.787003
C85.000099,111.945915 89.022926,107.378128 97.667000,104.911552
C110.303314,101.305817 122.758316,97.070175 135.348236,93.295280
C166.987869,83.808617 198.673264,74.474655 230.314102,64.992065
C251.955551,58.506241 273.501160,51.692432 295.219971,45.480957
C298.929932,44.419918 303.532532,44.845856 307.321503,45.955250
C336.096710,54.380497 364.763824,63.174244 393.481079,71.798355
C416.667694,78.761536 439.865509,85.688034 463.080963,92.554390
C477.383026,96.784447 491.770203,100.730957 506.034790,105.081421
C513.107544,107.238472 517.163025,114.049454 517.134766,120.030487
C516.897888,170.186081 517.000000,220.343277 517.000000,271.000000
z
"""

raw_path = parse_path(shield_d)
scale_factor = 1.15
bbox = raw_path.get_extents()
cx, cy = (bbox.x0 + bbox.x1) / 2, (bbox.y0 + bbox.y1) / 2

# calculate offsets to center the shield in axes space
tx = 0.5 - (scale_factor * cx / 601.0)
ty = 0.5 + (scale_factor * cy / 601.0)

# build an affine transform that scales, flips y, and recenters the shield
transform = (
    Affine2D()
    .scale(scale_factor / 601.0, -scale_factor / 601.0)
    .translate(tx, ty)
)

# apply the combined transform to create the final shield path
shield_path = raw_path.transformed(transform)


def plot_point(ax, coords, color, size, edgecolor, linewidth, zorder=1):
    """
    Plots points on the given axis.

    Parameters:
    - ax: Matplotlib axis object.
    - coords: 2D numpy array of shape (n, 2) with x, y coordinates.
    - color: Color of the points.
    - size: Size of the points.
    - edgecolor: Edge color of the points.
    - zorder: Drawing order.
    - linewidth: Width of the edge lines.
    """
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        color=color,
        edgecolor=edgecolor,
        s=size,
        linewidth=linewidth,
        zorder=zorder,
    )


def plot_comet(
    ax,
    start_coords,
    end_coords,
    pitch_color,
    event_color,
    linewidth,
    curvature=0.0,
    segments_per_unit=10.0,
    min_width=0.3,
    min_alpha=0.5,
    zorder=2,
):
    """
    Draw a comet-style gradient tail from each start point to its end point on a Matplotlib axis.

    Args:
        ax (matplotlib.axes.Axes): Axis to draw on.
        start_coords (ndarray[n,2]): Tail origins.
        end_coords   (ndarray[n,2]): Tail tips.
        pitch_color    (color): Color at the tail base.
        event_color    (color or list[color]): Tip color(s) for gradient.
        linewidth      (float): Max width at the tip.
        curvature      (float): 0.0 = straight, up to ~1.0 for strong bow.
        segments_per_unit (float): How many segments per data‐unit.
        min_width      (float): Fraction of linewidth at the base.
        min_alpha      (float): Starting alpha at the base.
        zorder         (int): Drawing order.
    """
    # 1) prep segment counts
    vectors = end_coords - start_coords
    lengths = np.hypot(vectors[:,0], vectors[:,1])
    lengths = np.nan_to_num(lengths)
    counts  = np.maximum(3, (lengths * segments_per_unit).astype(int))

    # 2) drop zero‐length tails
    mask = lengths > 0
    starts = start_coords[mask]
    ends   = end_coords[mask]
    counts = counts[mask]

    # 3) build one colormap for all tails
    if isinstance(event_color, (list, tuple)):
        cmap_colors = list(event_color)
    else:
        cmap_colors = [event_color, event_color]
    cmap = LinearSegmentedColormap.from_list("custom_gradient", cmap_colors)

    all_segments = []
    all_widths   = []
    all_colors   = []

    for P0, P2, n in zip(starts, ends, counts):
        # jitter identical endpoints ever so slightly
        P2 = P2.copy()
        if P0[0] == P2[0]: P2[0] += 1e-6
        if P0[1] == P2[1]: P2[1] += 1e-6

        # calculate a single quadratic‐Bezier path
        vec    = P2 - P0
        L      = np.hypot(*vec) or 1.0
        midpt  = (P0 + P2) / 2
        normal = np.array([-vec[1], vec[0]]) / L
        ctrl   = midpt + normal * (L * curvature)

        # sample t from 0 to 1
        t      = np.linspace(0, 1, n)[:,None]
        pts    = (1-t)**2 * P0 + 2*(1-t)*t * ctrl + t**2 * P2

        # break into little segments for LineCollection
        segs = [(pts[i-1], pts[i]) for i in range(1, n)]

        # taper width from base to tip
        widths = np.linspace(min_width*linewidth, linewidth, len(segs))

        # full-span color sampling, then fade alpha
        m = len(segs)
        ct = np.linspace(0, 1, m)
        al = np.linspace(min_alpha, 1, m)
        cols = cmap(ct)
        cols[:, -1] = al

        all_segments.extend(segs)
        all_widths.extend(widths)
        all_colors.extend(cols)

    # 4) draw everything at once
    lc = LineCollection(
        all_segments,
        linewidths=all_widths,
        colors=all_colors,
        transform=ax.transData,
        zorder=zorder,
    )
    ax.add_collection(lc)


def plot_dotted_line(
    ax, start_coords, end_coords, color, dot_size, dot_spacing, zorder=3
):
    """
    Plots dotted lines between start and end coordinates using interpolation for accuracy.

    Parameters:
    - ax: Matplotlib axis object.
    - start_coords: 2D numpy array of starting coordinates (n, 2).
    - end_coords: 2D numpy array of ending coordinates (n, 2).
    - color: Color of the dotted lines.
    - dot_size: Size of the dots.
    - dot_spacing: Spacing between dots.
    """
    all_dots_x = []
    all_dots_y = []

    for start, end in zip(start_coords, end_coords):
        if np.all(start == end):  # Skip zero-length lines
            continue

        # Calculate the vector and length of the line
        vector = end - start
        length = np.linalg.norm(vector)

        # Compute the number of dots and their positions along the line
        num_dots = int(length / dot_spacing)
        t_values = np.linspace(0, 1, num_dots)

        # Interpolate points along the line
        dots_x = start[0] + t_values * vector[0]
        dots_y = start[1] + t_values * vector[1]

        all_dots_x.extend(dots_x)
        all_dots_y.extend(dots_y)

    # Scatter plot for all dots
    ax.scatter(all_dots_x, all_dots_y, color=color, s=dot_size, zorder=zorder)

def plot_logo(color=futicolor.green, fill=False, format="png", tight=True):
    """
    plot the futi logo and save as png or svg if desired

    parameters
    - color: color of the shield and outline of the comet head
    - fill: whether to fill the background with dark color (if true) or leave it transparent (if false)
    - format: output format, either "png" or "svg"
    - tight: whether to trim whitespace around the figure when saving

    returns
    fig and ax objects for further customization
    """
    background = futicolor.dark if fill else "none"
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=background)
    ax.set_facecolor(background)
    ax.axis('off')
    ax.set_aspect('equal')

    # draw the shield using the chosen color
    ax.add_patch(PathPatch(
        shield_path,
        facecolor=color,
        edgecolor='none',
        zorder=0
    ))

    # draw the vertical stem of the f in dark color
    ax.plot([0.5, 0.5], [0.2, 0.55],
            color=futicolor.dark,
            linewidth=70,
            zorder=1)

    # draw the curved top of the f in dark color
    start = np.array((0.5, 0.6))
    end   = np.array((0.7, 0.8))
    baseline = end - start
    length = np.hypot(*baseline) or 1.0
    normal = np.array((-baseline[1], baseline[0])) / length
    one_third = start + baseline * 0.33
    two_thirds = start + baseline * 0.67
    bow = 0.095
    c1 = tuple(one_third + normal * bow)
    c2 = tuple(two_thirds + normal * bow)
    path = Path(
        [tuple(start), c1, c2, tuple(end)],
        [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    )
    ax.add_patch(PathPatch(
        path,
        edgecolor=futicolor.dark,
        linewidth=70,
        fill=False,
        zorder=1
    ))

    # draw the comet tail in dark color
    tail_start = np.array([[0.18, 0.25]])
    tail_end   = np.array([[0.78, 0.5]])
    for _ in range(4):
        plot_comet(
            ax,
            tail_start,
            tail_end,
            event_color=[futicolor.dark] * 5,
            pitch_color='none',
            linewidth=40,
            min_width=0.1,
            min_alpha=0.0,
            curvature=0.28,
            segments_per_unit=3000,
            zorder=2
        )

    # draw the head of the comet with dark fill and colored outline
    plot_point(
        ax,
        coords=tail_end,
        color=futicolor.dark,
        edgecolor=color,
        size=5000,
        linewidth=8,
        zorder=3
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if format in ("png", "svg"):
        # assemble savefig keyword arguments
        save_kwargs = {
            "format": format,
            "dpi": 1000,
            "transparent": not fill,
            "pad_inches": 0
        }
        if tight:
            save_kwargs["bbox_inches"] = "tight"

        fig.savefig(f"futi_logo.{format}", **save_kwargs)

    return fig, ax


def plot_player(
        ax, 
        data, 
        image_path, 
        bg_color, 
        size=6, 
        pitch=None,
        zorder=5
):
    """
    plot player headshots as clipped circles

    this function loads the image at `image_path` (either absolute/relative
    or just a filename in futiplot/resources), optionally transforms the
    raw (x_start, y_start) coordinates through `transform_xy` if you pass
    a pitch object, then for each row draws:

      • a filled circle of diameter `size` at the transformed point  
      • the headshot, clipped to that same circle, with its aspect ratio
        preserved

    parameters
    ----------
    ax : matplotlib.axes.Axes
        axis to draw on
    data : pandas.DataFrame
        must contain numeric 'x_start' and 'y_start'
    image_path : str or pathlib.Path
        full path or filename of the image
    bg_color : any matplotlib color
        fill color for the circular background
    size : float
        diameter of the headshot circle in data units
    pitch : futiplot.soccer.pitch.PlotPitch, optional
        if provided, raw coordinates will be transformed for pitch orientation
    zorder : int, default 5
        draw order for the background circle; image is drawn at zorder+1

    returns
    -------
    matplotlib.axes.Axes
        the same axis with headshots drawn
    """
    # load headshot
    img = imread(image_path)

    # compute a single aspect ratio factor
    img_h, img_w = img.shape[:2]
    aspect = img_w / img_h

    # radius of image
    radius = size / 2

    # half‐width and half‐height for the image extents
    half_h = radius
    half_w = radius * aspect

    # plot headshot
    df = transform_xy(data, pitch) if pitch is not None else data

    for _, row in df.iterrows():
        x, y = row["x_start"], row["y_start"]

        # draw the circular background
        circle = Circle(
            (x, y),
            radius,
            facecolor=bg_color,
            edgecolor="none",
            transform=ax.transData,
            zorder=zorder
        )
        ax.add_patch(circle)

        # place the image clipped to that circle
        xmin, xmax = x - half_w, x + half_w
        ymin, ymax = y - half_h, y + half_h
        im = ax.imshow(
            img,
            extent=(xmin, xmax, ymin, ymax),
            clip_path=circle,
            clip_on=True,
            zorder=zorder + 1
        )
        im.set_clip_path(circle)

    return ax