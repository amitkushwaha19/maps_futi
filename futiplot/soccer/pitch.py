import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arc, FancyBboxPatch
from matplotlib.collections import PatchCollection
from matplotlib.transforms import Affine2D
from importlib import resources

from futiplot.utils.colors import futicolor
from futiplot.utils.utils import transform_xy

class PlotPitch:
    """
    A class to construct and draw a soccer pitch using Matplotlib.

    Parameters:
        pitch_length (float): Length of the pitch in meters. Default is 105.
        pitch_width (float): Width of the pitch in meters. Default is 68.
        orientation (str): "tall" | "wide" | "top" | "bottom". Default is "tall".
        pitch_color (str): Background color. Default is futicolor.dark.
        line_color (str): Color of pitch markings. Default is futicolor.light.
        linewidth (float): Width of lines. Default is 1.
        spot_radius (float): Radius of center and penalty spots. Default is 0.2.
        buffer (float): Buffer around pitch in meters. Default is 5.
        markings (str): One of "futi", "round" or "all".
            "round" draws only the rounded outline, half-line, center circle, and 18-yd boxes.
            "futi" draws the trimmed half-line, center circle, 18-yd boxes, penalty spots, six-yd boxes,
                   penalty arcs and goals, and omits corner arcs and the center spot.
            "all" draws every marking in full (full half-line, center spot, corner arcs, etc.).
        logo (bool): Whether to overlay the futi logo at center for eligible markings. Default is True.
                     When True, the logo is shown for "futi" and "round"; never for "all".
    """

    def __init__(
        self,
        pitch_length=105,
        pitch_width=68,
        orientation="tall",
        pitch_color=futicolor.dark,
        line_color=futicolor.light,
        linewidth=1,
        spot_radius=0.2,
        buffer=5,
        markings="futi",
        logo=True,
    ):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.orientation = orientation
        self.pitch_color = pitch_color
        self.line_color = line_color
        self.linewidth = linewidth
        self.spot_radius = spot_radius
        self.buffer = buffer
        self.markings = markings
        self.logo = logo
        self.patches = []

        # fixed dimensions
        self.center_circle_radius = 9.15
        self.halfway_line = self.pitch_length / 2
        self.penalty_area_depth = 16.5
        self.penalty_area_width = 40.32
        self.penalty_spot_distance = 11
        self.six_yard_box_depth = 5.5
        self.six_yard_box_width = 18.32
        self.goal_width = 7.32
        self.goal_depth = 2
        self.corner_radius = 1

    def get_transform(self, ax):
        # Rotate for portrait-style orientations
        if self.orientation in ("tall", "top", "bottom"):
            return Affine2D().rotate_deg(90).translate(self.pitch_width, 0) + ax.transData
        return ax.transData


    def construct_pitch(self):
        self.patches = []

        if self.markings == "round":
            self.patches.append(
                FancyBboxPatch(
                    (0, 0), self.pitch_length, self.pitch_width,
                    boxstyle=f"round,pad=0,rounding_size={3*self.corner_radius}",
                    linewidth=self.linewidth, edgecolor=self.line_color, facecolor="none"
                )
            )

            # halfway line
            if self.orientation in ("top", "bottom"):
                # full line across for top/bottom
                self.patches.append(
                    Rectangle(
                        (self.halfway_line, 0),
                        0,
                        self.pitch_width,
                        linewidth=self.linewidth,
                        edgecolor=self.line_color,
                        facecolor="none"
                    )
                )
            else:
                # gap for center circle (original round style)
                y_mid = self.pitch_width / 2
                r     = self.center_circle_radius
                self.patches.append(
                    Rectangle(
                        (self.halfway_line, 0),
                        0,
                        y_mid - r,
                        linewidth=self.linewidth,
                        edgecolor=self.line_color,
                        facecolor="none"
                    )
                )
                self.patches.append(
                    Rectangle(
                        (self.halfway_line, y_mid + r),
                        0,
                        y_mid - r,
                        linewidth=self.linewidth,
                        edgecolor=self.line_color,
                        facecolor="none"
                    )
                )

            # center circle
            self.patches.append(
                Circle(
                    (self.halfway_line, self.pitch_width/2),
                    radius=self.center_circle_radius,
                    linewidth=self.linewidth,
                    edgecolor=self.line_color,
                    facecolor="none"
                )
            )

            # 18-yard boxes with rounded inner corners
            depth = self.penalty_area_depth
            height = self.penalty_area_width
            y0 = (self.pitch_width - height) / 2
            r = 3 * self.corner_radius

            def make_18yd_box(x_goal, inward):
                x_center = x_goal + inward*(depth - r)
                x_vert = x_goal + inward*depth
                x_start = x_goal if inward > 0 else x_center
                if inward > 0:
                    low1, low2, top1, top2 = 270, 360, 0, 90
                else:
                    low1, low2, top1, top2 = 180, 270, 90, 180
                segments = [
                    Rectangle((x_start, y0), depth - r, 0, linewidth=self.linewidth,
                              edgecolor=self.line_color, facecolor="none"),
                    Arc((x_center, y0 + r), 2*r, 2*r, angle=0,
                        theta1=low1, theta2=low2,
                        linewidth=self.linewidth, color=self.line_color),
                    Rectangle((x_vert, y0 + r), 0, height - 2*r,
                              linewidth=self.linewidth, edgecolor=self.line_color, facecolor="none"),
                    Arc((x_center, y0 + height - r), 2*r, 2*r, angle=0,
                        theta1=top1, theta2=top2,
                        linewidth=self.linewidth, color=self.line_color),
                    Rectangle((x_start, y0 + height), depth - r, 0,
                              linewidth=self.linewidth, edgecolor=self.line_color, facecolor="none"),
                ]
                box = PatchCollection(segments, match_original=True, snap=True)
                box.set_capstyle("round")
                box.set_joinstyle("round")
                return box

            self.patches.append(make_18yd_box(0, +1))
            self.patches.append(make_18yd_box(self.pitch_length, -1))
            return

        # outline for futi and all
        self.patches.append(
            Rectangle(
                (0, 0), self.pitch_length, self.pitch_width,
                linewidth=self.linewidth,
                edgecolor=self.line_color,
                facecolor="none"
            )
        )

        # halfway line
        if self.markings == "all" or self.orientation in ("top", "bottom"):
            # full line across for all/top/bottom
            self.patches.append(
                Rectangle(
                    (self.halfway_line, 0), 0, self.pitch_width,
                    linewidth=self.linewidth,
                    edgecolor=self.line_color,
                    facecolor="none"
                )
            )
        else:
            # trimmed around center circle (original 'futi' behavior)
            y_mid = self.pitch_width/2
            r = self.center_circle_radius
            self.patches.append(
                Rectangle((self.halfway_line, 0), 0, y_mid - r,
                          linewidth=self.linewidth,
                          edgecolor=self.line_color, facecolor="none")
            )
            self.patches.append(
                Rectangle((self.halfway_line, y_mid + r), 0, y_mid - r,
                          linewidth=self.linewidth,
                          edgecolor=self.line_color, facecolor="none")
            )

        # center circle
        self.patches.append(
            Circle(
                (self.halfway_line, self.pitch_width/2),
                radius=self.center_circle_radius,
                linewidth=self.linewidth,
                edgecolor=self.line_color,
                facecolor="none"
            )
        )

        # center spot only in "all"
        if self.markings == "all":
            self.patches.append(
                Circle(
                    (self.halfway_line, self.pitch_width/2),
                    radius=self.spot_radius,
                    color=self.line_color
                )
            )

        # 18-yard boxes
        self.patches.append(
            Rectangle(
                (0, (self.pitch_width - self.penalty_area_width)/2),
                self.penalty_area_depth, self.penalty_area_width,
                linewidth=self.linewidth, edgecolor=self.line_color, facecolor="none"
            )
        )
        self.patches.append(
            Rectangle(
                (self.pitch_length - self.penalty_area_depth,
                 (self.pitch_width - self.penalty_area_width)/2),
                self.penalty_area_depth, self.penalty_area_width,
                linewidth=self.linewidth, edgecolor=self.line_color, facecolor="none"
            )
        )

        # penalty spots
        for x in (self.penalty_spot_distance, self.pitch_length - self.penalty_spot_distance):
            self.patches.append(
                Circle((x, self.pitch_width/2),
                       radius=self.spot_radius,
                       color=self.line_color)
            )

        # six-yard boxes
        for x in (0, self.pitch_length - self.six_yard_box_depth):
            self.patches.append(
                Rectangle(
                    (x, (self.pitch_width - self.six_yard_box_width)/2),
                    self.six_yard_box_depth, self.six_yard_box_width,
                    linewidth=self.linewidth,
                    edgecolor=self.line_color,
                    facecolor="none"
                )
            )

        # penalty arcs
        arc_w = 2*self.center_circle_radius
        for center in (self.penalty_spot_distance, self.pitch_length - self.penalty_spot_distance):
            theta1, theta2 = (
                (-53.13, 53.13) if center < self.halfway_line else (126.87, 233.13)
            )
            self.patches.append(
                Arc((center, self.pitch_width/2),
                    width=arc_w, height=arc_w,
                    angle=0, theta1=theta1, theta2=theta2,
                    linewidth=self.linewidth, color=self.line_color)
            )

        # corner arcs only in "all"
        if self.markings == "all":
            for x, y, start, end in (
                (0, 0,   0,  90),
                (0, self.pitch_width, 270, 360),
                (self.pitch_length, 0,   90, 180),
                (self.pitch_length, self.pitch_width, 180, 270),
            ):
                self.patches.append(
                    Arc((x, y),
                        width=2*self.corner_radius,
                        height=2*self.corner_radius,
                        angle=0, theta1=start, theta2=end,
                        linewidth=self.linewidth, color=self.line_color)
                )

        # goals
        for x in (-self.goal_depth, self.pitch_length):
            self.patches.append(
                Rectangle(
                    (x, (self.pitch_width - self.goal_width)/2),
                    self.goal_depth, self.goal_width,
                    linewidth=self.linewidth,
                    edgecolor=self.line_color,
                    facecolor="none"
                )
            )

    def draw(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)

        transform = self.get_transform(ax)
        ax.set_facecolor(self.pitch_color)

        # overlay the futi logo only for full-field portrait/round/futi (not top/bottom)
        if (self.logo
            and self.markings in ("futi", "round")
            and self.orientation not in ("top", "bottom")):
            df_center = pd.DataFrame({
                'x_start': [self.halfway_line],
                'y_start': [self.pitch_width/2]
            })
            center_plot = transform_xy(df_center, pitch=self, flip_coords=False)
            x0 = center_plot.loc[0, 'x_start']
            y0 = center_plot.loc[0, 'y_start']
            r = self.center_circle_radius * 0.75

            with resources.path("futiplot.resources", "futi_logo.png") as logo_path:
                logo_img = mpimg.imread(logo_path)

            ax.imshow(
                logo_img,
                extent=(x0-r, x0+r, y0-r, y0+r),
                aspect="auto",
                alpha=0.12,
                zorder=0
            )

        # add all patches
        for art in self.patches:
            art.set_transform(transform)
            if isinstance(art, PatchCollection):
                ax.add_collection(art)
            else:
                ax.add_patch(art)

        # set limits
        if self.orientation in ("tall", "top", "bottom"):
            ax.set_xlim(-self.buffer, self.pitch_width + self.buffer)
            eps = 0.2
            if self.orientation == "top":
                ax.set_ylim(self.halfway_line - eps, self.pitch_length + self.buffer)
            elif self.orientation == "bottom":
                ax.set_ylim(-self.buffer, self.halfway_line + eps)   # <- add eps here
            else:
                ax.set_ylim(-self.buffer, self.pitch_length + self.buffer)
        else:
            ax.set_xlim(-self.buffer, self.pitch_length + self.buffer)
            ax.set_ylim(-self.buffer, self.pitch_width + self.buffer)

        ax.set_aspect("equal")
        ax.axis("off")
        return ax

def plot_pitch(
    ax=None,
    figsize=None,
    dpi=100,
    pitch_length=105,
    pitch_width=68,
    orientation="tall",
    pitch_color=futicolor.dark,
    line_color=futicolor.light,
    linewidth=1,
    spot_radius=0.2,
    buffer=5,
    markings="futi",
    show_legend=False,
    logo=True,
):
    # choose a default size if none was provided
    if figsize is None:
        tall_size = (10.8, 19.2)   # (width, height)
        if orientation == "tall":
            figsize = tall_size
        elif orientation in ("top", "bottom"):
            # half the height of tall: 19.2 → 9.6
            figsize = (tall_size[0], tall_size[1] * 0.5)  # (10.8, 9.6)
        else:  # "wide"
            figsize = (19.2, 10.8)

    # carve out two rows for pitch + legend
    if show_legend:
        fig = plt.figure(figsize=figsize, facecolor=pitch_color, dpi=dpi)
        gs  = fig.add_gridspec(2, 1, height_ratios=[9, 1], hspace=0.05)
        ax_pitch  = fig.add_subplot(gs[0])
        ax_legend = fig.add_subplot(gs[1])
        ax_pitch.set_facecolor(pitch_color)
    else:
        # single axis fallback
        if ax is None:
            fig, ax_pitch = plt.subplots(
                figsize=figsize,
                facecolor=pitch_color,
                dpi=dpi
            )
            ax_pitch.set_facecolor(pitch_color)
        else:
            fig      = ax.figure
            ax_pitch = ax
        ax_legend = None

    # draw the pitch
    pitch = PlotPitch(
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        orientation=orientation,
        pitch_color=pitch_color,
        line_color=line_color,
        linewidth=linewidth,
        spot_radius=spot_radius,
        buffer=buffer,
        markings=markings,
        logo=logo,
    )
    pitch.construct_pitch()
    pitch.draw(ax_pitch)

    # draw the legend if requested
    if show_legend:
        from futiplot.format.legends import draw_event_legend
        draw_event_legend(
            ax_legend,
            orientation="horizontal",
            symbol_color=line_color,
            label_color=line_color,
            label_fontsize=14,
        )
        return fig, ax_pitch, pitch, ax_legend

    return fig, ax_pitch, pitch
