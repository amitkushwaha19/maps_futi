from matplotlib import rcParams

# import fonts
from .utils.fontutils import (
    get_font,
    get_default_font_name,
    autosize_text,
    story_title,
)

# set Inter as the global default font
rcParams['font.family'] = get_default_font_name()

# import utilities
from .utils.utils import load_sample_data, transform_xy, get_zones

# import database stuff
from .utils.database import db_connect

# import colors
from .utils.colors import futicolor

# import plotting functions
from .utils.plotting import plot_point, plot_comet, plot_dotted_line, plot_logo

# import pitch plotting
from .soccer.pitch import PlotPitch, plot_pitch

# import event plotting
from .soccer.events import (
    plot_ball_kicked,
    plot_ball_carried,
    plot_ball_static,
    plot_events,
    plot_passage,
)

# import heatmap
from .soccer.heatmap import (
    get_heatmap,
    plot_heatmap,
    plot_territory,
    plot_zscore_heatmap,
)

# import momentum
from .soccer.momentum import plot_momentum

# import xg step
from .soccer.xg_step import plot_xg_step

# import shotmap
from .soccer.shotmap import plot_shotmap

# define the public api
__all__ = [
    "load_sample_data",
    "transform_xy",
    "get_zones",
    "db_connect",
    "get_font",
    "autosize_text",
    "story_title",
    "futicolor",
    "plot_point",
    "plot_comet",
    "plot_dotted_line",
    "plot_logo",
    "PlotPitch",
    "plot_pitch",
    "plot_ball_kicked",
    "plot_ball_carried",
    "plot_ball_static",
    "plot_events",
    "plot_passage",
    "get_heatmap",
    "plot_heatmap",
    "plot_territory",
    "plot_zscore_heatmap",
    "plot_momentum",
    "plot_xg_step",
    "plot_shotmap",
]
