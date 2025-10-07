from matplotlib.font_manager import FontProperties
from importlib import resources
from .colors import futicolor

def get_font(weight="regular"):
    weight = weight.lower()
    filename = {
        "regular": "Inter-Regular.ttf",
        "semibold": "Inter-SemiBold.ttf",
        "bold": "Inter-Bold.ttf"
    }.get(weight, "Inter-Regular.ttf")

    with resources.path("futiplot.fonts", filename) as font_path:
        return FontProperties(fname=str(font_path))

def get_default_font_name():
    """Returns the actual name of the Inter font as seen by matplotlib."""
    return get_font("regular").get_name()

def autosize_text(fig, text, x_start=0.05, x_end=0.95, weight="regular", max_size=60, min_size=24):
    """
    Dynamically finds the largest font size that fits `text` between `x_start` and `x_end` in figure coordinates.
    If text doesn't fit, it tries splitting it into two lines.
    Returns (FontProperties, fontsize, possibly modified multiline text).
    """
    font = get_font(weight)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_width_px = fig.bbox.width

    def fits(text_str, fontsize):
        temp = fig.text(x_start, 0.5, text_str, fontproperties=font, fontsize=fontsize, alpha=0)
        bbox = temp.get_window_extent(renderer=renderer)
        temp.remove()
        text_width_norm = bbox.width / fig_width_px
        return text_width_norm <= (x_end - x_start)

    # Try full line first
    for size in range(max_size, min_size - 1, -1):
        if fits(text, size):
            return font, size, text

    # Try two-line fallback
    words = text.split()
    mid = len(words) // 2
    split_text = " ".join(words[:mid]) + "\n" + " ".join(words[mid:])

    for size in range(max_size, min_size - 1, -1):
        if fits(split_text, size):
            return font, size, split_text

    return font, min_size, split_text

from .colors import futicolor

def story_title(fig, text, max_size=60, min_size=14):
    """
    Adds a bold, autosized title near the top-left of a story-style figure.
    Returns the created Text object.
    """
    font, size, final_text = autosize_text(
        fig,
        text,
        x_start=0.05,
        x_end=0.95,
        weight="bold",
        max_size=max_size,
        min_size=min_size,
    )

    return fig.text(
        0.05, 0.85,
        final_text,
        fontproperties=font,
        fontsize=size,
        color=futicolor.light,
        ha="left",
        va="bottom"
    )
