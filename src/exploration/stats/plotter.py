import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes


def _plot_line(data: pd.Series, ax: Axes) -> Axes:
    data.plot(kind="line", color=sns.xkcd_rgb["pumpkin orange"])
    return ax


def _plot_bars(data: pd.Series, ax: Axes) -> Axes:
    data.plot(kind="bar", color=sns.xkcd_rgb["pumpkin orange"])
    return ax