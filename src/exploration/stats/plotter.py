"""
A Plotter is a function that takes a pandas Series and transform it to the desired
plot.
"""

import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes


def plot_line(data: pd.Series, ax: Axes) -> Axes:
    data.plot(kind="line", color=sns.xkcd_rgb["pumpkin orange"])
    return ax


def plot_bars(data: pd.Series, ax: Axes) -> Axes:
    ax.bar(x=range(len(data.values)), height=data.values)
    #sns.barplot(x=data.index, y=data.values, color=sns.xkcd_rgb["pumpkin orange"])
    return ax
