"""
A Plotter is a function that takes a pandas Series and transform it to the desired
plot.
"""
from abc import ABC, abstractmethod
from typing import Callable

import pandas as pd
from matplotlib.axes import Axes


def plot_line(data: pd.Series, ax: Axes) -> Axes:
    ax.plot(data.index.to_pydatetime(), data.values)
    return ax


def plot_bars(data: pd.Series, ax: Axes) -> Axes:
    ax.bar(x=range(len(data.values)), height=data.values)
    return ax


class Plotter(ABC):
    @property
    @abstractmethod
    def plotter(self) -> Callable:
        pass

    @property
    @abstractmethod
    def patch(self) -> bool:
        pass


class LinePlotter(Plotter):
    @property
    def plotter(self) -> Callable:
        return plot_line

    @property
    def patch(self):
        return False


class BarPlotter(Plotter):
    @property
    def plotter(self) -> Callable:
        return plot_bars

    @property
    def patch(self):
        return True
