from matplotlib.figure import Figure

from src.exploration.core.cohort import Cohort
from src.exploration.core.decorators import title
from src.exploration.stats.time_distribution import plot_events_per_month_as_bars, \
    plot_events_per_week_as_bars, plot_events_per_day_as_bars, \
    plot_events_per_month_as_timeseries, plot_events_per_week_as_timeseries, \
    plot_events_per_day_as_timeseries

registry = []


def register(f):
    registry.append(f)
    return f


@register
@title("Exposures start per month")
def plot_exposures_per_month_as_bars(figure: Figure, cohort: Cohort) -> Figure:
    f = plot_events_per_month_as_bars(figure, cohort)
    return f


@register
@title("Exposures start per month")
def plot_exposures_per_week_as_bars(figure: Figure, cohort: Cohort) -> Figure:
    plot_events_per_week_as_bars(figure, cohort)
    return figure


@register
@title("Exposures start per month")
def plot_exposures_per_day_as_bars(figure: Figure, cohort: Cohort) -> Figure:
    plot_events_per_day_as_bars(figure, cohort)
    return figure


@register
@title("Exposures start per month")
def plot_exposures_per_month_as_timeseries(figure: Figure, cohort: Cohort) -> Figure:
    plot_events_per_month_as_timeseries(figure, cohort)
    return figure


@register
@title("Exposures start per month")
def plot_exposures_per_week_as_timeseries(figure: Figure, cohort: Cohort) -> Figure:
    plot_events_per_week_as_timeseries(figure, cohort)
    return figure


@register
@title("Exposures start per month")
def plot_exposures_per_day_as_timeseries(figure: Figure, cohort: Cohort) -> Figure:
    plot_events_per_day_as_timeseries(figure, cohort)
    return figure
