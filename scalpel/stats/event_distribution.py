# License: BSD 3 clause

from matplotlib.figure import Figure

from scalpel.core.cohort import Cohort
from scalpel.stats.decorators import title, xlabel, ylabel
from scalpel.stats.grouper import Aggregator, event_start_agg
from scalpel.stats.time_decorator import (
    DayCounterBar,
    DayCounterLine,
    MonthCounterBar,
    MonthCounterLine,
    WeekCounterBar,
    WeekCounterLine,
)

registry = []


def register(f):
    registry.append(f)
    return f


@register
@xlabel("Month")
@ylabel("Count")
@title("distribution per month")
def plot_events_per_month_as_bars(figure: Figure, cohort: Cohort) -> Figure:
    return MonthCounterBarEvent()(figure, cohort)


@register
@xlabel("Week")
@ylabel("Count")
@title("distribution per week")
def plot_events_per_week_as_bars(figure: Figure, cohort: Cohort) -> Figure:
    return WeekCounterBarEvent()(figure, cohort)


@register
@xlabel("Day")
@ylabel("Count")
@title("distribution per day")
def plot_events_per_day_as_bars(figure: Figure, cohort: Cohort) -> Figure:
    return DayCounterBarEvent()(figure, cohort)


@register
@xlabel("Month")
@ylabel("Count")
@title("distribution per month")
def plot_events_per_month_as_timeseries(figure: Figure, cohort: Cohort) -> Figure:
    return MonthCounterLineEvent()(figure, cohort)


@register
@xlabel("Week")
@ylabel("Count")
@title("distribution per week")
def plot_events_per_week_as_timeseries(figure: Figure, cohort: Cohort) -> Figure:
    return WeekCounterLineEvent()(figure, cohort)


@register
@xlabel("Day")
@ylabel("Count")
@title("distribution per day")
def plot_events_per_day_as_timeseries(figure: Figure, cohort: Cohort) -> Figure:
    return DayCounterLineEvent()(figure, cohort)


class EventStartAgg(Aggregator):
    @property
    def agg(self):
        return event_start_agg


class MonthCounterBarEvent(MonthCounterBar, EventStartAgg):
    pass


class WeekCounterBarEvent(WeekCounterBar, EventStartAgg):
    pass


class DayCounterBarEvent(DayCounterBar, EventStartAgg):
    pass


class MonthCounterLineEvent(MonthCounterLine, EventStartAgg):
    pass


class WeekCounterLineEvent(WeekCounterLine, EventStartAgg):
    pass


class DayCounterLineEvent(DayCounterLine, EventStartAgg):
    pass
