from matplotlib.figure import Figure

from src.exploration.core.cohort import Cohort
from src.exploration.core.decorators import xlabel, ylabel
from src.exploration.stats.grouper import Aggregator, event_start_agg
from src.exploration.stats.time_decorator import DayCounterBar, DayCounterLine, \
    MonthCounterBar, MonthCounterLine, WeekCounterBar, WeekCounterLine


@xlabel("Month")
@ylabel("Count")
def plot_events_per_month_as_bars(figure: Figure, cohort: Cohort) -> Figure:
    return MonthCounterBarEvent()(figure, cohort)


@xlabel("Week")
@ylabel("Count")
def plot_events_per_week_as_bars(figure: Figure, cohort: Cohort) -> Figure:
    return WeekCounterBarEvent()(figure, cohort)


@xlabel("Day")
@ylabel("Count")
def plot_events_per_day_as_bars(figure: Figure, cohort: Cohort) -> Figure:
    return DayCounterBarEvent()(figure, cohort)


@xlabel("Month")
@ylabel("Count")
def plot_events_per_month_as_timeseries(figure: Figure, cohort: Cohort) -> Figure:
    return MonthCounterLineEvent()(figure, cohort)


@xlabel("Week")
@ylabel("Count")
def plot_events_per_week_as_timeseries(figure: Figure, cohort: Cohort) -> Figure:
    return WeekCounterLineEvent()(figure, cohort)


@xlabel("Day")
@ylabel("Count")
def plot_events_per_day_as_timeseries(figure: Figure, cohort: Cohort) -> Figure:
    return DayCounterLineEvent()(figure, cohort)


class EventStartAgg(Aggregator):
    @property
    def agg(self): return event_start_agg


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
