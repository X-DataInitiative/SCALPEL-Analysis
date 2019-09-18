import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.exploration.core.cohort import Cohort
from src.exploration.core.decorators import title, xlabel, ylabel

registry = []


def register(f):
    registry.append(f)
    return f


@register
@xlabel("Number of Days")
@ylabel("Count (log)")
@title("duration distribution")
def plot_co_events(cohort: Cohort, figure: Figure, minimum_count=300) -> Figure:
    events = cohort.events
    events_2 = (
        events.select(["patientID", "value", "start", "end"])
        .withColumnRenamed("start", "start2")
        .withColumnRenamed("end", "end2")
        .withColumnRenamed("value", "value2")
    )
    cond = [
        events.patientID == events_2.patientID,
        events.start.between(events_2.start2, events_2.end2),
        events.value != events_2.value2,
    ]
    result = events.join(events_2, cond).groupBy(["value", "value2"]).count().toPandas()
    data = pd.pivot_table(
        result[result["count"] > minimum_count],
        values="count",
        index="value",
        columns="value2",
    )

    fig, ax = plt.subplots(figsize=(16, 16))
    ax2 = ax.twiny()

    im = ax.pcolor(data.values, cmap="YlGnBu", edgecolors="k", linewidths=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)
    ax.set_yticks(np.arange(len(data.index)) + 0.5)
    ax.set_yticklabels(data.index)
    ax.set_xticks(np.arange(len(data.columns)) + 0.5)
    ax.set_xticklabels(data.columns)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    im = ax2.pcolor(data.values, cmap="YlGnBu", edgecolors="k", linewidths=1)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax2.set_xticks(np.arange(len(data.columns)) + 0.5)
    ax2.set_xticklabels(data.columns)
    plt.setp(
        ax2.get_xticklabels(),
        rotation=45,
        ha="left",
        rotation_mode="anchor",
        va="center",
    )
    ax2.set_xlabel("While taking")
    ax.set_ylabel("Begin taking")
    ax2.grid(False)

    return figure
