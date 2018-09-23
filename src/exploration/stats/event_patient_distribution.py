import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from src.exploration.core.cohort import Cohort
from src.exploration.core.decorators import xlabel, ylabel, title
from src.exploration.stats.grouper import agg

registry = []


def register(f):
    registry.append(f)
    return f


def _get_distinct(data, group_columns) -> pd.DataFrame:
    return data[group_columns].drop_duplicates()


@register
@xlabel("Events count")
@ylabel("Subjects count")
@title("Distinct events count per patient")
def plot_unique_event_distribution_per_patient(figure: Figure, cohort: Cohort) -> Figure:
    group_columns = ["patientID", "value"]
    data = agg(cohort.events, frozenset(group_columns), "count")
    data = _get_distinct(data, group_columns)
    data = data.groupby("patientID").count().reset_index().groupby("value").count()
    sns.barplot(x=data.index.values, y=data.patientID.values)
    return figure


@register
@xlabel("Subjects count")
@ylabel("Events count")
@title("Distinct patients count per event")
def plot_patient_distribution_per_unique_event(figure: Figure, cohort: Cohort) -> Figure:
    group_columns = ["patientID", "value"]
    data = _get_distinct(cohort.events, group_columns)
    data = agg(data[group_columns], frozenset(["value"]), "count")
    sns.barplot(x=data.value.values, y=data["count(1)"].values)
    return figure
