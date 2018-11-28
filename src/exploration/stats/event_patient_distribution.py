import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from src.exploration.core.cohort import Cohort
from src.exploration.core.decorators import title, ylabel
from src.exploration.stats.grouper import agg

registry = []


def register(f):
    registry.append(f)
    return f


def _get_distinct(data, group_columns) -> pd.DataFrame:
    return data[group_columns].drop_duplicates()


@register
@ylabel("Number of subjects")
@title("Number of distinct events per subject")
def plot_unique_event_distribution_per_patient(
    figure: Figure, cohort: Cohort
) -> Figure:
    group_columns = ["patientID", "value"]
    data = agg(cohort.events, frozenset(group_columns), "count")
    data = _get_distinct(data, group_columns)
    data = data.groupby("patientID").count().reset_index().groupby("value").count()
    sns.barplot(x=data.index.values, y=data.patientID.values, color="salmon")
    plt.xticks(rotation=90)
    plt.xlabel("Number of distinct {}".format(cohort.name))
    return figure


@register
@ylabel("Number of distinct patients")
@title("Number of distinct patients count per event")
def plot_patient_distribution_per_unique_event(
    figure: Figure, cohort: Cohort
) -> Figure:
    group_columns = ["patientID", "value"]
    data = _get_distinct(cohort.events, group_columns)
    data = agg(data[group_columns], frozenset(["value"]), "count")
    sns.barplot(x=data.value.values, y=data["count(1)"].values, color="salmon")
    plt.xticks(rotation=90)
    plt.xlabel("{} names".format(cohort.name.title()))
    return figure
