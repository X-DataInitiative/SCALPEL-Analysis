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
@xlabel("Event count")
@ylabel("Subjects Counts")
@title("distinct event per patient")
def plot_unique_events_type_per_patient_distribution(figure: Figure, cohort: Cohort) -> Figure:
    group_columns = ["patientID", "value"]
    data = agg(cohort.events, frozenset(group_columns), "count")
    data = _get_distinct(data, group_columns)
    data = data.groupby("patientID").count().reset_index().groupby("value").count()
    sns.barplot(x=data.index.values, y=data.patientID.values)
    return figure


