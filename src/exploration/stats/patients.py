import seaborn as sns

from src.exploration.core.decorators import save_plots, title, xlabel, ylabel
from src.exploration.stats.graph_utils import (
    BUCKET_INTEGER_TO_STR,
    BUCKET_STR_TO_COLOR,
    GENDER_MAPPING,
)
from src.exploration.stats.grouper import agg

registry = []


def register(f):
    registry.append(f)
    return f


@register
@title("Gender distribution")
@xlabel("Gender")
@ylabel("Subjects count")
def distribution_by_gender(figure, cohort):
    df = agg(cohort.subjects, frozenset(["gender"]), "count")
    ax = figure.gca()
    ax = sns.barplot(x=df["gender"].values, y=df["count(1)"].values, ax=ax)
    ax.set_xticklabels(["Male", "Female"])
    return figure


@register
@title("Age bucket distribution")
@xlabel("Age bucket")
@ylabel("Subjects count")
def distribution_by_age_bucket(figure, cohort):
    df = agg(cohort.subjects, frozenset(["ageBucket"]), "count").sort_values(
        "ageBucket"
    )
    df.ageBucket = df.ageBucket.map(lambda x: BUCKET_INTEGER_TO_STR[x])
    buckets = df.ageBucket.values.tolist()
    colors = [BUCKET_STR_TO_COLOR[bucket] for bucket in buckets]
    ax = figure.gca()
    ax = sns.barplot(
        x=df["ageBucket"].values, y=df["count(1)"].values, ax=ax, palette=colors
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    return figure


@register
@title("Gender and age bucket distribution")
@xlabel("Age bucket")
@ylabel("Subjects count")
def distribution_by_gender_age_bucket(figure, cohort):
    df = agg(cohort.subjects, frozenset(["gender", "ageBucket"]), "count").sort_values(
        ["gender", "ageBucket"]
    )
    df.ageBucket = df.ageBucket.map(lambda x: BUCKET_INTEGER_TO_STR[x])
    ax = figure.gca()
    ax = sns.barplot(
        x=df["ageBucket"].values,
        y=df["count(1)"].values,
        hue=df["gender"].values,
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.legend(title="Gender")
    legend = ax.get_legend()
    [
        label.set_text(GENDER_MAPPING[int(label.get_text())])
        for label in legend.get_texts()
    ]
    return figure


def save_patients_stats(path, cohort):
    save_plots(registry, path, cohort)
