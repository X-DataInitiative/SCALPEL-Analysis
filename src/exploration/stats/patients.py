import seaborn as sns

from src.exploration.core.decorators import register, save_plots

registry = []


def _get_string_maps(max_age):
    age_lists = range(0, max_age, 5)
    buckets = zip(age_lists[:-1], age_lists[1:])
    string_maps = {i: "[{}, {}[".format(bucket[0], bucket[1]) for (
        i, bucket) in enumerate(buckets)}
    return string_maps


def _get_color_maps(max_age):
    age_lists = range(0, max_age, 5)
    size = len(age_lists)
    buckets = zip(age_lists[:-1], age_lists[1:])
    palette = sns.color_palette("Paired", n_colors=size)
    return {"[{}, {}[".format(bucket[0], bucket[1]): palette[i] for (
        i, bucket) in enumerate(buckets)}


BUCKET_INTEGER_TO_STR = _get_string_maps(120)
BUCKET_STR_TO_COLOR = _get_color_maps(120)
GENDER_MAPPING = {1: "Homme", 2: "Femme"}


@register
def distribution_by_gender(figure, cohort):
    df = cohort.subjects.groupBy("gender").count().alias(
        "num_patients").toPandas()
    ax = figure.gca()
    ax = sns.barplot(x="gender", data=df, y="count", ax=ax)
    ax.set_xticklabels(["Homme", "Femme"])
    ax.set_ylabel("Nombre de patients")
    ax.set_xlabel("Genre")
    ax.set_title("Distribution des {}\nsuivant le genre".format(cohort.characteristics))
    return figure


@register
def distribution_by_age_bucket(figure, cohort):
    df = cohort.subjects.groupBy("ageBucket").count().alias(
        "num_patients").orderBy("ageBucket").toPandas()
    df.ageBucket = df.ageBucket.map(
        lambda x: BUCKET_INTEGER_TO_STR[x])
    buckets = df.ageBucket.values.tolist()
    colors = [BUCKET_STR_TO_COLOR[bucket] for bucket in buckets]
    ax = figure.gca()
    ax = sns.barplot(x="ageBucket", data=df, y="count", ax=ax, palette=colors)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_ylabel("Nombre de patients")
    ax.set_xlabel("Tranche d'age")
    return figure


@register
def distribution_by_gender_age_bucket(figure, cohort):
    df = cohort.subjects.groupBy("gender", "ageBucket").count().alias(
        "num_patients").orderBy("gender", "ageBucket").toPandas()
    df.ageBucket = df.ageBucket.map(
        lambda x: BUCKET_INTEGER_TO_STR[x])
    ax = figure.gca()
    ax = sns.barplot(x="ageBucket", y="count",
                     hue="gender", data=df, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_ylabel("Nombre de patients")
    ax.set_xlabel("Tranche d'age")
    ax.set_title("Distribution des {}\nsuivant le genre et la tranche d'age".format(
        cohort.characteristics
    ))

    ax.legend(loc=1, title="Genre")
    legend = ax.get_legend()
    [label.set_text(GENDER_MAPPING[int(label.get_text())])
     for label in legend.get_texts()]
    return figure


def save_patients_stats(path, cohort):
    save_plots(registry, path, cohort)

