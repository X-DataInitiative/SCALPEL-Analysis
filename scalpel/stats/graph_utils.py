# License: BSD 3 clause

import seaborn as sns


def _get_string_maps(max_age):
    age_lists = range(0, max_age, 5)
    buckets = zip(age_lists[:-1], age_lists[1:])
    string_maps = {
        i: "[{}, {}[".format(bucket[0], bucket[1]) for (i, bucket) in enumerate(buckets)
    }
    return string_maps


def _get_color_maps(max_age):
    age_lists = range(0, max_age, 5)
    size = len(age_lists)
    buckets = zip(age_lists[:-1], age_lists[1:])
    palette = sns.color_palette("Paired", n_colors=size)
    return {
        "[{}, {}[".format(bucket[0], bucket[1]): palette[i]
        for (i, bucket) in enumerate(buckets)
    }


BUCKET_INTEGER_TO_STR = _get_string_maps(250)
BUCKET_STR_TO_COLOR = _get_color_maps(250)
GENDER_MAPPING = {1: "Homme", 2: "Femme"}


def format_title(string, every=64):
    lines = []
    length = len(string)
    i = every
    while i < length:
        if string[i - 1] == " ":
            lines.append(string[i - every : i])
        else:
            lines.append(string[i - every : i] + "-")
        i += every
    else:
        lines.append(string[i - every :])
    return "\n".join(lines)
