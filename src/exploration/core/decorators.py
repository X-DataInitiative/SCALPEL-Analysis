import math
from functools import partial, wraps

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.backends.backend_pdf import PdfPages

from src.exploration.core.io import get_logger
from src.exploration.stats.graph_utils import format_title


def attach_wrapper(obj, func=None):
    if func is None:
        return partial(attach_wrapper, obj)
    setattr(obj, func.__name__, func)
    return func


def logged(level, message=None):
    """
    Add logging to a function.  level is the logging
    level, name is the logger name, and message is the
    log message.  If name and message aren't specified,
    they default to the function's module and name.
    """

    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            log = get_logger()
            logmsg = message if message else func.__name__
            log.log(level, logmsg)
            return func(*args, **kwargs)

        return wrapper

    return decorate


def saver(pdf, figsize=(8, 5)):
    def decorate(f):
        def save_result(*args, **kwargs):
            fig = plt.figure(figsize=figsize)
            f(figure=fig, *args, **kwargs)
            plt.tight_layout()
            pdf.savefig(fig)

        return save_result

    return decorate


def xlabel(label):
    def decorate(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            ax = plt.gca()
            ax.set_xlabel(label)
            return f(*args, **kwargs)

        return wrapper

    return decorate


def ylabel(label):
    def decorate(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            ax = plt.gca()
            ax.set_ylabel(label)
            return f(*args, **kwargs)

        return wrapper

    return decorate


def ylabel_fontsize(size):
    def decorate(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            ax = plt.gca()
            [item.set_fontsize(size) for item in ax.get_yticklabels()]
            return f(*args, **kwargs)

        return wrapper

    return decorate


def show_labels(frequency):
    def decorate(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            ax = plt.gca()
            for i, label in enumerate(ax.xaxis.get_ticklabels()):
                if i % frequency == 0:
                    label.set_visible(True)
                else:
                    label.set_visible(False)
            return f(*args, **kwargs)

        return wrapper

    return decorate


def title(header):
    def decorate(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            ax = plt.gca()

            try:
                cohort = args[1]
            except IndexError:
                cohort = kwargs["cohort"]

            formatted_title = format_title(
                "{} among {}".format(header, cohort.characteristics)
            )

            ax.set_title(formatted_title)
            return f(*args, **kwargs)

        return wrapper

    return decorate


def percentage_y(total, y_limit):
    def decorate(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            ax = plt.gca()
            ax2 = ax.twinx()

            # Use a LinearLocator to ensure the correct number of ticks
            ax.set_ylabel("[%]")
            ax.yaxis.set_major_locator(ticker.LinearLocator(11))
            ax.set_ylim(0, y_limit)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

            # Fix the frequency range to 0-100
            ax2.set_ylim(0, (y_limit / 100) * total)
            ax2.yaxis.set_major_formatter(millify)

            ax2.grid(None)

            return f(*args, **kwargs)

        @attach_wrapper(wrapper)
        def set_total(new_total):
            nonlocal total
            total = new_total

        @attach_wrapper(wrapper)
        def set_ylimit(new_ylimit):
            nonlocal y_limit
            y_limit = new_ylimit

        return wrapper

    return decorate


millnames = ["", " K", " M", " Mi", " Tr"]


@ticker.FuncFormatter
def millify(x, pos):
    x = float(x)
    millidx = max(
        0,
        min(
            len(millnames) - 1, int(math.floor(0 if x == 0 else math.log10(abs(x)) / 3))
        ),
    )

    if millidx > 1:
        return "{:.1f}{}".format(x / 10 ** (3 * millidx), millnames[millidx])
    else:
        return "{:.0f}{}".format(x / 10 ** (3 * millidx), millnames[millidx])


def save_plots(plot_functions, path, cohort, figsize=(8, 5)):
    with PdfPages(path) as pdf:
        for p in plot_functions:
            saver(pdf, figsize=figsize)(p)(cohort=cohort)
