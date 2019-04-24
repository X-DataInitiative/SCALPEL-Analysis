import math
from contextlib import contextmanager
from functools import partial, wraps

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.backends.backend_pdf import PdfPages

from src.exploration.core.io import get_logger
from src.exploration.stats.graph_utils import format_title

CONTEXT_SEABORN = 'seaborn'
CONTEXT_PYPLOT = 'pyplot'
_CONTEXT_RUN_DECORATOR_AFTER = frozenset([CONTEXT_SEABORN])
_CONTEXT_RUN_DECORATOR_BEFORE = frozenset([CONTEXT_PYPLOT])


# simple implementation of AOP https://en.wikipedia.org/wiki/Aspect-oriented_programming
@contextmanager
def _before_or_after_by_context(func, context, *args, **kwargs):
    if context in _CONTEXT_RUN_DECORATOR_BEFORE:
        func(*args, **kwargs)
    yield
    if context in _CONTEXT_RUN_DECORATOR_AFTER:
        func(*args, **kwargs)


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


def xlabel(label, context=CONTEXT_PYPLOT):
    def set_xlabel(value):
        ax = plt.gca()
        ax.set_xlabel(value)

    def decorate(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            with _before_or_after_by_context(set_xlabel, context, label):
                res = f(*args, **kwargs)
            return res

        return wrapper

    return decorate


def ylabel(label, context=CONTEXT_PYPLOT):
    def set_ylabel(value):
        ax = plt.gca()
        ax.set_ylabel(value)

    def decorate(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            with _before_or_after_by_context(set_ylabel, context, label):
                res = f(*args, **kwargs)
            return res

        return wrapper

    return decorate


def ylabel_fontsize(size, context=CONTEXT_PYPLOT):
    def set_ylabel_fontsize(value):
        ax = plt.gca()
        [item.set_fontsize(value) for item in ax.get_yticklabels()]

    def decorate(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            with _before_or_after_by_context(set_ylabel_fontsize, context, size):
                res = f(*args, **kwargs)
            return res

        return wrapper

    return decorate


def show_labels(frequency, context=CONTEXT_PYPLOT):
    def set_xaxis_labels_visibility(value):
        ax = plt.gca()
        for i, label in enumerate(ax.xaxis.get_ticklabels()):
            if i % value == 0:
                label.set_visible(True)
            else:
                label.set_visible(False)

    def decorate(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            with _before_or_after_by_context(set_xaxis_labels_visibility, context, frequency):
                res = f(*args, **kwargs)
            return res

        return wrapper

    return decorate


def title(header, context=CONTEXT_PYPLOT):
    def set_title(value, cohort):
        ax = plt.gca()
        formatted_title = format_title(
            "{} among {}".format(value, cohort.characteristics)
        )
        ax.set_title(formatted_title)

    def decorate(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                cohort = args[1]
            except IndexError:
                cohort = kwargs["cohort"]
            with _before_or_after_by_context(set_title, context, header, cohort):
                res = f(*args, **kwargs)
            return res

        return wrapper

    return decorate


def percentage_y(total, y_limit, context=CONTEXT_PYPLOT):
    def set_percentage_y(tl, limit):
        ax = plt.gca()
        ax2 = ax.twinx()

        # Use a LinearLocator to ensure the correct number of ticks
        ax.set_ylabel("[%]")
        ax.yaxis.set_major_locator(ticker.LinearLocator(11))
        ax.set_ylim(0, limit)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

        # Fix the frequency range to 0-100
        ax2.set_ylim(0, (limit / 100) * tl)
        ax2.yaxis.set_major_formatter(millify)

        ax2.grid(None)

    def decorate(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            with _before_or_after_by_context(set_percentage_y, context, total, y_limit):
                res = f(*args, **kwargs)
            return res

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


def save_plots(plot_functions, path, cohort, figsize=(8, 5), **kwargs):
    with PdfPages(path) as pdf:
        for p in plot_functions:
            saver(pdf, figsize=figsize)(p)(cohort=cohort, **kwargs)
