import logging
import math
from functools import partial, wraps

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.backends.backend_pdf import PdfPages


def register(f, registry):
    registry.add(f)
    return f


def attach_wrapper(obj, func=None):
    if func is None:
        return partial(attach_wrapper, obj)
    setattr(obj, func.__name__, func)
    return func


def logged(level, name=None, message=None):
    '''
    Add logging to a function.  level is the logging
    level, name is the logger name, and message is the
    log message.  If name and message aren't specified,
    they default to the function's module and name.
    '''

    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logname = name if name else func.__module__
            log = logging.getLogger(logname)
            logmsg = message if message else func.__name__
            log.log(level, logmsg)
            return func(*args, **kwargs)

        return wrapper

    return decorate


def saver(pdf):
    def decorate(f):
        def save_result(*args, **kwargs):
            fig = plt.figure()
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


def title(label):
    def decorate(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            ax = plt.gca()
            ax.set_title(label)
            return f(*args, **kwargs)

        return wrapper

    return decorate


def percentage_y(total, y_limit):
    def decorate(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            ax = plt.gca()
            ax2 = ax.twinx()

            # Switch so count axis is on right, frequency on left
            ax2.yaxis.tick_left()
            ax.yaxis.tick_right()

            # Also switch the labels over
            ax.yaxis.set_label_position('right')
            ax2.yaxis.set_label_position('left')

            ax2.set_ylabel('[%]')

            # Use a LinearLocator to ensure the correct number of ticks
            ax.yaxis.set_major_locator(ticker.LinearLocator(11))

            # Fix the frequency range to 0-100
            ax2.set_ylim(0, y_limit)
            ax.set_ylim(0, (y_limit / 100) * total)
            ax.yaxis.set_major_formatter(millify)
            # And use a MultipleLocator to ensure a tick spacing of 10
            ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))

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


millnames = ['', ' K', ' M', ' Mi', ' Tr']


@ticker.FuncFormatter
def millify(x, pos):
    x = float(x)
    millidx = max(0, min(len(millnames) - 1,
                         int(math.floor(0 if x == 0 else math.log10(abs(x)) / 3))))

    if millidx > 1:
        return '{:.1f}{}'.format(x / 10 ** (3 * millidx), millnames[millidx])
    else:
        return '{:.0f}{}'.format(x / 10 ** (3 * millidx), millnames[millidx])


def save_plots(plot_functions, path, cohort):
    with PdfPages(path) as pdf:
        for p in plot_functions:
            saver(pdf)(p)(cohort=cohort)
