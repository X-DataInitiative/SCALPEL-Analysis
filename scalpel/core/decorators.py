# License: BSD 3 clause
import warnings
from functools import partial, wraps

from scalpel.core.io import get_logger


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


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        msg = "Function {} is deprecated".format(func.__name__)
        warnings.warn(msg, category=DeprecationWarning)
        return func(*args, **kwargs)

    return wrapper
