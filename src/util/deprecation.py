import warnings
import functools


def doublewrap(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return func(args[0])
        else:
            return lambda f: func(f, *args, **kwargs)

    return wrapper


@doublewrap
def deprecated(func, replacement="", message="{name} is deprecated."):
    """This is decorator marks functions as deprecated."""
    msg = message.format(name=func.__name__)
    if replacement != "":
        msg += " "
        msg += replacement

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(msg,
                      category=DeprecationWarning,
                      stacklevel=2)
        return func(*args, **kwargs)
    return wrapper
