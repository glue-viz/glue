from __future__ import absolute_import, division, print_function

from functools import wraps

__all__ = ['memoize', 'singleton', 'memoize_attr_check']


def _make_key(args, kwargs):
    return args, frozenset(kwargs.items())


def memoize(func):
    """Save results of function calls to avoid repeated calculation"""
    memo = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        key = _make_key(args, kwargs)
        try:
            return memo[key]
        except KeyError:
            result = func(*args, **kwargs)
            memo[key] = result
            return result
        except TypeError:  # unhashable input
            return func(*args, **kwargs)

    wrapper.__memoize_cache = memo
    return wrapper


def clear_cache(func):
    """
    Clear the cache of a function that has potentially been
    decorated by memoize. Safely ignores non-decorated functions
    """
    try:
        func.__memoize_cache.clear()
    except AttributeError:
        pass


def memoize_attr_check(attr):
    """ Memoize a method call, cached both on arguments and given attribute
    of first argument (which is presumably self)

    Has the effect of re-calculating results if a specific attribute changes
    """

    def decorator(func):
        # must return a decorator function

        @wraps(func)
        def result(*args, **kwargs):
            first_arg = getattr(args[0], attr)
            return memo(first_arg, *args, **kwargs)

        @memoize
        def memo(*args, **kwargs):
            return func(*args[1:], **kwargs)

        return result

    return decorator


def singleton(cls):
    """Turn a class into a singleton, such that new objects
    in this class share the same instance"""
    instances = {}

    @wraps(cls)
    def getinstance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]
    return getinstance
