from functools import wraps

__all__ = ['memoize', 'singleton', 'memoize_attr_check']


def memoize(func):
    """Save results of function calls to avoid repeated calculation"""
    memo = {}

    @wraps(func)
    def wrapper(*args):
        try:
            return memo[args]
        except KeyError:
            result = func(*args)
            memo[args] = result
            return result
        except TypeError:  # unhashable input
            return func(*args)

    return wrapper

def memoize_attr_check(attr):
    """ Memoize a method call, cached both on arguments and given attribute
    of first argument (which is presumably self)

    Has the effect of re-calculating results if a specific attribute changes
    """

    def decorator(func):
        #must return a decorator function

        @wraps(func)
        def result(*args):
            first_arg = getattr(args[0], attr)
            return memo(first_arg, *args)

        @memoize
        def memo(*args):
            return func(*args[1:])

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
