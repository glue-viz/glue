from functools import wraps

__all__ = ['memoize']


def memoize(func):
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
