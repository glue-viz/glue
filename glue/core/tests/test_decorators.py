# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103, R0903

from ..decorators import singleton, memoize, memoize_attr_check


@singleton
class SingletonOne(object):
    """test docstring"""
    pass


@singleton
class SingletonTwo(object):
    pass


class MemoAtt(object):
    def __init__(self):
        self.target = 1
        self.trigger = 0

    @memoize_attr_check('trigger')
    def test(self):
        return self.target

    @memoize_attr_check('trigger')
    def test_kwarg(self, x=0):
        return self.target + x


def test_singleton():
    f = SingletonOne()
    g = SingletonOne()
    h = SingletonTwo()
    k = SingletonTwo()
    assert f is g
    assert h is k
    assert f is not h


def test_memoize():
    class Bar(object):
        pass

    @memoize
    def func(x):
        return x.att

    b = Bar()
    b.att = 5

    assert func(b) == 5
    b.att = 7
    assert func(b) == 5  # should return memoized func


def test_memoize_unhashable():

    @memoize
    def func(x, view=None):
        return 2 * x

    assert func(1, view=slice(1, 2, 3)) == 2
    assert func(1, view=slice(1, 2, 3)) == 2


def test_memoize_attribute():
    f = MemoAtt()
    assert f.test() == 1
    f.target = 2
    assert f.test() == 1
    f.trigger = 1
    assert f.test() == 2


def test_decorators_maintain_docstrings():
    assert SingletonOne.__doc__ == "test docstring"

    @memoize
    def test():
        """test docstring"""

    assert test.__doc__ == "test docstring"

    class MemoClass(object):
        @memoize_attr_check('test')
        def test(self):
            """123"""
            pass

    assert MemoClass.test.__doc__ == "123"


def test_memoize_kwargs():

    @memoize
    def memoadd(x, y=0):
        return x + y

    assert memoadd(3) == 3
    assert memoadd(3, 2) == 5
    assert memoadd(3, y=3) == 6


def test_memoize_attribute_kwargs():

    f = MemoAtt()
    assert f.test_kwarg() == 1
    assert f.test_kwarg(x=5) == 6
    f.target = 2
    assert f.test_kwarg() == 1
    f.trigger = 1
    assert f.test_kwarg() == 2
    assert f.test_kwarg(x=6) == 8
