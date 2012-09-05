#pylint: disable=W0613,W0201,W0212,E1101,E1103
import pytest

from ..decorators import singleton, memoize, memoize_attr_check


@singleton
class Foo(object):
    """test docstring"""
    pass


@singleton
class Bar(object):
    pass


def test_singleton():
    f = Foo()
    g = Foo()
    h = Bar()
    k = Bar()
    assert f is g
    assert h is k
    assert f is not h


def test_memoize():
    class Bar(object):
        pass

    @memoize
    def foo(x):
        return x.bar

    b = Bar()
    b.bar = 5

    assert foo(b) == 5
    b.bar = 7
    assert foo(b) == 5  # should return memoized func


def test_memoize_attribute():

    class Foo(object):
        def __init__(self):
            self.target = 1
            self.trigger = 0

        @memoize_attr_check('trigger')
        def test(self):
            return self.target

    f = Foo()
    assert f.test() == 1
    f.target = 2
    assert f.test() == 1
    f.trigger = 1
    assert f.test() == 2


def test_decorators_maintain_docstrings():
    assert Foo.__doc__ == "test docstring"

    @memoize
    def test():
        """test docstring"""

    assert test.__doc__ == "test docstring"

    class Bar(object):
        @memoize_attr_check('test')
        def test(self):
            """123"""
            pass

    assert Bar.test.__doc__ == "123"


def test_memoize_kwargs():

    @memoize
    def test(x, y=0):
        return x + y

    assert test(3) == 3
    assert test(3, 2) == 5
    assert test(3, y=3) == 6


def test_memoize_attribute_kwargs():

    class Foo(object):
        def __init__(self):
            self.target = 1
            self.trigger = 0

        @memoize_attr_check('trigger')
        def test(self, x=0):
            return self.target + x

    f = Foo()
    assert f.test() == 1
    assert f.test(x=5) == 6
    f.target = 2
    assert f.test() == 1
    f.trigger = 1
    assert f.test() == 2
    assert f.test(x=6) == 8
