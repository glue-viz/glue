import pytest

from ..decorators import singleton

@singleton
class Foo(object):
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

