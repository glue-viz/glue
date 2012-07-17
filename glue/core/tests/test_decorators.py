from ..decorators import singleton

@singleton
class Foo(object):
    pass


class test_singleton():
    f = Foo()
    g = Foo()
    assert f is g