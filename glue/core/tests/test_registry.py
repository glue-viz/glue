# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

from ..registry import Registry


def setup_function(function):
    Registry().clear()


def test_singleton():
    assert Registry() is Registry()


def test_unique():
    r = Registry()
    assert r.register(3, "test") == "test"
    assert r.register(4, "test2") == "test2"


def test_disambiguate():
    r = Registry()
    assert r.register(3, "test") == "test"
    assert r.register(4, "test") == "test_01"


def test_rename():
    r = Registry()
    assert r.register(3, "test") == "test"
    assert r.register(4, "test2") == "test2"
    assert r.register(3, "test") == "test"


def test_rename_then_new():
    r = Registry()
    assert r.register(3, "test") == "test"
    assert r.register(3, "test2") == "test2"
    assert r.register(4, "test") == "test"


def test_cross_class():
    r = Registry()
    assert r.register(3, "test") == "test"
    assert r.register(3.5, "test") == "test"
    assert r.register(4.5, "test") == "test_01"


def test_group_override():
    r = Registry()
    assert r.register(3, "test") == "test"
    assert r.register(3.5, "test", group=int) == "test_01"
    assert r.register(4, "test", group=float) == "test"


def test_unregister():
    r = Registry()
    assert r.register(3, "test") == "test"
    r.unregister(3)
    assert r.register(4, "test") == "test"


def test_relabel_to_self():
    r = Registry()
    assert r.register(3, "test") == "test"
    assert r.register(3, "test") == "test"


def test_lowest_disambiguation():
    r = Registry()
    assert r.register(3, "test") == "test"
    assert r.register(4, "test") == "test_01"
    assert r.register(4, "test") == "test_01"
