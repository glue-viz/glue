from __future__ import absolute_import, division, print_function

from ..config import qt_client, link_function, data_factory


def test_default_clients():
    from glue.qt.widgets.image_widget import ImageWidget
    from glue.qt.widgets.scatter_widget import ScatterWidget
    from glue.qt.widgets.histogram_widget import HistogramWidget
    import glue.config

    assert ImageWidget in qt_client
    assert ScatterWidget in qt_client
    assert HistogramWidget in qt_client


def test_add_client():
    @qt_client
    class TestClient(object):
        pass

    assert TestClient in qt_client


def test_link_defaults():
    from ..core.link_helpers import __LINK_FUNCTIONS__
    assert len(__LINK_FUNCTIONS__) > 0

    for l in __LINK_FUNCTIONS__:
        assert l in [ll[0] for ll in link_function]


def test_add_link_default():
    @link_function(info='maps x to y', output_labels=['y'])
    def foo(x):
        return 3
    val = (foo, 'maps x to y', ['y'])
    assert val in link_function


def test_data_facotry_defaults():
    from ..core.data_factories import __factories__
    assert len(__factories__) > 0

    for f in __factories__:
        assert f in (ff[0] for ff in data_factory.members)


def test_add_data_factory():
    @data_factory('XYZ file', "*txt")
    def foo(x):
        pass
    assert (foo, 'XYZ file', '*txt', 0) in data_factory
