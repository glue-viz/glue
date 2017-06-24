from __future__ import absolute_import, division, print_function

from ..config import qt_client, link_function, data_factory
from glue.tests.helpers import requires_qt


@requires_qt
def test_default_clients():

    from glue.viewers.image.qt import ImageViewer
    from glue.viewers.scatter.qt import ScatterViewer
    from glue.viewers.histogram.qt import HistogramViewer

    assert ImageViewer in qt_client
    assert ScatterViewer in qt_client
    assert HistogramViewer in qt_client


def test_add_client():
    @qt_client
    class TestClient(object):
        pass

    assert TestClient in qt_client


def test_add_link_default():
    @link_function(info='maps x to y', output_labels=['y'])
    def foo(x):
        return 3
    val = (foo, 'maps x to y', ['y'], 'General')
    assert val in link_function


def test_add_data_factory():
    @data_factory('XYZ file', "*txt")
    def foo(x):
        pass
    assert (foo, 'XYZ file', '*txt', 0, False) in data_factory
