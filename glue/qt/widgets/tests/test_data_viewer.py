# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

from ....core import Data, DataCollection
from ..data_viewer import DataViewer
from ....viewers.histogram.qt_widget import HistogramWidget
from ....viewers.scatter.qt_widget import ScatterWidget
from ....viewers.image.qt_widget import ImageWidget
from ....viewers.dendro.qt_widget import DendroWidget

from . import simple_session

import pytest
from mock import MagicMock, patch

all_widgets = pytest.mark.parametrize(('widget'),
                                      [HistogramWidget, ScatterWidget,
                                       ImageWidget, DendroWidget])


def setup_function(func):
    import os
    os.environ['GLUE_TESTING'] = 'True'


@all_widgets
def test_unregister_on_close(widget):
    session = simple_session()
    hub = session.hub

    w = widget(session)
    w.register_to_hub(hub)
    with patch.object(DataViewer, 'unregister') as unregister:
        w.close()
    unregister.assert_called_once_with(hub)


@all_widgets
def test_single_draw_call_on_create(widget):
    from ....app.glue_application import GlueApplication
    d = Data(x=[[1, 2], [3, 4]])
    dc = DataCollection([d])
    app = GlueApplication(dc)

    try:
        from glue.qt.widgets.mpl_widget import MplCanvas
        draw = MplCanvas.draw
        MplCanvas.draw = MagicMock()

        app.new_data_viewer(widget, data=d)

        # each Canvas instance gives at most 1 draw call
        selfs = [c[0][0] for c in MplCanvas.draw.call_arg_list]
        assert len(set(selfs)) == len(selfs)
    finally:
        MplCanvas.draw = draw


@all_widgets
def test_close_on_last_layer_remove(widget):

    from ....app.glue_application import GlueApplication

    # regression test for 391

    d = Data(x=[[1, 2], [3, 4]])
    d2 = Data(z=[1, 2, 3])
    dc = DataCollection([d, d2])
    app = GlueApplication(dc)
    with patch.object(widget, 'close') as close:
        w = app.new_data_viewer(widget, data=d)
        w.add_data(d2)
        dc.remove(d)
        dc.remove(d2)
    assert close.call_count >= 1
