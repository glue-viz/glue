# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from ....core import Hub, Data, DataCollection
from ..histogram_widget import HistogramWidget
from ..scatter_widget import ScatterWidget
from ..image_widget import ImageWidget
from ...glue_application import GlueApplication

from . import simple_session

import pytest
from mock import MagicMock, patch

ALL_WIDGETS = [HistogramWidget, ScatterWidget, ImageWidget]


def setup_function(func):
    import os
    os.environ['GLUE_TESTING'] = 'True'


@pytest.mark.parametrize(('widget'), ALL_WIDGETS)
def test_unregister_on_close(widget):
    unreg = MagicMock()
    session = simple_session()
    hub = session.hub
    collect = session.data_collection

    w = widget(session)
    w.unregister = unreg
    w.register_to_hub(hub)
    w.close()
    unreg.assert_called_once_with(hub)


@pytest.mark.parametrize(('widget'), ALL_WIDGETS)
def test_single_draw_call_on_create(widget):
    d = Data(x=[[1, 2], [3, 4]])
    dc = DataCollection([d])
    app = GlueApplication(dc)

    try:
        from glue.qt.widgets.mpl_widget import MplCanvas
        draw = MplCanvas.draw
        MplCanvas.draw = MagicMock()

        w = app.new_data_viewer(widget, data=d)
        assert MplCanvas.draw.call_count == 1
    finally:
        MplCanvas.draw = draw
