from PyQt4.QtGui import QApplication

from ....core.data_collection import DataCollection
from ..data_viewer import DataViewer
from ..histogram_widget import HistogramWidget
from ..scatter_widget import ScatterWidget
from ..image_widget import ImageWidget

import pytest
from mock import MagicMock, patch

ALL_WIDGETS = [HistogramWidget, ScatterWidget, ImageWidget]

def setup_module(module):
    module.app = QApplication([''])

def teardown_module(module):
    del module.app

@pytest.mark.parametrize(('widget'), ALL_WIDGETS)
def test_unregister_on_close(widget):
    unreg = MagicMock()
    hub = MagicMock()
    collect = DataCollection()
    w = widget(collect)
    w.unregister = unreg
    w.register_to_hub(hub)
    w.close()
    unreg.assert_called_once_with(hub)
