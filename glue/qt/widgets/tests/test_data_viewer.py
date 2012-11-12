#pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from ....core import Hub
from ....core.data_collection import DataCollection
from ..histogram_widget import HistogramWidget
from ..scatter_widget import ScatterWidget
from ..image_widget import ImageWidget

import pytest
from mock import MagicMock

ALL_WIDGETS = [HistogramWidget, ScatterWidget, ImageWidget]


def setup_function(func):
    import os
    os.environ['GLUE_TESTING'] = 'True'


@pytest.mark.parametrize(('widget'), ALL_WIDGETS)
def test_unregister_on_close(widget):
    unreg = MagicMock()
    hub = Hub()
    collect = DataCollection()
    w = widget(collect)
    w.unregister = unreg
    w.register_to_hub(hub)
    w.close()
    unreg.assert_called_once_with(hub)
