import numpy as np
from mock import MagicMock

from ..pv_slicer import _slice_from_path, _slice_label, _slice_index, PVSliceWidget

def mock_image_widget():
    result = MagicMock()

    return result


class TestPV(object):

    def test_home_signal(self):
        # regression test

        self.called = 0

        def increment(*args):
            self.called += 1

        im = np.zeros((3, 4))
        x = np.arange(4)
        y = np.arange(3)
        parent = mock_image_widget()
        w = PVSliceWidget(im, x, y, parent)
        w.central_widget.canvas.homeButton.connect(increment)

        tb = w.toolbar
        tb.home()

        assert self.called == 1
