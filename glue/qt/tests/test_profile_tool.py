import numpy as np
from mock import MagicMock

from ...core.tests.util import simple_session
from ...core import Data, Coordinates
from ...core.roi import RectangularROI
from ..widgets import ImageWidget

from ..spectrum_tool import Extractor


class TestCoordinates(Coordinates):

    def pixel2world(self, *args):
        return [a * 2 for a in args]

    def world2pixel(self, *args):
        return [a / 2 for a in args]


class TestSpectrumTool(object):

    def setup_method(self, method):
        d = Data(x=np.zeros((3, 3, 3)))
        session = simple_session()
        session.data_collection.append(d)

        self.image = ImageWidget(session)
        self.image.add_data(d)
        self.image.data = d
        self.image.attribute = d.id['x']
        self.tool = self.image._spectrum_tool
        self.tool.show = lambda *args: None

    def build_spectrum(self):
        roi = RectangularROI()
        roi.update_limits(0, 2, 0, 2)
        self.tool._update_profile()

    def test_reset_on_view_change(self):
        self.build_spectrum()

        self.tool.reset = MagicMock()
        self.image.client.slice = ('x', 1, 'y')
        assert self.tool.reset.call_count > 0


class Test3DExtractor(object):

    def setup_method(self, method):
        self.data = Data()
        self.data.coords = TestCoordinates()
        self.data.add_component(np.ones((3, 4, 5)), label='x')

    def test_abcissa(self):
        expected = [0, 2, 4]
        actual = Extractor.abcissa(self.data, 0)
        np.testing.assert_equal(expected, actual)

        expected = [0, 2, 4, 6]
        actual = Extractor.abcissa(self.data, 1)
        np.testing.assert_equal(expected, actual)

        expected = [0, 2, 4, 6, 8]
        actual = Extractor.abcissa(self.data, 2)
        np.testing.assert_equal(expected, actual)

    def test_spectrum(self):
        roi = RectangularROI()
        roi.update_limits(0, 0, 3, 3)

        expected = [1, 1, 1]
        _, actual = Extractor.spectrum(
            self.data, self.data.id['x'], roi, 1, 2, 0)
        np.testing.assert_array_equal(expected, actual)

    def test_spectrum_oob(self):
        roi = RectangularROI()
        roi.update_limits(-1, -1, 3, 3)
        expected = [1, 1, 1]
        _, actual = Extractor.spectrum(self.data, self.data.id['x'],
                                       roi, 1, 2, 0)
        np.testing.assert_array_equal(expected, actual)

    def test_pixel2world(self):
        # p2w(x) = 2x, 0 <= x <= 2
        assert Extractor.pixel2world(self.data, 0, 1) == 2

        # clips to boundary
        assert Extractor.pixel2world(self.data, 0, -1) == 0
        assert Extractor.pixel2world(self.data, 0, 5) == 4

    def test_world2pixel(self):
        # w2p(x) = x/2, 0 <= x <= 4
        assert Extractor.world2pixel(self.data, 0, 2) == 1

        # clips to boundary
        assert Extractor.world2pixel(self.data, 0, -1) == 0
        assert Extractor.world2pixel(self.data, 0, 8) == 2
