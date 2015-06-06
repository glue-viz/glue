import pytest
import numpy as np
from mock import MagicMock

from ....core.tests.util import simple_session
from ....core import Data, Coordinates
from ....core.roi import RectangularROI
from ....qt.widgets import ImageWidget

from ..spectrum_tool import Extractor, ConstraintsWidget, FitSettingsWidget, SpectrumTool
from ....core.fitters import PolynomialFitter

needs_modeling = lambda x: x
try:
    from ....core.fitters import SimpleAstropyGaussianFitter
except ImportError:
    needs_modeling = pytest.mark.skipif(True, reason='Needs astropy >= 0.3')


class MockCoordinates(Coordinates):

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
        for tool in self.image._tools:
            if isinstance(tool, SpectrumTool):
                self.tool = tool
                break
        else:
            raise Exception("SpectrumTool not found")
        self.tool.show = lambda *args: None

    def build_spectrum(self):
        roi = RectangularROI()
        roi.update_limits(0, 2, 0, 2)
        self.tool._update_profile()

    def test_reset_on_view_change(self):
        self.build_spectrum()
        self.tool.widget = MagicMock()
        self.tool.widget.isVisible.return_value = True
        self.tool.reset = MagicMock()
        self.image.client.slice = ('x', 1, 'y')
        assert self.tool.reset.call_count > 0


class Test3DExtractor(object):

    def setup_method(self, method):
        self.data = Data()
        self.data.coords = MockCoordinates()
        self.data.add_component(np.random.random((3, 4, 5)), label='x')
        self.x = self.data['x']

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

        expected = self.x[:, :3, :3].mean(axis=1).mean(axis=1)
        _, actual = Extractor.spectrum(
            self.data, self.data.id['x'], roi, (0, 'x', 'y'), 0)
        np.testing.assert_array_almost_equal(expected, actual)

    def test_spectrum_oob(self):
        roi = RectangularROI()
        roi.update_limits(-1, -1, 3, 3)

        expected = self.x[:, :3, :3].mean(axis=1).mean(axis=1)

        _, actual = Extractor.spectrum(self.data, self.data.id['x'],
                                       roi, (0, 'x', 'y'), 0)
        np.testing.assert_array_almost_equal(expected, actual)

    def test_pixel2world(self):
        # p2w(x) = 2x, 0 <= x <= 2
        assert Extractor.pixel2world(self.data, 0, 1) == 2

        # clips to boundary
        assert Extractor.pixel2world(self.data, 0, -1) == 0
        assert Extractor.pixel2world(self.data, 0, 5) == 4

    def test_world2pixel(self):
        # w2p(x) = x/2, 0 <= x <= 4
        assert Extractor.world2pixel(self.data, 0, 2.01) == 1

        # clips to boundary
        assert Extractor.world2pixel(self.data, 0, -1) == 0
        assert Extractor.world2pixel(self.data, 0, 8) == 2

    def test_extract_subset(self):
        sub = self.data.new_subset()
        sub.subset_state = self.data.id['x'] > .5
        slc = (0, 'y', 'x')
        mask = sub.to_mask()[0]
        mask = mask.reshape(-1, mask.shape[0], mask.shape[1])

        expected = (self.x * mask).sum(axis=1).sum(axis=1)
        expected /= mask.sum(axis=1).sum(axis=1)
        _, actual = Extractor.subset_spectrum(sub, self.data.id['x'],
                                              slc, 0)
        np.testing.assert_array_almost_equal(expected, actual)


class Test4DExtractor(object):

    def setup_method(self, method):
        self.data = Data()
        self.data.coords = MockCoordinates()
        x, y, z, w = np.mgrid[:3, :4, :5, :4]
        self.data.add_component(1. * w, label='x')

    def test_extract(self):

        roi = RectangularROI()
        roi.update_limits(0, 0, 2, 3)

        expected = self.data['x'][:, :2, :3, 1].mean(axis=1).mean(axis=1)
        _, actual = Extractor.spectrum(self.data, self.data.id['x'],
                                       roi, (0, 'x', 'y', 1), 0)

        np.testing.assert_array_equal(expected, actual)


class TestConstraintsWidget(object):

    def setup_method(self, method):
        self.constraints = dict(a=dict(fixed=True, value=1, limits=None))
        self.widget = ConstraintsWidget(self.constraints)

    def test_settings(self):
        assert self.widget.settings('a') == dict(fixed=True, value=1,
                                                 limits=None)

    def test_update_settings(self):
        self.widget._widgets['a'][2].setChecked(False)
        assert self.widget.settings('a')['fixed'] is False

    def test_update_constraints(self):
        self.widget._widgets['a'][2].setChecked(False)
        fitter = MagicMock()
        self.widget.update_constraints(fitter)
        fitter.set_constraint.assert_called_once_with('a',
                                                      fixed=False, value=1,
                                                      limits=None)


class TestFitSettingsWidget(object):

    def test_option(self):
        f = PolynomialFitter()
        f.degree = 1
        w = FitSettingsWidget(f)
        w.widgets['degree'].setValue(5)
        w.update_fitter_from_settings()
        assert f.degree == 5

    @needs_modeling
    def test_set_constraints(self):
        f = SimpleAstropyGaussianFitter()
        w = FitSettingsWidget(f)
        w.constraints._widgets['amplitude'][2].setChecked(True)
        w.update_fitter_from_settings()
        assert f.constraints['amplitude']['fixed']


def test_4d_single_channel():

    x = np.random.random((1, 7, 5, 9))
    d = Data(x=x)
    slc = (0, 0, 'x', 'y')
    zaxis = 1
    expected = x[0, :, :, :].mean(axis=1).mean(axis=1)
    roi = RectangularROI()
    roi.update_limits(0, 0, 10, 10)

    _, actual = Extractor.spectrum(d, d.id['x'], roi, slc, zaxis)

    np.testing.assert_array_almost_equal(expected, actual)
