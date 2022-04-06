from astropy.utils import NumpyRNGContext

from glue.core import Data, DataCollection
from glue.app.qt.application import GlueApplication
from glue.viewers.matplotlib.qt.tests.test_python_export import BaseTestExportPython, random_with_nan
from glue.viewers.profile.qt import ProfileViewer
from glue.viewers.profile.tests.test_state import SimpleCoordinates


class TestExportPython(BaseTestExportPython):

    def setup_method(self, method):

        self.data = Data(label='d1')
        self.data.coords = SimpleCoordinates()
        with NumpyRNGContext(12345):
            self.data['x'] = random_with_nan(48, 5).reshape((6, 4, 2))
            self.data['y'] = random_with_nan(48, 12).reshape((6, 4, 2))
        self.data_collection = DataCollection([self.data])
        self.app = GlueApplication(self.data_collection)
        self.viewer = self.app.new_data_viewer(ProfileViewer)
        self.viewer.add_data(self.data)
        # Make legend location deterministic
        self.viewer.state.legend.location = 'lower left'

    def teardown_method(self, method):
        self.viewer.close()
        self.viewer = None
        self.app.close()
        self.app = None

    def test_simple(self, tmpdir):
        self.assert_same(tmpdir)

    def test_simple_legend(self, tmpdir):
        self.viewer.state.legend.visible = True
        self.assert_same(tmpdir)

    def test_color(self, tmpdir):
        self.viewer.state.layers[0].color = '#ac0567'
        self.assert_same(tmpdir)

    def test_linewidth(self, tmpdir):
        self.viewer.state.layers[0].linewidth = 7.25
        self.assert_same(tmpdir)

    def test_max(self, tmpdir):
        self.viewer.state.function = 'maximum'
        self.assert_same(tmpdir)

    def test_min(self, tmpdir):
        self.viewer.state.function = 'minimum'
        self.assert_same(tmpdir)

    def test_mean(self, tmpdir):
        self.viewer.state.function = 'mean'
        self.assert_same(tmpdir)

    def test_median(self, tmpdir):
        self.viewer.state.function = 'median'
        self.assert_same(tmpdir)

    def test_sum(self, tmpdir):
        self.viewer.state.function = 'sum'
        self.assert_same(tmpdir)

    def test_normalization(self, tmpdir):
        self.viewer.state.normalize = True
        self.assert_same(tmpdir)

    def test_subset(self, tmpdir):
        self.viewer.state.function = 'mean'
        self.data_collection.new_subset_group('mysubset', self.data.id['x'] > 0.25)
        self.assert_same(tmpdir)

    def test_subset_legend(self, tmpdir):
        self.viewer.state.legend.visible = True
        self.viewer.state.function = 'mean'
        self.viewer.state.layers[0].linewidth = 7.25
        self.data_collection.new_subset_group('mysubset', self.data.id['x'] > 0.25)
        self.assert_same(tmpdir)

    def test_xatt(self, tmpdir):
        self.viewer.x_att = self.data.pixel_component_ids[1]
        self.assert_same(tmpdir)

    def test_profile_att(self, tmpdir):
        self.viewer.layers[0].state.attribute = self.data.id['y']
        self.assert_same(tmpdir)
