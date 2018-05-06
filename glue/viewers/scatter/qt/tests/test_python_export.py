from __future__ import absolute_import, division, print_function

import numpy as np
import matplotlib.pyplot as plt
from astropy.utils import NumpyRNGContext

from glue.core import Data, DataCollection
from glue.app.qt.application import GlueApplication
from glue.viewers.matplotlib.qt.tests.test_python_export import BaseTestExportPython, random_with_nan
from glue.viewers.scatter.qt import ScatterViewer


class TestExportPython(BaseTestExportPython):

    def setup_method(self, method):

        with NumpyRNGContext(12345):
            self.data = Data(**dict((name, random_with_nan(100, nan_index=idx + 1)) for idx, name in enumerate('abcdefgh')))
        self.data['angle'] = np.random.uniform(0, 360, 100)
        self.data_collection = DataCollection([self.data])
        self.app = GlueApplication(self.data_collection)
        self.viewer = self.app.new_data_viewer(ScatterViewer)
        self.viewer.add_data(self.data)
        self.viewer.state.x_att = self.data.id['a']
        self.viewer.state.y_att = self.data.id['b']

    def teardown_method(self, method):
        self.viewer.close()
        self.viewer = None
        self.app.close()
        self.app = None

    def test_simple(self, tmpdir):
        self.assert_same(tmpdir)

    def test_simple_nofill(self, tmpdir):
        self.viewer.state.layers[0].fill = False
        self.viewer.state.layers[0].size_scaling = 10
        self.assert_same(tmpdir)

    def test_simple_visual(self, tmpdir):
        self.viewer.state.layers[0].color = 'blue'
        self.viewer.state.layers[0].markersize = 30
        self.viewer.state.layers[0].alpha = 0.5
        self.assert_same(tmpdir)

    def test_cmap_mode(self, tmpdir):
        self.viewer.state.layers[0].cmap_mode = 'Linear'
        self.viewer.state.layers[0].cmap_att = self.data.id['c']
        self.viewer.state.layers[0].cmap = plt.cm.BuGn
        self.viewer.state.layers[0].cmap_vmin = 0.2
        self.viewer.state.layers[0].cmap_vmax = 0.7
        self.viewer.state.layers[0].alpha = 0.8
        self.assert_same(tmpdir)

    def test_cmap_mode_nofill(self, tmpdir):
        self.viewer.state.layers[0].fill = False
        self.test_cmap_mode(tmpdir)

    def test_size_mode(self, tmpdir):
        self.viewer.state.layers[0].size_mode = 'Linear'
        self.viewer.state.layers[0].size_att = self.data.id['d']
        self.viewer.state.layers[0].size_vmin = 0.1
        self.viewer.state.layers[0].size_vmax = 0.8
        self.viewer.state.layers[0].size_scaling = 0.4
        self.viewer.state.layers[0].alpha = 0.7
        self.assert_same(tmpdir)

    def test_size_mode_nofill(self, tmpdir):
        self.viewer.state.layers[0].fill = False
        self.test_size_mode(tmpdir)

    def test_line(self, tmpdir):
        self.viewer.state.layers[0].line_visible = True
        self.viewer.state.layers[0].linewidth = 10
        self.viewer.state.layers[0].linestype = 'dashed'
        self.viewer.state.layers[0].color = 'orange'
        self.viewer.state.layers[0].alpha = 0.7
        self.viewer.state.layers[0].markersize = 100
        self.assert_same(tmpdir, tol=5)

    def test_line_cmap(self, tmpdir):
        self.viewer.state.layers[0].cmap_mode = 'Linear'
        self.viewer.state.layers[0].cmap_vmin = 0.2
        self.viewer.state.layers[0].cmap_vmax = 0.7
        self.viewer.state.layers[0].cmap = plt.cm.BuGn
        self.test_line(tmpdir)

    def test_errorbarx(self, tmpdir):
        self.viewer.state.layers[0].xerr_visible = True
        self.viewer.state.layers[0].xerr_att = self.data.id['e']
        self.viewer.state.layers[0].color = 'purple'
        self.viewer.state.layers[0].alpha = 0.5
        self.assert_same(tmpdir)

    def test_errorbary(self, tmpdir):
        self.viewer.state.layers[0].yerr_visible = True
        self.viewer.state.layers[0].yerr_att = self.data.id['f']
        self.viewer.state.layers[0].color = 'purple'
        self.viewer.state.layers[0].alpha = 0.5
        self.assert_same(tmpdir)

    def test_errorbarxy(self, tmpdir):
        self.viewer.state.layers[0].xerr_visible = True
        self.viewer.state.layers[0].xerr_att = self.data.id['e']
        self.viewer.state.layers[0].yerr_visible = True
        self.viewer.state.layers[0].yerr_att = self.data.id['f']
        self.viewer.state.layers[0].color = 'purple'
        self.viewer.state.layers[0].alpha = 0.5
        self.assert_same(tmpdir)

    def test_errorbarxy_cmap(self, tmpdir):
        self.viewer.state.layers[0].cmap_mode = 'Linear'
        self.viewer.state.layers[0].cmap_vmin = 0.2
        self.viewer.state.layers[0].cmap_vmax = 0.7
        self.viewer.state.layers[0].cmap = plt.cm.BuGn
        self.test_errorbarxy(tmpdir)

    def _vector_common(self, tmpdir):
        self.viewer.state.layers[0].vector_visible = True
        self.viewer.state.layers[0].vy_att = self.data.id['g']
        self.viewer.state.layers[0].vector_arrowhead = True
        self.viewer.state.layers[0].vector_origin = 'tail'
        self.viewer.state.layers[0].vector_scaling = 1.5
        self.viewer.state.layers[0].color = 'teal'
        self.viewer.state.layers[0].alpha = 0.9
        self.assert_same(tmpdir)

    def test_vector_cartesian(self, tmpdir):
        self.viewer.state.layers[0].vector_mode = 'Cartesian'
        self.viewer.state.layers[0].vx_att = self.data.id['h']
        self._vector_common(tmpdir)

    def test_vector_polar(self, tmpdir):
        self.viewer.state.layers[0].vector_mode = 'Polar'
        self.viewer.state.layers[0].vx_att = self.data.id['angle']
        self._vector_common(tmpdir)

    def test_vector_cartesian_cmap(self, tmpdir):
        self.viewer.state.layers[0].cmap_mode = 'Linear'
        self.viewer.state.layers[0].cmap_vmin = 0.2
        self.viewer.state.layers[0].cmap_vmax = 0.7
        self.viewer.state.layers[0].cmap = plt.cm.BuGn
        self.test_vector_cartesian(tmpdir)

    def test_vector_cartesian_xflip(self, tmpdir):
        # Regression test for a bug that caused vectors to not be flipped
        self.viewer.state.layers[0].vector_mode = 'Cartesian'
        self.viewer.state.layers[0].vx_att = self.data.id['h']
        self.viewer.state.flip_x()
        self._vector_common(tmpdir)

    def test_subset(self, tmpdir):
        self.data_collection.new_subset_group('mysubset', self.data.id['a'] > 0.5)
        self.assert_same(tmpdir)

    def test_density_map_with_subset(self, tmpdir):
        self.viewer.state.dpi = 2
        self.viewer.state.layers[0].density_map = True
        self.data_collection.new_subset_group('mysubset', self.data.id['a'] > 0.5)
        self.assert_same(tmpdir)

    def test_density_map_cmap_with_subset(self, tmpdir):
        self.viewer.state.dpi = 2
        self.viewer.state.layers[0].density_map = True
        self.viewer.state.layers[0].cmap_mode = 'Linear'
        self.viewer.state.layers[0].cmap_vmin = 0.2
        self.viewer.state.layers[0].cmap_vmax = 0.7
        self.viewer.state.layers[0].cmap = plt.cm.BuGn
        self.data_collection.new_subset_group('mysubset', self.data.id['a'] > 0.5)
        self.assert_same(tmpdir)

    def test_cmap_mode_change(self, tmpdir):
        # Regression test for a bug that caused scatter markers to not change
        # color when going from Linear to Fixed mode
        self.viewer.state.layers[0].size_mode = 'Linear'
        self.viewer.state.layers[0].cmap_mode = 'Linear'
        self.viewer.state.layers[0].cmap_mode = 'Fixed'
        self.assert_same(tmpdir)
