import os
import pytest

import numpy as np
import matplotlib.pyplot as plt

from glue.core import Data, DataCollection
from glue.app.qt.application import GlueApplication
from glue.viewers.scatter.qt import ScatterViewer
from matplotlib.testing.compare import compare_images


def random_with_nan(nsamples, nan_index):
    x = np.random.random(nsamples)
    x[nan_index] = np.nan
    return x


class TestExportPython:

    def setup_method(self, method):

        self.data = Data(**dict((name, random_with_nan(100, nan_index=idx + 1)) for idx, name in enumerate('abcdefgh')))
        self.data['angle'] = np.random.uniform(0, 360, 100)
        self.data_collection = DataCollection([self.data])
        ga = GlueApplication(self.data_collection)
        self.scatter = ga.new_data_viewer(ScatterViewer)
        self.scatter.add_data(self.data)
        self.scatter.state.x_att = self.data.id['a']
        self.scatter.state.y_att = self.data.id['b']

    def test_simple(self, tmpdir):
        self.assert_same(tmpdir)

    def test_simple_visual(self, tmpdir):
        self.scatter.state.layers[0].color = 'blue'
        self.scatter.state.layers[0].markersize = 30
        self.scatter.state.layers[0].alpha = 0.5
        self.assert_same(tmpdir)

    def test_cmap_mode(self, tmpdir):
        self.scatter.state.layers[0].cmap_mode = 'Linear'
        self.scatter.state.layers[0].cmap_att = self.data.id['c']
        self.scatter.state.layers[0].cmap = plt.cm.BuGn
        self.scatter.state.layers[0].cmap_vmin = 0.2
        self.scatter.state.layers[0].cmap_vmax = 0.7
        self.scatter.state.layers[0].alpha = 0.8
        self.assert_same(tmpdir)

    def test_size_mode(self, tmpdir):
        self.scatter.state.layers[0].size_mode = 'Linear'
        self.scatter.state.layers[0].size_att = self.data.id['d']
        self.scatter.state.layers[0].size_vmin = 0.1
        self.scatter.state.layers[0].size_vmax = 0.8
        self.scatter.state.layers[0].size_scaling = 0.4
        self.scatter.state.layers[0].alpha = 0.7
        self.assert_same(tmpdir)

    def test_line(self, tmpdir):
        self.scatter.state.layers[0].line_visible = True
        self.scatter.state.layers[0].linewidth = 10
        self.scatter.state.layers[0].linestype = 'dashed'
        self.scatter.state.layers[0].color = 'orange'
        self.scatter.state.layers[0].alpha = 0.7
        self.scatter.state.layers[0].markersize = 100
        self.assert_same(tmpdir, tol=5)

    def test_line_cmap(self, tmpdir):
        self.scatter.state.layers[0].cmap_mode = 'Linear'
        self.scatter.state.layers[0].cmap_vmin = 0.2
        self.scatter.state.layers[0].cmap_vmax = 0.7
        self.scatter.state.layers[0].cmap = plt.cm.BuGn
        self.test_line(tmpdir)

    def test_errorbarx(self, tmpdir):
        self.scatter.state.layers[0].xerr_visible = True
        self.scatter.state.layers[0].xerr_att = self.data.id['e']
        self.scatter.state.layers[0].color = 'purple'
        self.scatter.state.layers[0].alpha = 0.5
        self.assert_same(tmpdir)

    def test_errorbary(self, tmpdir):
        self.scatter.state.layers[0].yerr_visible = True
        self.scatter.state.layers[0].yerr_att = self.data.id['f']
        self.scatter.state.layers[0].color = 'purple'
        self.scatter.state.layers[0].alpha = 0.5
        self.assert_same(tmpdir)

    def test_errorbarxy(self, tmpdir):
        self.scatter.state.layers[0].xerr_visible = True
        self.scatter.state.layers[0].xerr_att = self.data.id['e']
        self.scatter.state.layers[0].yerr_visible = True
        self.scatter.state.layers[0].yerr_att = self.data.id['f']
        self.scatter.state.layers[0].color = 'purple'
        self.scatter.state.layers[0].alpha = 0.5
        self.assert_same(tmpdir)

    def test_errorbarxy_cmap(self, tmpdir):
        self.scatter.state.layers[0].cmap_mode = 'Linear'
        self.scatter.state.layers[0].cmap_vmin = 0.2
        self.scatter.state.layers[0].cmap_vmax = 0.7
        self.scatter.state.layers[0].cmap = plt.cm.BuGn
        self.test_errorbarxy(tmpdir)

    def _vector_common(self, tmpdir):
        self.scatter.state.layers[0].vector_visible = True
        self.scatter.state.layers[0].vy_att = self.data.id['g']
        self.scatter.state.layers[0].vector_arrowhead = True
        self.scatter.state.layers[0].vector_origin = 'tail'
        self.scatter.state.layers[0].vector_scaling = 1.5
        self.scatter.state.layers[0].color = 'teal'
        self.scatter.state.layers[0].alpha = 0.9
        self.assert_same(tmpdir)

    def test_vector_cartesian(self, tmpdir):
        self.scatter.state.layers[0].vector_mode = 'Cartesian'
        self.scatter.state.layers[0].vx_att = self.data.id['h']
        self._vector_common(tmpdir)

    def test_vector_polar(self, tmpdir):
        self.scatter.state.layers[0].vector_mode = 'Polar'
        self.scatter.state.layers[0].vx_att = self.data.id['angle']
        self._vector_common(tmpdir)

    def test_vector_cartesian_cmap(self, tmpdir):
        self.scatter.state.layers[0].cmap_mode = 'Linear'
        self.scatter.state.layers[0].cmap_vmin = 0.2
        self.scatter.state.layers[0].cmap_vmax = 0.7
        self.scatter.state.layers[0].cmap = plt.cm.BuGn
        self.test_vector_cartesian(tmpdir)

    def test_subset(self, tmpdir):
        self.data_collection.new_subset_group('mysubset', self.data.id['a'] > 0.5)
        self.assert_same(tmpdir)

    def assert_same(self, tmpdir, tol=0.1):

        os.chdir(tmpdir.strpath)

        expected = tmpdir.join('expected.png').strpath
        script = tmpdir.join('actual.py').strpath
        actual = tmpdir.join('myplot.png').strpath

        self.scatter.axes.figure.savefig(expected)

        self.scatter.export_as_script(script)
        with open(script) as f:
            exec(f.read())

        msg = compare_images(expected, actual, tol=tol)
        if msg:
            pytest.fail(msg, pytrace=False)
