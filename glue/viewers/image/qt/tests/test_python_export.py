import os
import pytest

import numpy as np
import matplotlib.pyplot as plt

from glue.core import Data, DataCollection
from glue.app.qt.application import GlueApplication
from glue.viewers.image.qt import ImageViewer
from matplotlib.testing.compare import compare_images


class TestExportPython:

    def setup_method(self, method):

        self.data = Data(cube=np.random.random((30, 50, 20)))
        self.data_collection = DataCollection([self.data])
        ga = GlueApplication(self.data_collection)
        self.image = ga.new_data_viewer(ImageViewer)
        self.image.add_data(self.data)

    def test_simple(self, tmpdir):
        self.assert_same(tmpdir)

    def test_simple_visual(self, tmpdir):
        self.image.state.layers[0].cmap = plt.cm.RdBu
        self.image.state.layers[0].v_min = 0.2
        self.image.state.layers[0].v_max = 0.8
        self.image.state.layers[0].stretch = 'sqrt'
        self.image.state.layers[0].stretch = 'sqrt'
        self.image.state.layers[0].contrast = 0.9
        self.image.state.layers[0].bias = 0.6
        self.assert_same(tmpdir)

    def test_slice(self, tmpdir):
        self.image.state.x_att = self.data.pixel_component_ids[1]
        self.image.state.y_att = self.data.pixel_component_ids[0]
        self.image.state.slices = (2, 3, 4)
        self.assert_same(tmpdir)

    def test_aspect(self, tmpdir):
        self.image.state.aspect = 'auto'
        self.assert_same(tmpdir)

    def test_subset(self, tmpdir):
        self.data_collection.new_subset_group('mysubset', self.data.id['cube'] > 0.5)
        self.assert_same(tmpdir)

    def assert_same(self, tmpdir, tol=0.1):

        self.image.axes.figure.savefig('image.png')

        os.chdir(tmpdir.strpath)

        expected = tmpdir.join('expected.png').strpath
        script = tmpdir.join('actual.py').strpath
        actual = tmpdir.join('myplot.png').strpath

        self.image.axes.figure.savefig(expected)

        self.image.export_as_script(script)
        with open(script) as f:
            exec(f.read())

        msg = compare_images(expected, actual, tol=tol)
        if msg:
            pytest.fail(msg, pytrace=False)
