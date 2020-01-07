import os
from shutil import rmtree
from tempfile import mkdtemp

import numpy as np

from glue.core import Data, DataCollection
from glue.tests.helpers import requires_astropy
from glue.tests.helpers import requires_qt

from ..export_d3po import make_data_file, save_d3po


@requires_astropy
def test_make_data_file():

    from astropy.table import Table

    # astropy.Table interface has changed across versions. Check
    # that we build a valid table
    d = Data(x=[1, 2, 3], y=[2, 3, 4], label='data')
    s = d.new_subset(label='test')
    s.subset_state = d.id['x'] > 1

    dir = mkdtemp()
    try:
        make_data_file(d, (s,), dir)
        t = Table.read(os.path.join(dir, 'data.csv'), format='ascii')
        np.testing.assert_array_equal(t['x'], [1, 2, 3])
        np.testing.assert_array_equal(t['y'], [2, 3, 4])
        np.testing.assert_array_equal(t['selection_0'], [0, 1, 1])
    finally:
        rmtree(dir, ignore_errors=True)


@requires_qt
def test_save_d3po(tmpdir):

    from glue.app.qt.application import GlueApplication
    from glue.viewers.scatter.qt import ScatterViewer
    from glue.viewers.histogram.qt import HistogramViewer

    output = tmpdir.join('output.html').strpath

    d = Data(x=[1, 2, 3], y=[2, 3, 4], label='data')
    dc = DataCollection([d])
    app = GlueApplication(dc)
    app.new_data_viewer(ScatterViewer, data=d)
    app.new_data_viewer(HistogramViewer, data=d)
    save_d3po(app, output, launch=False)
    app.close()
