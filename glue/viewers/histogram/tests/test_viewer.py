import numpy as np
from numpy.testing import assert_allclose

from astropy.utils import NumpyRNGContext

from glue.tests.visual.helpers import visual_test
from glue.viewers.histogram.viewer import SimpleHistogramViewer
from glue.core.application_base import Application
from glue.core.data import Data
from glue.core.data_derived import IndexedData


@visual_test
def test_simple_histogram_viewer():

    # Make sure the simple viewer can be instantiated

    with NumpyRNGContext(12345):

        data1 = Data(x=np.random.normal(1, 2, 1000), label='data1')
        data2 = Data(y=np.random.uniform(-1, 5, 1000), label='data2')

    app = Application()
    app.data_collection.append(data1)
    app.data_collection.append(data2)

    viewer = app.new_data_viewer(SimpleHistogramViewer)
    viewer.add_data(data1)
    viewer.add_data(data2)

    app.data_collection.new_subset_group(label='subset1', subset_state=data1.id['x'] > 2)

    return viewer.figure


def test_remove_data_collection():

    # Regression test for a bug that caused an IncompatibleAttribute
    # error when updating the number of bins in a histogram after
    # removing a dataset from the DataCollection (this was due to
    # a caching issue)

    data1 = Data(x=[1, 2, 3], label='data1')
    data2 = Data(y=[1, 2, 3], label='data2')

    app = Application()
    app.data_collection.append(data1)
    app.data_collection.append(data2)

    viewer = app.new_data_viewer(SimpleHistogramViewer)
    viewer.add_data(data1)
    viewer.add_data(data2)

    viewer.state.hist_n_bin = 30

    app.data_collection.remove(data1)

    viewer.state.hist_n_bin = 20


def test_incompatible_datasets():

    # Regression test for a bug that caused an IncompatibleAttribute
    # error when changing the dataset used in the histogram viewer to one that
    # is not linked to the first dataset.

    data1 = Data(x=[1, 2, 3], label='data1')
    data2 = Data(y=[1, 2, 3], label='data2')

    app = Application()
    app.data_collection.append(data1)
    app.data_collection.append(data2)

    viewer = app.new_data_viewer(SimpleHistogramViewer)
    viewer.add_data(data1)
    viewer.add_data(data2)

    viewer.state.x_att = data1.id['x']

    viewer.state.hist_n_bin = 30

    viewer.state.x_att = data2.id['y']

    viewer.state.hist_n_bin = 20


def test_reset_limits():

    data1 = Data(x=np.arange(1000), label='data')

    app = Application()
    app.data_collection.append(data1)

    viewer = app.new_data_viewer(SimpleHistogramViewer)
    viewer.add_data(data1)

    viewer.state.reset_limits()

    assert_allclose(viewer.state.x_min, 0)
    assert_allclose(viewer.state.x_max, 999)

    viewer.state.x_limits_percentile = 90

    viewer.state.reset_limits()

    assert_allclose(viewer.state.x_min, 49.95)
    assert_allclose(viewer.state.x_max, 949.05)

    assert_allclose(viewer.state.hist_x_min, 49.95)
    assert_allclose(viewer.state.hist_x_max, 949.05)

    viewer.state.update_bins_on_reset_limits = False

    viewer.state.x_limits_percentile = 80

    viewer.state.reset_limits()

    assert_allclose(viewer.state.x_min, 99.9)
    assert_allclose(viewer.state.x_max, 899.1)

    assert_allclose(viewer.state.hist_x_min, 49.95)
    assert_allclose(viewer.state.hist_x_max, 949.05)


def test_indexed_data():

    # Make sure that the scatter viewer works properly with IndexedData objects

    data_4d = Data(label='hypercube',
                   x=np.random.random((3, 5, 4, 3)),
                   y=np.random.random((3, 5, 4, 3)))

    data_2d = IndexedData(data_4d, (2, None, 3, None))

    application = Application()

    session = application.session

    hub = session.hub

    data_collection = session.data_collection
    data_collection.append(data_4d)
    data_collection.append(data_2d)

    viewer = application.new_data_viewer(SimpleHistogramViewer)
    viewer.add_data(data_2d)

    assert viewer.state.x_att is data_2d.main_components[0]
