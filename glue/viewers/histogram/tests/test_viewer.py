from glue.viewers.histogram.viewer import SimpleHistogramViewer
from glue.core.application_base import Application
from glue.core.data import Data


def test_simple_viewer():

    # Make sure the simple viewer can be instantiated

    data1 = Data(x=[1, 2, 3], label='data1')
    data2 = Data(y=[1, 2, 3], label='data2')

    app = Application()
    app.data_collection.append(data1)
    app.data_collection.append(data2)

    viewer = app.new_data_viewer(SimpleHistogramViewer)
    viewer.add_data(data1)
    viewer.add_data(data2)

    app.data_collection.new_subset_group(label='subset1', subset_state=data1.id['x'] > 2)


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
