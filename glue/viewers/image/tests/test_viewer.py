import numpy as np

from glue.viewers.image.viewer import SimpleImageViewer
from glue.core.application_base import Application
from glue.core.data import Data


def test_simple_viewer():

    # Make sure the simple viewer can be instantiated

    data1 = Data(x=np.arange(6).reshape((2, 3)), label='data1')
    data2 = Data(y=2 * np.arange(6).reshape((2, 3)), label='data2')

    app = Application()
    app.data_collection.append(data1)
    app.data_collection.append(data2)

    viewer = app.new_data_viewer(SimpleImageViewer)
    viewer.add_data(data1)
    viewer.add_data(data2)

    app.data_collection.new_subset_group(label='subset1', subset_state=data1.id['x'] > 2)
