from glue.tests.visual.helpers import visual_test
from glue.viewers.profile.viewer import SimpleProfileViewer
from glue.core.application_base import Application
from glue.core.data import Data


@visual_test
def test_simple_viewer():

    # Make sure the simple viewer can be instantiated

    data1 = Data(x=[1, 2, 3], label='data1')
    data2 = Data(y=[1, 2, 3], label='data2')

    app = Application()
    app.data_collection.append(data1)
    app.data_collection.append(data2)

    viewer = app.new_data_viewer(SimpleProfileViewer)
    viewer.add_data(data1)
    viewer.add_data(data2)

    app.data_collection.new_subset_group(label='subset1', subset_state=data1.pixel_component_ids[0] > 0.8)

    viewer.state.layers[2].linewidth = 5

    return viewer.figure
