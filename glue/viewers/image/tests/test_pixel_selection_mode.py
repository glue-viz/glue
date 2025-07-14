# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

import numpy as np
from matplotlib.backend_bases import MouseEvent, MouseButton


from glue.core.data import Data
from glue.viewers.image.viewer import SimpleImageViewer
from glue.core.application_base import Application
from glue.viewers.image.pixel_selection_mode import PixelSelectionTool
from glue.viewers.image.pixel_selection_subset_state import PixelSubsetState


def test_pixel_selection_mode():

    data1 = Data(x=np.arange(6).reshape((2,3)), label='x')
    data2 = Data(y=2 * np.arange(6).reshape((2, 3)), label='y')

    app = Application()
    app.data_collection.append(data1)
    app.data_collection.append(data2)

    viewer = app.new_data_viewer(SimpleImageViewer)
    viewer.add_data(data1)
    viewer.add_data(data2)
    viewer.central_widget = viewer.figure

    tool = PixelSelectionTool(viewer)
    tool.activate()

    event = MouseEvent(name='button_press_event', canvas=viewer.figure.canvas,
                       x=200,y=300,button=MouseButton.LEFT)
    viewer.figure.canvas.callbacks.process('button_press_event', event)

    assert len(data1.subsets) == 1
    assert len(data2.subsets) == 1
    assert isinstance(data1.subsets[0].subset_state,
                      PixelSubsetState)
    assert isinstance(data2.subsets[0].subset_state,
                      PixelSubsetState)
