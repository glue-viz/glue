# Miscellaneous regression tests for the image viewer

import pytest
import numpy as np

from glue.core import Data
from glue.viewers.image.qt import ImageWidget
from glue.core.tests.util import simple_session
from glue.tests.helpers import requires_matplotlib_ge_14


@requires_matplotlib_ge_14
@pytest.mark.mpl_image_compare(tolerance=1, savefig_kwargs={'dpi': 50})
def test_resample_on_zoom():

    # For images where the aspect ratio of pixels is fixed to be square, when
    # the user zooms in, the limits of the axes are actually changed twice by
    # matplotlib - a second time when the aspect ratio is enforced. So we need
    # to make sure that we update the modest_image when this is the case.

    session = simple_session()

    np.random.seed(12345)

    data = Data(x=np.random.random((2048, 2048)), label='image')
    session.data_collection.append(data)

    image = ImageWidget(session=session)
    image.add_data(data)

    image.show()

    image.axes.figure.canvas.key_press_event('o')
    image.axes.figure.canvas.button_press_event(200, 200, 1)
    image.axes.figure.canvas.motion_notify_event(400, 210)
    image.axes.figure.canvas.button_release_event(400, 210, 1)

    return image.axes.figure
