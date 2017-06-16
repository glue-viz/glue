# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

import os

import pytest

import numpy as np
from numpy.testing import assert_allclose

from glue.core import Data
from glue.core.roi import RectangularROI
from glue.core.subset import RoiSubsetState, AndState
from glue import core
from glue.core.component_id import ComponentID
from glue.core.tests.util import simple_session
from glue.utils.qt import combo_as_string
from glue.viewers.matplotlib.qt.tests.test_data_viewer import BaseTestMatplotlibDataViewer
from glue.core.state import GlueUnSerializer

from ..data_viewer import ImageViewer

DATA = os.path.join(os.path.dirname(__file__), 'data')


class TestImageCommon(BaseTestMatplotlibDataViewer):
    def init_data(self):
        return Data(label='d1', x=np.arange(12).reshape((3, 4)), y=np.ones((3, 4)))
    viewer_cls = ImageViewer
