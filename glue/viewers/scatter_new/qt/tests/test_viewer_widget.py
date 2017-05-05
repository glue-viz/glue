# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

import os

import pytest

from numpy.testing import assert_equal, assert_allclose

from glue.core import Data
from glue.core.roi import XRangeROI
from glue.core.subset import RangeSubsetState, CategoricalROISubsetState
from glue import core
from glue.core.component_id import ComponentID
from glue.core.tests.util import simple_session
from glue.utils.qt import combo_as_string
from glue.viewers.common.qt.tests.test_mpl_data_viewer import BaseTestMatplotlibDataViewer
from glue.core.state import GlueUnSerializer

from ..data_viewer import ScatterViewer

DATA = os.path.join(os.path.dirname(__file__), 'data')


class TestScatterCommon(BaseTestMatplotlibDataViewer):
    def init_data(self):
        return Data(label='d1', x=[3.4, 2.3, -1.1, 0.3], y=['a', 'b', 'c', 'a'])
    viewer_cls = ScatterViewer
