from __future__ import print_function, division

import pytest
from ....tests.helpers import GINGA_INSTALLED

if not GINGA_INSTALLED:
    pytest.skip()

from ....qt.widgets.tests.test_image_widget import _TestImageWidgetBase

from ..qt_widget import GingaWidget


class TestGingaWidget(_TestImageWidgetBase):
    widget_cls = GingaWidget
