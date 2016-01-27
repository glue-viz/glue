from __future__ import absolute_import, division, print_function

import pytest

from glue.tests.helpers import GINGA_INSTALLED

if not GINGA_INSTALLED:
    pytest.skip()

from glue.viewers.image.qt.tests.test_viewer_widget import _TestImageWidgetBase

from ..qt_widget import GingaWidget


class TestGingaWidget(_TestImageWidgetBase):
    widget_cls = GingaWidget
