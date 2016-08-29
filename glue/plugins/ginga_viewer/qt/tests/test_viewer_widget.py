from __future__ import absolute_import, division, print_function

import pytest

pytest.importorskip('ginga')

from glue.viewers.image.qt.tests.test_viewer_widget import _TestImageWidgetBase

from ..viewer_widget import GingaWidget


class TestGingaWidget(_TestImageWidgetBase):
    widget_cls = GingaWidget
