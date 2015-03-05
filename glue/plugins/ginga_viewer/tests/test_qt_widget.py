from __future__ import print_function, division

from ....qt.widgets.tests.test_image_widget import _TestImageWidgetBase
from ....tests.helpers import requires_ginga

from ..qt_widget import GingaWidget


@requires_ginga
class TestGingaWidget(_TestImageWidgetBase):
    widget_cls = GingaWidget
