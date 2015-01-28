from __future__ import print_function, division

from ...image.tests.test_qt_widget import _TestImageWidgetBase
from ....tests.helpers import requires_ginga

try:
    from ..qt_widget import GingaWidget
except ImportError:
    GingaWidget = None


@requires_ginga
class TestGingaWidget(_TestImageWidgetBase):
    widget_cls = GingaWidget
