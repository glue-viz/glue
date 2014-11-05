from .test_image_widget import _TestImageWidgetBase
from ....tests.helpers import requires_ginga

try:
    from ..ginga_widget import GingaWidget
except ImportError:
    GingaWidget = None


@requires_ginga
class TestGingaWidget(_TestImageWidgetBase):
    widget_cls = GingaWidget
