from .test_image_widget import _TestImageWidgetBase
from ..ginga_widget import GingaWidget


class TestGingaWidget(_TestImageWidgetBase):
    widget_cls = GingaWidget
