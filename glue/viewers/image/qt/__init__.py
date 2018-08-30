from .data_viewer import ImageViewer  # noqa
from .standalone_image_viewer import StandaloneImageViewer  # noqa


def setup():
    from glue.config import qt_client
    qt_client.add(ImageViewer)
