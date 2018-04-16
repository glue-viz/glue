from .data_viewer import ImageViewer  # noqa


def setup():
    from glue.config import qt_client
    qt_client.add(ImageViewer)
