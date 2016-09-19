def setup():
    from glue.config import qt_client
    from .data_viewer import ImageViewer
    qt_client.add(ImageViewer)
