def setup():
    from glue.config import qt_client
    from .qt.data_viewer import ImageViewer
    qt_client.add(ImageViewer)
