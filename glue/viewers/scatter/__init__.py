def setup():
    from glue.config import qt_client
    from .qt.data_viewer import ScatterViewer
    qt_client.add(ScatterViewer)
