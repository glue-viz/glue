def setup():
    from glue.config import qt_client
    from .qt import ScatterViewer
    qt_client.add(ScatterViewer)
