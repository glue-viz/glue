def setup():
    from glue.config import qt_client
    from .qt import FastScatterViewer
    qt_client.add(FastScatterViewer)
