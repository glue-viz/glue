def setup():
    from glue.config import qt_client
    from .qt import TableViewer
    qt_client.add(TableViewer)
