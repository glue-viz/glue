def setup():
    from glue.config import qt_client
    from .qt.data_viewer import HistogramViewer
    qt_client.add(HistogramViewer)
