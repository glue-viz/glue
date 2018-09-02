from .data_viewer import HistogramViewer  # noqa


def setup():
    from glue.config import qt_client
    qt_client.add(HistogramViewer)
