from .data_viewer import DendrogramViewer  # noqa


def setup():
    from glue.config import qt_client
    qt_client.add(DendrogramViewer)
