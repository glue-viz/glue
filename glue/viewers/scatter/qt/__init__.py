from .data_viewer import ScatterViewer  # noqa


def setup():
    from glue.config import qt_client
    qt_client.add(ScatterViewer)
