from .data_viewer import TableViewer, TableLayerArtist  # noqa


def setup():
    from glue.config import qt_client
    qt_client.add(TableViewer)
