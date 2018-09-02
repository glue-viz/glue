from .data_viewer import ProfileViewer  # noqa


def setup():
    from glue.config import qt_client
    qt_client.add(ProfileViewer)
