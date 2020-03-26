from .pv_slicer import *  # noqa


def setup():
    from glue.viewers.image.qt import ImageViewer
    from glue.plugins.tools.pv_slicer.qt import PVSlicerMode  # noqa
    ImageViewer.tools.append('slice')
