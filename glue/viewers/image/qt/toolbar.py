from __future__ import absolute_import, division, print_function

from glue.viewers.matplotlib.qt.toolbar import MatplotlibViewerToolbar

from .mouse_mode import RoiClickAndDragMode


class ImageViewerToolbar(MatplotlibViewerToolbar):

    def __init__(self, *args, **kwargs):
        super(ImageViewerToolbar, self).__init__(*args, **kwargs)

    def activate_tool(self, tool):
        print("ACTIVATING!")
        super(ImageViewerToolbar, self).activate_tool(tool)

    def deactivate_tool(self, tool):
        print("DEACTIVATING!")
        super(ImageViewerToolbar, self).deactivate_tool(tool)
