from __future__ import absolute_import, division, print_function

from qtpy import QtCore, QtWidgets

from glue.viewers.matplotlib.qt.toolbar import MatplotlibViewerToolbar
from glue.viewers.matplotlib.qt.widget import MplWidget

from glue.external.modest_image import imshow

# Import the mouse mode to make sure it gets registered
from glue.viewers.image.contrast_mouse_mode import ContrastBiasMode  # noqa

__all__ = ['StandaloneImageViewer']


class StandaloneImageViewer(QtWidgets.QMainWindow):
    """
    A simplified image viewer, without any brushing or linking,
    but with the ability to adjust contrast and resample.
    """
    window_closed = QtCore.Signal()
    _toolbar_cls = MatplotlibViewerToolbar
    tools = ['image:contrast', 'image:colormap']

    def __init__(self, image=None, wcs=None, parent=None, **kwargs):
        """
        :param image: Image to display (2D numpy array)
        :param parent: Parent widget (optional)

        :param kwargs: Extra keywords to pass to imshow
        """
        super(StandaloneImageViewer, self).__init__(parent)

        self.central_widget = MplWidget()
        self.setCentralWidget(self.central_widget)
        self._setup_axes()

        self._im = None

        self.initialize_toolbar()

        if image is not None:
            self.set_image(image=image, wcs=wcs, **kwargs)

    def _setup_axes(self):
        from glue.viewers.common.viz_client import init_mpl
        _, self._axes = init_mpl(self.central_widget.canvas.fig, axes=None, wcs=True)
        self._axes.set_aspect('equal', adjustable='datalim')

    def set_image(self, image=None, wcs=None, **kwargs):
        """
        Update the image shown in the widget
        """
        if self._im is not None:
            self._im.remove()
            self._im = None

        kwargs.setdefault('origin', 'upper')

        if wcs is not None:
            # In the following we force the color and linewith of the WCSAxes
            # frame to be restored after calling reset_wcs. This can be removed
            # once we support Astropy 1.3.1 or later.
            color = self._axes.coords.frame.get_color()
            linewidth = self._axes.coords.frame.get_linewidth()
            self._axes.reset_wcs(wcs)
            self._axes.coords.frame.set_color(color)
            self._axes.coords.frame.set_linewidth(linewidth)
            del color, linewidth

        self._im = imshow(self._axes, image, cmap='gray', **kwargs)
        self._im_array = image
        self._wcs = wcs
        self._redraw()

    @property
    def axes(self):
        """
        The Matplolib axes object for this figure
        """
        return self._axes

    def show(self):
        super(StandaloneImageViewer, self).show()
        self._redraw()

    def _redraw(self):
        self.central_widget.canvas.draw()

    def set_cmap(self, cmap):
        self._im.set_cmap(cmap)
        self._redraw()

    def mdi_wrap(self):
        """
        Embed this widget in a GlueMdiSubWindow
        """
        from glue.app.qt.mdi_area import GlueMdiSubWindow
        sub = GlueMdiSubWindow()
        sub.setWidget(self)
        self.destroyed.connect(sub.close)
        self.window_closed.connect(sub.close)
        sub.resize(self.size())
        self._mdi_wrapper = sub

        return sub

    def closeEvent(self, event):
        self.window_closed.emit()
        return super(StandaloneImageViewer, self).closeEvent(event)

    def _set_norm(self, mode):
        """ Use the `ContrastMouseMode` to adjust the transfer function """
        clip_lo, clip_hi = mode.get_clip_percentile()
        vmin, vmax = mode.get_vmin_vmax()
        stretch = mode.stretch
        self._norm.clip_lo = clip_lo
        self._norm.clip_hi = clip_hi
        self._norm.stretch = stretch
        self._norm.bias = mode.bias
        self._norm.contrast = mode.contrast
        self._norm.vmin = vmin
        self._norm.vmax = vmax
        self._im.set_norm(self._norm)
        self._redraw()

    def initialize_toolbar(self):

        # TODO: remove once Python 2 is no longer supported - see below for
        #       simpler code.

        from glue.config import viewer_tool

        self.toolbar = self._toolbar_cls(self)

        for tool_id in self.tools:
            mode_cls = viewer_tool.members[tool_id]
            mode = mode_cls(self)
            self.toolbar.add_tool(mode)

        self.addToolBar(self.toolbar)
