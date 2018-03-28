from __future__ import absolute_import, division, print_function

import numpy as np

from qtpy import QtCore, QtWidgets

from glue.config import colormaps
from glue.viewers.common.qt.toolbar import BasicToolbar
from glue.viewers.matplotlib.qt.widget import MplWidget
from glue.viewers.image.composite_array import CompositeArray
from glue.external.modest_image import imshow
from glue.utils import defer_draw

# Import the mouse mode to make sure it gets registered
from glue.viewers.image.contrast_mouse_mode import ContrastBiasMode  # noqa

__all__ = ['StandaloneImageViewer']


class StandaloneImageViewer(QtWidgets.QMainWindow):
    """
    A simplified image viewer, without any brushing or linking,
    but with the ability to adjust contrast and resample.
    """
    window_closed = QtCore.Signal()
    _toolbar_cls = BasicToolbar
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

        self._composite = CompositeArray()
        self._composite.allocate('image')

        self._im = None

        self.initialize_toolbar()

        if image is not None:
            self.set_image(image=image, wcs=wcs, **kwargs)

    def _setup_axes(self):
        from glue.viewers.common.viz_client import init_mpl
        _, self._axes = init_mpl(self.central_widget.canvas.fig, axes=None, wcs=True)
        self._axes.set_aspect('equal', adjustable='datalim')

    @defer_draw
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

        self._composite.set('image', array=image, color=colormaps.members[0][1])
        self._im = imshow(self._axes, self._composite, **kwargs)
        self._im_array = image
        self._set_norm(self._contrast_mode)

        if 'extent' in kwargs:
            self.axes.set_xlim(kwargs['extent'][:2])
            self.axes.set_ylim(kwargs['extent'][2:])
        else:
            ny, nx = image.shape
            self.axes.set_xlim(-0.5, nx - 0.5)
            self.axes.set_ylim(-0.5, ny - 0.5)

        # FIXME: for a reason I don't quite understand, dataLim doesn't
        # get updated immediately here, which means that there are then
        # issues in the first draw of the image (the limits are such that
        # only part of the image is shown). We just set dataLim manually
        # to avoid this issue. This is also done in ImageViewer.
        self.axes.dataLim.intervalx = self.axes.get_xlim()
        self.axes.dataLim.intervaly = self.axes.get_ylim()

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
        self._composite.set('image', color=cmap)
        self._im.invalidate_cache()
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
        if self._im is not None:
            self._im.remove()
            self._im = None
        self.window_closed.emit()
        return super(StandaloneImageViewer, self).closeEvent(event)

    def _set_norm(self, mode):
        """
        Use the `ContrastMouseMode` to adjust the transfer function
        """

        pmin, pmax = mode.get_clip_percentile()

        if pmin is None:
            clim = mode.get_vmin_vmax()
        else:
            clim = (np.nanpercentile(self._im_array, pmin),
                    np.nanpercentile(self._im_array, pmax))

        stretch = mode.stretch
        self._composite.set('image', clim=clim, stretch=stretch,
                            bias=mode.bias, contrast=mode.contrast)

        self._im.invalidate_cache()
        self._redraw()

    def initialize_toolbar(self):

        from glue.config import viewer_tool

        self.toolbar = self._toolbar_cls(self)

        for tool_id in self.tools:
            mode_cls = viewer_tool.members[tool_id]
            if tool_id == 'image:contrast':
                mode = mode_cls(self, move_callback=self._set_norm)
                self._contrast_mode = mode
            else:
                mode = mode_cls(self)
            self.toolbar.add_tool(mode)

        self.addToolBar(self.toolbar)

    def set_status(self, message):
        sb = self.statusBar()
        sb.showMessage(message)
