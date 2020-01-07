import os

from qtpy import QtGui, QtWidgets
from glue.utils import nonpartial
from glue.utils.qt import load_ui, cmap2pixmap
from glue.viewers.common.tool import Tool
from glue.config import viewer_tool
from glue.viewers.matplotlib.toolbar_mode import ToolbarModeBase

__all__ = ['ContrastMode', 'ColormapMode']


@viewer_tool
class ContrastMode(ToolbarModeBase):
    """
    Uses right mouse button drags to set bias and contrast, DS9-style.

    The horizontal position of the mouse sets the bias, the vertical position
    sets the contrast.

    The move_callback defaults to calling _set_norm on the viewer with the
    instance of ConstrastMode as the sole argument.
    """

    icon = 'glue_contrast'
    tool_id = 'image:contrast'
    action_text = 'Contrast'
    tool_tip = 'Adjust the bias/contrast'
    shortcut = 'B'

    def __init__(self, viewer, **kwargs):

        super(ContrastMode, self).__init__(viewer, **kwargs)

        self.bias = 0.5
        self.contrast = 1.0

        self._last = None
        self._result = None
        self._percent_lo = 1.
        self._percent_hi = 99.
        self.stretch = 'linear'
        self._vmin = None
        self._vmax = None

    def set_clip_percentile(self, lo, hi):
        """Percentiles at which to clip the data at black/white"""
        if lo == self._percent_lo and hi == self._percent_hi:
            return
        self._percent_lo = lo
        self._percent_hi = hi
        self._vmin = None
        self._vmax = None

    def get_clip_percentile(self):
        if self._vmin is None and self._vmax is None:
            return self._percent_lo, self._percent_hi
        return None, None

    def get_vmin_vmax(self):
        if self._percent_lo is None or self._percent_hi is None:
            return self._vmin, self._vmax
        return None, None

    def set_vmin_vmax(self, vmin, vmax):
        if vmin == self._vmin and vmax == self._vmax:
            return
        self._percent_hi = self._percent_lo = None
        self._vmin = vmin
        self._vmax = vmax

    def choose_vmin_vmax(self):
        dialog = load_ui('contrastlimits.ui', None,
                         directory=os.path.dirname(__file__))
        v = QtGui.QDoubleValidator()
        dialog.vmin.setValidator(v)
        dialog.vmax.setValidator(v)

        vmin, vmax = self.get_vmin_vmax()
        if vmin is not None:
            dialog.vmin.setText(str(vmin))
        if vmax is not None:
            dialog.vmax.setText(str(vmax))

        def _apply():
            try:
                vmin = float(dialog.vmin.text())
                vmax = float(dialog.vmax.text())
                self.set_vmin_vmax(vmin, vmax)
                if self._move_callback is not None:
                    self._move_callback(self)
            except ValueError:
                pass

        bb = dialog.buttonBox
        bb.button(bb.Apply).clicked.connect(_apply)
        dialog.accepted.connect(_apply)
        dialog.show()
        dialog.raise_()
        dialog.exec_()

    def move(self, event):
        """ MoveEvent. Update bias and contrast on Right Mouse button drag """
        if event.button != 3:  # RMB drag only
            return
        x, y = event.x, event.y
        dx, dy = self._axes.figure.canvas.get_width_height()
        x = 1.0 * x / dx
        y = 1.0 * y / dy

        self.bias = x
        self.contrast = (1 - y) * 10

        super(ContrastMode, self).move(event)

    def menu_actions(self):
        result = []

        a = QtWidgets.QAction("minmax", None)
        a.triggered.connect(nonpartial(self.set_clip_percentile, 0, 100))
        result.append(a)

        a = QtWidgets.QAction("99%", None)
        a.triggered.connect(nonpartial(self.set_clip_percentile, 1, 99))
        result.append(a)

        a = QtWidgets.QAction("95%", None)
        a.triggered.connect(nonpartial(self.set_clip_percentile, 5, 95))
        result.append(a)

        a = QtWidgets.QAction("90%", None)
        a.triggered.connect(nonpartial(self.set_clip_percentile, 10, 90))
        result.append(a)

        rng = QtWidgets.QAction("Set range...", None)
        rng.triggered.connect(nonpartial(self.choose_vmin_vmax))
        result.append(rng)

        a = QtWidgets.QAction("", None)
        a.setSeparator(True)
        result.append(a)

        a = QtWidgets.QAction("linear", None)
        a.triggered.connect(nonpartial(setattr, self, 'stretch', 'linear'))
        result.append(a)

        a = QtWidgets.QAction("log", None)
        a.triggered.connect(nonpartial(setattr, self, 'stretch', 'log'))
        result.append(a)

        a = QtWidgets.QAction("sqrt", None)
        a.triggered.connect(nonpartial(setattr, self, 'stretch', 'sqrt'))
        result.append(a)

        a = QtWidgets.QAction("asinh", None)
        a.triggered.connect(nonpartial(setattr, self, 'stretch', 'arcsinh'))
        result.append(a)

        for r in result:
            if r is rng:
                continue
            if self._move_callback is not None:
                r.triggered.connect(nonpartial(self._move_callback, self))

        return result


class ColormapAction(QtWidgets.QAction):

    def __init__(self, label, cmap, parent):
        super(ColormapAction, self).__init__(label, parent)
        self.cmap = cmap
        pm = cmap2pixmap(cmap)
        self.setIcon(QtGui.QIcon(pm))


@viewer_tool
class ColormapMode(Tool):
    """
    A tool to change the colormap used in a viewer.

    This calls a ``set_cmap`` method on the viewer, which should take the name
    of the colormap as the sole argument.
    """

    icon = 'glue_rainbow'
    tool_id = 'image:colormap'
    action_text = 'Set color scale'
    tool_tip = 'Set color scale'

    def menu_actions(self):
        from glue import config
        acts = []
        for label, cmap in config.colormaps:
            a = ColormapAction(label, cmap, self.viewer)
            a.triggered.connect(nonpartial(self.viewer.set_cmap, cmap))
            acts.append(a)
        return acts
