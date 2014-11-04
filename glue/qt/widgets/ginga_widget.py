import sys
import os.path
import numpy as np

from ...external.qt.QtGui import (QAction,
                                  QToolButton, QToolBar, QIcon,
                                  QActionGroup, QWidget,
                                  QVBoxLayout, QColor, QImage, QPixmap)

from ...external.qt.QtCore import Qt, QSize

from ginga.qtw.ImageViewCanvasQt import ImageViewCanvas
from ginga.qtw import Readout, ColorBar
from ginga.misc import log
from ginga import cmap as ginga_cmap
# ginga_cmap.add_matplotlib_cmaps()

from .image_widget import ImageWidgetBase

from ...clients.ginga_client import GingaClient

from ...core import roi as roimod
from ...core.callback_property import add_callback

from ..qtutil import get_icon, nonpartial
from ...plugins.pv_slicer import PVSlicerTool
from ...plugins.spectrum_tool import SpectrumTool
from ...config import tool_registry

# Find out location of ginga module so we can some of its icons
ginga_home = os.path.split(sys.modules['ginga'].__file__)[0]
ginga_icon_dir = os.path.join(ginga_home, 'icons')

__all__ = ['GingaWidget']


class GingaWidget(ImageWidgetBase):

    LABEL = "Ginga Viewer"

    def __init__(self, session, parent=None):

        self.logger = log.get_logger(name='ginga', log_stderr=True)

        self.canvas = ImageViewCanvas(self.logger, render='widget')

        # prevent widget from grabbing focus
        self.canvas.set_follow_focus(False)
        self.canvas.enable_overlays(True)

        # enable interactive features
        bindings = self.canvas.get_bindings()
        bindings.enable_all(True)
        self.canvas.set_callback('none-move', self.motion_readout)
        self.canvas.set_callback('draw-event', self._apply_roi_cb)
        self.canvas.set_callback('draw-down', self._clear_roi_cb)
        self.canvas.enable_draw(False)
        self.canvas.enable_autozoom('off')
        self.canvas.set_zoom_algorithm('rate')
        self.canvas.set_zoomrate(1.4)

        bm = self.canvas.get_bindmap()
        bm.add_callback('mode-set', self.mode_set_cb)
        self.mode_w = None
        self.mode_actns = {}

        # Create settings and set defaults
        settings = self.canvas.get_settings()
        self.settings = settings
        settings.getSetting('cuts').add_callback('set', self.cut_levels_cb)
        settings.set(autozoom='off', autocuts='override',
                     autocenter='override')

        # make color bar, with color maps shared from ginga canvas
        self.colorbar = ColorBar.ColorBar(self.logger)
        rgbmap = self.canvas.get_rgbmap()
        rgbmap.add_callback('changed', self.rgbmap_cb, self.canvas)
        self.colorbar.set_rgbmap(rgbmap)

        # make coordinates/value readout
        self.readout = Readout.Readout(-1, -1)
        self.roi_tag = None

        super(GingaWidget, self).__init__(session, parent)

    def make_client(self):
        return GingaClient(self._data, self.canvas, self._container)

    def make_central_widget(self):

        topw = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.canvas.get_widget(), stretch=1)
        layout.addWidget(self.colorbar, stretch=0)
        layout.addWidget(self.readout.get_widget(), stretch=0)
        topw.setLayout(layout)
        return topw

    def match_colorbar(self, canvas, colorbar):
        rgbmap = canvas.get_rgbmap()
        loval, hival = canvas.get_cut_levels()
        colorbar.set_range(loval, hival, redraw=False)
        colorbar.set_rgbmap(rgbmap)

    def rgbmap_cb(self, rgbmap, canvas):
        self.match_colorbar(canvas, self.colorbar)

    def cut_levels_cb(self, setting, tup):
        (loval, hival) = tup
        self.colorbar.set_range(loval, hival)

    def make_toolbar(self):
        tb = QToolBar(parent=self)
        tb.setIconSize(QSize(25, 25))
        tb.layout().setSpacing(1)
        tb.setFocusPolicy(Qt.StrongFocus)

        agroup = QActionGroup(tb)
        agroup.setExclusive(True)
        for (mode_text, mode_icon, mode_cb) in self._mouse_modes():
            # TODO: add icons similar to the Matplotlib toolbar
            action = tb.addAction(mode_icon, mode_text)
            action.setCheckable(True)
            action.toggled.connect(mode_cb)
            agroup.addAction(action)

        action = tb.addAction(get_icon('glue_move'), "Pan")
        self.mode_actns['pan'] = action
        action.setCheckable(True)
        action.toggled.connect(lambda tf: self.mode_cb('pan', tf))
        agroup.addAction(action)
        icon = QIcon(os.path.join(ginga_icon_dir, 'hand_48.png'))
        action = tb.addAction(icon, "Free Pan")
        self.mode_actns['freepan'] = action
        action.setCheckable(True)
        action.toggled.connect(lambda tf: self.mode_cb('freepan', tf))
        agroup.addAction(action)
        icon = QIcon(os.path.join(ginga_icon_dir, 'rotate_48.png'))
        action = tb.addAction(icon, "Rotate")
        self.mode_actns['rotate'] = action
        action.setCheckable(True)
        action.toggled.connect(lambda tf: self.mode_cb('rotate', tf))
        agroup.addAction(action)
        action = tb.addAction(get_icon('glue_contrast'), "Contrast")
        self.mode_actns['contrast'] = action
        action.setCheckable(True)
        action.toggled.connect(lambda tf: self.mode_cb('contrast', tf))
        agroup.addAction(action)
        icon = QIcon(os.path.join(ginga_icon_dir, 'cuts_48.png'))
        action = tb.addAction(icon, "Cuts")
        self.mode_actns['cuts'] = action
        action.setCheckable(True)
        action.toggled.connect(lambda tf: self.mode_cb('cuts', tf))
        agroup.addAction(action)

        cmap_w = _colormap_mode(self, self.client.set_cmap)
        tb.addWidget(cmap_w)
        return tb

    def _mouse_modes(self):
        modes = []
        modes.append(("Rectangle", get_icon('glue_square'),
                      lambda tf: self._set_roi_mode('rectangle', tf)))
        modes.append(("Circle", get_icon('glue_circle'),
                      lambda tf: self._set_roi_mode('circle', tf)))
        modes.append(("Polygon", get_icon('glue_lasso'),
                      lambda tf: self._set_roi_mode('polygon', tf)))

        for tool in self._tools:
            modes += tool._get_modes(self.canvas)
            add_callback(self.client, 'display_data', tool._display_data_hook)

        return modes

    def _set_roi_mode(self, name, tf):
        self.canvas.enable_draw(True)
        # XXX need better way of setting draw contexts
        self.canvas.draw_context = self
        self.canvas.set_drawtype(name, color='cyan', linestyle='dash')
        bm = self.canvas.get_bindmap()
        bm.set_modifier('draw', modtype='locked')

    def _clear_roi_cb(self, canvas, *args):
        try:
            self.canvas.deleteObjectByTag(self.roi_tag)
        except:
            pass

    def _apply_roi_cb(self, canvas, tag):
        if self.canvas.draw_context is not self:
            return
        self.roi_tag = tag
        obj = self.canvas.getObjectByTag(self.roi_tag)
        roi = ginga_graphic_to_roi(obj)
        # delete outline
        self.canvas.deleteObject(obj, redraw=False)
        self.apply_roi(roi)

    def _tweak_geometry(self):
        super(GingaWidget, self)._tweak_geometry()

        # rgb mode not supported yet, so hide option
        self.ui.monochrome.hide()
        self.ui.rgb.hide()

    def motion_readout(self, canvas, button, data_x, data_y):
        """This method is called when the user moves the mouse around the Ginga
        canvas.
        """

        d = self.client.point_details(data_x, data_y)

        # Get the value under the data coordinates
        try:
            # value = fitsimage.get_data(data_x, data_y)
            # We report the value across the pixel, even though the coords
            # change halfway across the pixel
            value = canvas.get_data(int(data_x + 0.5), int(data_y + 0.5))

        except Exception:
            value = None

        x_lbl, y_lbl = d['labels'][0], d['labels'][1]
        # x_txt, y_txt = d['world'][0], d['world'][1]

        text = "%s  %s  X=%.2f  Y=%.2f  Value=%s" % (
            x_lbl, y_lbl, data_x, data_y, value)
        self.readout.set_text(text)

    def mode_cb(self, modname, tf):
        """This method is called when a toggle button in the toolbar is pressed
        selecting one of the modes.
        """
        bm = self.canvas.get_bindmap()
        if not tf:
            bm.reset_modifier(self.canvas)
            return
        bm.set_modifier(modname, modtype='locked')
        return True

    def mode_set_cb(self, bm, modname, mtype):
        """This method is called when a mode is selected in the viewer widget.
        NOTE: it may be called when mode_cb() is not called (for example, when
        a keypress initiates a mode); however, the converse is not true: calling
        mode_cb() will always result in this method also being called as a result.

        This logic is to insure that the toggle buttons are left in a sane state
        that reflects the current mode, however it was initiated.
        """
        if modname in self.mode_actns:
            if self.mode_w and (self.mode_w != self.mode_actns[modname]):
                self.mode_w.setChecked(False)
            self.mode_w = self.mode_actns[modname]
            self.mode_w.setChecked(True)
        elif self.mode_w:
            # keystroke turned on a mode for which we have no GUI button
            # and a GUI button is selected--unselect it
            self.mode_w.setChecked(False)
            self.mode_w = None
        return True


class ColormapAction(QAction):

    def __init__(self, label, cmap, parent):
        super(ColormapAction, self).__init__(label, parent)
        self.cmap = cmap
        pm = cmap2pixmap(cmap)
        self.setIcon(QIcon(pm))


def _colormap_mode(parent, on_trigger):

    # actions for each colormap
    acts = []
    # for label, cmap in config.colormaps:
    for label in ginga_cmap.get_names():
        cmap = ginga_cmap.get_cmap(label)
        a = ColormapAction(label, cmap, parent)
        a.triggered.connect(nonpartial(on_trigger, cmap))
        acts.append(a)

    # Toolbar button
    tb = QToolButton()
    tb.setWhatsThis("Set color scale")
    tb.setToolTip("Set color scale")
    icon = get_icon('glue_rainbow')
    tb.setIcon(icon)
    tb.setPopupMode(QToolButton.InstantPopup)
    tb.addActions(acts)

    return tb


def ginga_graphic_to_roi(obj):
    if obj.kind == 'rectangle':
        roi = roimod.RectangularROI(xmin=obj.x1, xmax=obj.x2,
                                    ymin=obj.y1, ymax=obj.y2)
    elif obj.kind == 'circle':
        roi = roimod.CircularROI(xc=obj.x, yc=obj.y,
                                 radius=obj.radius)
    elif obj.kind == 'polygon':
        vx = map(lambda xy: xy[0], obj.points)
        vy = map(lambda xy: xy[1], obj.points)
        roi = roimod.PolygonalROI(vx=vx, vy=vy)

    else:
        raise Exception("Don't know how to convert shape '%s' to a ROI" % (
            obj.kind))

    return roi


class GingaTool(object):
    label = None
    icon = None
    shape = 'polygon'
    color = 'red'
    linestyle = 'solid'

    def __init__(self, canvas):
        self.parent_canvas = canvas
        self._shape_tag = None

        self.parent_canvas.set_callback('draw-event', self._extract_callback)
        self.parent_canvas.set_callback('draw-down', self._clear_shape_cb)

    def _get_modes(self, canvas):
        return [(self.label, get_icon(self.icon), self._set_path_mode)]

    def _display_data_hook(self, data):
        # XXX need access to mode here
        pass

    def _set_path_mode(self, enable):
        self.parent_canvas.enable_draw(True)
        self.parent_canvas.draw_context = self

        self.parent_canvas.set_drawtype(self.shape, color=self.color, linestyle=self.linestyle)
        bm = self.parent_canvas.get_bindmap()
        bm.set_modifier('draw', modtype='locked')

    def _clear_shape_cb(self, *args):
        try:
            self.parent_canvas.deleteObjectByTag(self._shape_tag)
        except:
            pass

    _clear_path = _clear_shape_cb


class GingaPVSlicer(GingaTool, PVSlicerTool):
    label = 'PV Slicer'
    icon = 'glue_slice'
    shape = 'path'

    def __init__(self, widget=None):
        GingaTool.__init__(self, widget.canvas)
        PVSlicerTool.__init__(self, widget)

    def _extract_callback(self, canvas, tag):
        if self.parent_canvas.draw_context is not self:
            return

        self._shape_tag = tag
        obj = self.parent_canvas.getObjectByTag(tag)
        vx, vy = zip(*obj.points)
        return self._build_from_vertices(vx, vy)


class GingaSpectrumTool(GingaTool, SpectrumTool):
    label = 'Spectrum'
    icon = 'glue_spectrum'
    shape = 'rectangle'

    def __init__(self, widget=None):
        GingaTool.__init__(self, widget.canvas)
        SpectrumTool.__init__(self, widget)

    def _extract_callback(self, canvas, tag):
        if self.parent_canvas.draw_context is not self:
            return

        self._shape_tag = tag
        obj = self.parent_canvas.getObjectByTag(tag)
        roi = ginga_graphic_to_roi(obj)
        return self._update_from_roi(roi)

    def _setup_mouse_mode(self):
        # XXX fix this ugliness
        class Dummy:

            def clear(self):
                pass
        return Dummy()

tool_registry.add(GingaPVSlicer, GingaWidget)
tool_registry.add(GingaSpectrumTool, GingaWidget)


def cmap2pixmap(cmap, steps=50):
    """Convert a Ginga colormap into a QPixmap

    :param cmap: The colormap to use
    :type cmap: Ginga colormap instance (e.g. ginga.cmap.get_cmap('gray'))
    :param steps: The number of color steps in the output. Default=50
    :type steps: int

    :rtype: QPixmap
    """
    inds = np.linspace(0, 1, steps)
    n = len(cmap.clst) - 1
    tups = [cmap.clst[int(x * n)] for x in inds]
    rgbas = [QColor(int(r * 255), int(g * 255),
                    int(b * 255), 255).rgba() for r, g, b in tups]
    im = QImage(steps, 1, QImage.Format_Indexed8)
    im.setColorTable(rgbas)
    for i in range(steps):
        im.setPixel(i, 0, i)
    im = im.scaled(128, 32)
    pm = QPixmap.fromImage(im)
    return pm
