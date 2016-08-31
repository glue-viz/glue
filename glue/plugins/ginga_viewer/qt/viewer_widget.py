from __future__ import absolute_import, division, print_function

from ginga.misc import log
from ginga import toolkit
try:
    toolkit.use('qt')
    from ginga.gw import ColorBar
except ImportError:
    # older versions of ginga
    from ginga.qtw import ColorBar
from ginga.qtw.ImageViewCanvasQt import ImageViewCanvas

from qtpy import QtWidgets

from glue.plugins.ginga_viewer.qt.client import GingaClient
from glue.viewers.image.qt import ImageWidgetBase
from glue.viewers.common.qt.toolbar import BasicToolbar
from glue.plugins.ginga_viewer.qt.utils import ginga_graphic_to_roi

try:
    from ginga.gw import Readout
except ImportError:  # older versions of ginga
    from ginga.qtw import Readout

__all__ = ['GingaWidget']


class GingaWidget(ImageWidgetBase):

    LABEL = "Ginga Viewer"

    _toolbar_cls = BasicToolbar
    tools = ['ginga:rectangle', 'ginga:circle', 'ginga:polygon', 'ginga:pan',
             'ginga:freepan', 'ginga:rotate', 'ginga:contrast', 'ginga:cuts',
             'ginga:colormap', 'ginga:slicer', 'ginga:spectrum']

    def __init__(self, session, parent=None):

        self.logger = log.get_logger(name='ginga', level=20, null=True,
                                     # uncomment for debugging
                                     # log_stderr=True
                                     )

        self.viewer = ImageViewCanvas(self.logger, render='widget')
        self.canvas = self.viewer

        # prevent widget from grabbing focus
        try:
            self.canvas.set_enter_focus(False)
        except AttributeError:
            self.canvas.set_follow_focus(False)

        # enable interactive features
        bindings = self.canvas.get_bindings()
        bindings.enable_all(True)
        self.canvas.add_callback('none-move', self.motion_readout)
        self.canvas.add_callback('draw-event', self._apply_roi_cb)
        self.canvas.add_callback('draw-down', self._clear_roi_cb)
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
        rgbmap = self.viewer.get_rgbmap()
        self.colorbar = ColorBar.ColorBar(self.logger)
        rgbmap.add_callback('changed', self.rgbmap_cb, self.viewer)
        self.colorbar.set_rgbmap(rgbmap)

        # make coordinates/value readout
        self.readout = Readout.Readout(-1, 20)
        self.roi_tag = None

        super(GingaWidget, self).__init__(session, parent)

    def make_client(self):
        return GingaClient(self._data, self.viewer, self._layer_artist_container)

    def make_central_widget(self):

        topw = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.viewer.get_widget(), stretch=1)
        cbar_w = self.colorbar.get_widget()
        if not isinstance(cbar_w, QtWidgets.QWidget):
            # ginga wrapped widget
            cbar_w = cbar_w.get_widget()
        layout.addWidget(cbar_w, stretch=0)
        readout_w = self.readout.get_widget()
        if not isinstance(readout_w, QtWidgets.QWidget):
            # ginga wrapped widget
            readout_w = readout_w.get_widget()
        layout.addWidget(readout_w, stretch=0)
        topw.setLayout(layout)
        return topw

    def match_colorbar(self, canvas, colorbar):
        rgbmap = self.viewer.get_rgbmap()
        loval, hival = self.viewer.get_cut_levels()
        colorbar.set_range(loval, hival)
        colorbar.set_rgbmap(rgbmap)

    def rgbmap_cb(self, rgbmap, canvas):
        self.match_colorbar(canvas, self.colorbar)

    def cut_levels_cb(self, setting, tup):
        (loval, hival) = tup
        self.colorbar.set_range(loval, hival)

    def _set_roi_mode(self, name, tf):
        self.canvas.enable_draw(True)
        # XXX need better way of setting draw contexts
        self.canvas.draw_context = self
        self.canvas.set_drawtype(name, color='cyan', linestyle='dash')
        bm = self.viewer.get_bindmap()
        bm.set_mode('draw', mode_type='locked')

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
            value = self.viewer.get_data(int(data_x + 0.5), int(data_y + 0.5))

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
        bm = self.viewer.get_bindmap()
        if not tf:
            bm.reset_mode(self.viewer)
            return
        bm.set_mode(modname, mode_type='locked')
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
